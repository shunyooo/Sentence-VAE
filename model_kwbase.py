import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from ptb import PAD_INDEX
import numpy as np

NLL = torch.nn.NLLLoss(reduction='sum', ignore_index=PAD_INDEX)

class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False, 
                use_bow_loss=True, bow_hidden_size=None, out_vocab_size=None,
                cond_vocab_size=None, cond_embedding_size=None, cond_hidden_size=None,
                ):
        """
        Extention
        ■ bow loss : use_bow_loss, bow_hidden_size で指定
        ■ diff vocab input : InputとOutputでvocabが違う場合に指定. forwardのInput等に影響あり
        """

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        # tgtのmax_sequence_lengthで良さそう
        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.is_conditional = is_conditional(cond_vocab_size, cond_embedding_size, cond_hidden_size)
        if self.is_conditional:
            self.cond_hidden_size = cond_hidden_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        hidden_size = hidden_size if not self.is_conditional else hidden_size + cond_hidden_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.is_inout_vocab_diff = out_vocab_size is not None
        if self.is_inout_vocab_diff:
            self.decoder_embedding = nn.Embedding(out_vocab_size, embedding_size)
        else:
            self.decoder_embedding = self.embedding
            out_vocab_size = vocab_size

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        if self.is_conditional:
            # Conditional Encoder
            self.cond_embedding = nn.Embedding(cond_vocab_size, cond_embedding_size)
            self.cond_encoder_rnn = rnn(cond_embedding_size, cond_hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
            # Prior latent
            self.cond_hidden2mean = nn.Linear(cond_hidden_size * self.hidden_factor, latent_size)
            self.cond_hidden2logv = nn.Linear(cond_hidden_size * self.hidden_factor, latent_size)

        # Encoder(recogition) latent
        # ガウス分布のパラメタ推定NNへの入力サイズ
        self.enc_latent_input_size = (self.hidden_size + (self.cond_hidden_size if self.is_conditional else 0)) * self.hidden_factor
        self.hidden2mean = nn.Linear(self.enc_latent_input_size, latent_size)
        self.hidden2logv = nn.Linear(self.enc_latent_input_size, latent_size)
        # デコーダの隠れサイズへ調整する用NNへの入力サイズ
        self.dec_before_input_size = latent_size + (self.cond_hidden_size if self.is_conditional else 0)
        self.latent2hidden = nn.Linear(self.dec_before_input_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), out_vocab_size)

        self.use_bow_loss = use_bow_loss
        if use_bow_loss:
            assert bow_hidden_size is not None
            self.latent2bow = nn.Sequential(
                nn.Linear(self.dec_before_input_size, bow_hidden_size),
                nn.Tanh(),
                nn.Dropout(p=embedding_dropout),
                nn.Linear(bow_hidden_size, out_vocab_size)
            )



    def forward(self, input_sequence, input_length, out_sequence=None, out_length=None,
        cond_sequence=None, cond_length=None):
        # --------------- CHECK -------------------
        assert (out_sequence is not None) == self.is_inout_vocab_diff, f'out_sequence: {out_sequence is not None} == inout_vocab_diff: {self.is_inout_vocab_diff} であるべきです.'
        assert (cond_sequence is not None and cond_length is not None) == self.is_conditional, f'cond_sequence: {cond_sequence is not None}, cond_length: {cond_length is not None} == is_conditional: {self.is_conditional} であるべきです.'

        res_dict = {}
        # --------- CONDITIONAL ENCODE ------------
        cond_hidden = None
        if self.is_conditional:
            cond_hidden, cond_mean, cond_logv, cond_z = self.encode_condition(cond_sequence, cond_length)
            res_dict.update({'cond_hidden': cond_hidden, 'cond_mean': cond_mean, 'cond_logv': cond_logv, 'cond_z': cond_z,})

        # --------------- ENCODE ------------------
        hidden = self.encode(input_sequence, input_length)

        # ---------- CONCAT CONDITIONAL -----------
        if cond_hidden is not None:
            hidden = torch.cat([hidden, cond_hidden], dim=1)

        # ----------- REPARAMETERIZE --------------
        mean, logv, z = self.hidden2latent(hidden)

        # --------------- DECODE ------------------
        dec_input = z

        if self.is_conditional:
            dec_input = torch.cat([z, cond_hidden], dim=1)

        if out_sequence is None:
            out_sequence, out_length = input_sequence, input_length

        logp = self.decode_batch(dec_input, out_sequence, out_length)

        res_dict.update({'logp': logp, 'mean': mean, 'logv': logv, 'z': z, 'dec_input': dec_input})
        return res_dict


    def encode(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        _, reversed_idx = torch.sort(sorted_idx)

        # -------------------- ENCODER ------------------------
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)
        hidden = self._reshape_hidden_for_bidirection(hidden, batch_size, self.hidden_size)
        hidden = hidden[reversed_idx]

        assert hidden.size(0) == batch_size, hidde.size(1) == self.hidden_size
        return hidden 


    def hidden2latent(self, hidden):
        # --------------- REPARAMETERIZATION ------------------
        batch_size = hidden.size(0)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        return mean, logv, z


    def encode_condition(self, cond_sequence, cond_length):
        assert self.is_conditional is not None and cond_sequence is not None and cond_length is not None
        batch_size = cond_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(cond_length, descending=True)
        cond_sequence = cond_sequence[sorted_idx]
        _, reversed_idx = torch.sort(sorted_idx)

        # -------------------- CONDITIONAL ENCODER ------------------------
        cond_embedding = self.cond_embedding(cond_sequence)
        packed_input = rnn_utils.pack_padded_sequence(cond_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.cond_encoder_rnn(packed_input)
        hidden = self._reshape_hidden_for_bidirection(hidden, batch_size, self.cond_hidden_size)
        hidden = hidden[reversed_idx]

        assert hidden.size(0) == batch_size, hidde.size(1) == self.cond_hidden_size

        # REPARAMETERIZATION
        mean = self.cond_hidden2mean(hidden)
        logv = self.cond_hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean

        return hidden, mean, logv, z


    def _reshape_hidden_for_bidirection(self, hidden, batch_size, hidden_size):
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            return hidden.view(batch_size, hidden_size*self.hidden_factor)
        else:
            return hidden.view(batch_size, hidden_size)


    def decode_batch(self, latent, out_sequence, out_length):
        # latent: padded. [batch_size, max_sentence_length]
        # out_length: [batch_size]
        # init_state: [batch_size, latent_size]

        batch_size = out_sequence.size(0)

        # -------------------- DECODER --------------------
        sorted_lengths, sorted_idx = torch.sort(out_length, descending=True)
        out_sequence, latent = out_sequence[sorted_idx], latent[sorted_idx]
        _, reversed_idx = torch.sort(sorted_idx)
        out_embedding = self.decoder_embedding(out_sequence)

        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(out_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(out_sequence.data - self.sos_idx) * (out_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = out_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            out_embedding = self.decoder_embedding(decoder_input_sequence)
        out_embedding = self.embedding_dropout(out_embedding)
        packed_input = rnn_utils.pack_padded_sequence(out_embedding, sorted_lengths.data.tolist(), batch_first=True)

        hidden = self.latent2hidden(latent)
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.decoder_embedding.num_embeddings)
        return logp

    @staticmethod
    def _kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    @staticmethod
    def gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar) 
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld


    def loss(self, logp, target, length, mean, logv, anneal_function, step, k, x0, bow_input=None, 
        cond_mean=None, cond_logv=None):
        batch_size = target.size(0)

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data].contiguous()
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target.view(-1))

        # KL Divergence
        if self.is_conditional:
            KL_loss = self.gaussian_kld(mean, logv, cond_mean, cond_logv)
            KL_loss = torch.mean(KL_loss)
        else:
            KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())

        KL_weight = self._kl_anneal_function(anneal_function, step, k, x0)

        loss = (NLL_loss + KL_weight * KL_loss)/batch_size
        
        loss_dict = {
            'loss': loss,
            'NLL_loss': NLL_loss,
            'KL_weight': KL_weight,
            'KL_loss': KL_loss,
        }
        
        # BOW loss
        if 'use_bow_loss' in self.__dict__.keys() and self.use_bow_loss:
            assert bow_input is not None, 'bow loss を使う場合は bow予測モデルへの入力 を input してください'
            target_mask = torch.sign(target).detach().float()
            bow_logit = self.latent2bow(bow_input) # [batch_size, vocab_size]
            # 各出現単語のlog_softmaxを出す. [batch_size, max_length_in_batch]
            bow_loss1 = -nn.functional.log_softmax(bow_logit, dim=1).gather(1, target) * target_mask
            bow_loss = torch.sum(bow_loss1, 1)
            avg_bow_loss = torch.mean(bow_loss) # /batch_size
            loss += avg_bow_loss
            loss_dict['avg_bow_loss'] = avg_bow_loss

        return loss_dict


    def inference(self, n=4, z=None):

        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)

        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=self.tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=self.tensor()).bool()

        running_seqs = torch.arange(0, batch_size, out=self.tensor()).long() # idx of still generating sequences with respect to current loop

        generations = self.tensor(batch_size, self.max_sequence_length).fill_(self.pad_idx).long()

        t=0
        while(t<self.max_sequence_length and len(running_seqs)>0):
            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(self.sos_idx).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = self.decoder_embedding(input_sequence)

            output, hidden = self.decoder_rnn(input_embedding, hidden)

            logits = self.outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != self.eos_idx).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != self.eos_idx).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence.view(-1)[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=self.tensor()).long()

            t += 1

        return generations, z

    def _sample(self, dist, mode='greedy'):

        if mode == 'greedy':
            _, sample = torch.topk(dist, 1, dim=-1)
        sample = sample.squeeze()

        return sample

    def _save_sample(self, save_to, sample, running_seqs, t):
        # select only still running
        running_latest = save_to[running_seqs]
        # update token at position t
        running_latest[:,t] = sample.data
        # save back
        save_to[running_seqs] = running_latest

        return save_to


def is_conditional(cond_vocab_size, cond_embedding_size, cond_hidden_size):
    param_dict = {k:v for k,v in locals().items() if 'cond' in k}
    valid_param_count = sum([bool(v) for k,v in param_dict.items()])
    if valid_param_count == len(param_dict):
        return True
    elif valid_param_count == 0:
        return False
    else:
        raise ValueError(f'invalid conditional params: {param_dict}')