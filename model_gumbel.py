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
                use_bow_loss=True, bow_hidden_size=None, is_gumbel=False, gumbel_tau=None):
        """
        Extention
        ■ bow loss : use_bow_loss, bow_hidden_size で指定
        """

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.decoder_embedding = self.embedding
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
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

        encoded_hidden_size = self.hidden_size * self.hidden_factor
        before_latent_input_size = encoded_hidden_size
        self.is_gumbel = is_gumbel
        if self.is_gumbel:
            assert gumbel_tau is not None
            self.gumbel_tau = gumbel_tau
            self.hidden2gumbel = nn.Linear(encoded_hidden_size, vocab_size)
            before_latent_input_size = embedding_size

        # Encoder(recogition) latent
        # ガウス分布のパラメタ推定NNへの入力サイズ
        self.hidden2mean = nn.Linear(before_latent_input_size, latent_size)
        self.hidden2logv = nn.Linear(before_latent_input_size, latent_size)
        # デコーダの隠れサイズへ調整する用NNへの入力サイズ
        self.before_dec_input_size = latent_size
        self.latent2hidden = nn.Linear(self.before_dec_input_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), out_vocab_size)

        self.use_bow_loss = use_bow_loss
        if use_bow_loss:
            assert bow_hidden_size is not None
            self.latent2bow = nn.Sequential(
                nn.Linear(self.before_dec_input_size, bow_hidden_size),
                nn.Tanh(),
                nn.Dropout(p=embedding_dropout),
                nn.Linear(bow_hidden_size, out_vocab_size)
            )



    def forward(self, input_sequence, input_length):
        res_dict = {}

        # --------------- ENCODE ------------------
        mean, logv, z = self.encode(input_sequence, input_length)

        # --------------- DECODE ------------------
        out_sequence, out_length = input_sequence, input_length
        dec_input = z
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

        # GUMBEL SOFTMAX
        if self.is_gumbel:
            gumbel_bow = self.hidden2gumbel(hidden)
            gumbel_bow = nn.functional.gumbel_softmax(gumbel_bow, tau=self.gumbel_tau)
            hidden = torch.matmul(gumbel_bow, self.embedding.weight)

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        return mean, logv, z


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


    def loss(self, logp, target, length, mean, logv, anneal_function, step, k, x0, bow_input=None):
        batch_size = target.size(0)

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).data].contiguous()
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target.view(-1))

        # KL Divergence
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