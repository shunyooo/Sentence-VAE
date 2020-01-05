import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from .model_utils import to_var
from constant import PAD_INDEX, UNK_INDEX, SOS_INDEX, EOS_INDEX
import numpy as np

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
LongTensor = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor
NLL = torch.nn.NLLLoss(reduction='sum', ignore_index=PAD_INDEX)

class POSVAE(nn.Module):

    def __init__(self, vocab_size, pos_vocab_size, embedding_size, pos_embedding_size, rnn_type, hidden_size, 
        word_dropout, embedding_dropout, latent_size, tgt_max_sequence_length, pos_max_sequence_length, 
        num_layers=1, bidirectional=False, ):
        """
        Extention
        ■ bow loss : use_bow_loss, bow_hidden_size で指定
        ■ diff vocab input : InputとOutputでvocabが違う場合に指定. forwardのInput等に影響あり
        """

        super().__init__()

        # decode時にミニバッチを一気に入れる用のmax length
        self.tgt_max_sequence_length = tgt_max_sequence_length
        self.pos_max_sequence_length = pos_max_sequence_length

        self.rnn_type = rnn_type 
        self.bidirectional = bidirectional
        self.num_layers = num_layers

        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.word_dropout_rate = word_dropout
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)

        self.pos_embedding = nn.Embedding(pos_vocab_size, pos_embedding_size)

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.hidden_factor = (2 if bidirectional else 1) * num_layers

        # ---- POS GENERATOR MODEL ------
        self.latent_size = latent_size
        self.encoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.pos_decoder_rnn = rnn(pos_embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        # Encoder(recogition) latent
        # ガウス分布のパラメタ推定NNへの入力サイズ
        self.before_latent_input_size = self.hidden_size * self.hidden_factor
        self.hidden2mean = nn.Linear(self.before_latent_input_size, latent_size)
        self.hidden2logv = nn.Linear(self.before_latent_input_size, latent_size)
        # デコーダの隠れサイズへ調整する用NNへの入力サイズ
        self.latent2pos_decoder_hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2pos = nn.Linear(hidden_size * (2 if bidirectional else 1), pos_vocab_size)

        # ---- TEXT GENERATOR MODEL ------
        self.pos_encoder_rnn = rnn(pos_embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.text_decoder_rnn = rnn(embedding_size, hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.latent2pos_encoder_hidden = nn.Linear(latent_size, hidden_size * self.hidden_factor)
        self.outputs2vocab = nn.Linear(hidden_size * (2 if bidirectional else 1), vocab_size)


    def forward(self, text_sequence, text_length, pos_sequence, pos_length):
        assert EOS_INDEX not in text_sequence, 'RNNの入力系列には<EOS>は入れてはいけない'
        assert SOS_INDEX in text_sequence 

        res_text2pos = self.forward_text2pos(text_sequence, text_length, pos_sequence, pos_length)
        z = res_text2pos['z']
        res_pos2text = self.forward_pos2text(text_sequence, text_length, pos_sequence, pos_length, z)

        res_dict = {}
        res_dict.update(res_text2pos)
        res_dict.update(res_pos2text)
        return res_dict


    def forward_text2pos(self, text_sequence, text_length, pos_sequence, pos_length):
        # Text → z → POS
        # Text Encode
        text_hidden = self.text_encode(text_sequence, text_length)
        # Reparametarize
        mean, logv, z = self.hidden2latent(text_hidden)
        # POS Decode
        pos_logp = self.pos_decode_batch(z, pos_sequence, pos_length)
        return {'text_hidden': text_hidden, 'mean': mean, 'logv': logv, 'z': z, 'pos_logp': pos_logp}


    def forward_pos2text(self, text_sequence, text_length, pos_sequence, pos_length, z):
        # z + POS → Text
        init_state = self.latent2pos_encoder_init_hidden(z)
        # POS Encode
        # NOTE: DecodeされたPOSを入力するためには length が必要になり、その対応方法がわからないため、一旦学習を分離する。
        # TODO: end2endでの学習
        pos_hidden = self.pos_encode(pos_sequence, pos_length, init_state)
        # Text Decode
        text_logp = self.text_decode_batch(pos_hidden, text_sequence, text_length)
        return {'init_state': init_state, 'pos_hidden': pos_hidden, 'text_logp': text_logp}


    def _sort_emb_encode_sequence(self, input_sequence, length, embedding, rnn, hidden_size, init_state=None):
        # Encode用：文長ソート→埋め込み→Encode→最終隠れ層を返却
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]
        _, reversed_idx = torch.sort(sorted_idx)
        input_embedding = embedding(input_sequence)
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        if init_state is not None:
            sorted_init_state = init_state[:, sorted_idx].contiguous()
            _, hidden = rnn(packed_input, sorted_init_state)
        else:
            _, hidden = rnn(packed_input)

        hidden = self._reshape_hidden_for_bidirection(hidden, batch_size, self.hidden_size)
        hidden = hidden[reversed_idx]
        assert hidden.size(0) == batch_size, hidde.size(1) == hidden_size
        return hidden


    def text_encode(self, input_sequence, length):
        return self._sort_emb_encode_sequence(input_sequence, length, 
            self.embedding, self.encoder_rnn, self.hidden_size)


    def pos_encode(self, input_sequence, length, init_state):
        return self._sort_emb_encode_sequence(input_sequence, length, 
            self.pos_embedding, self.pos_encoder_rnn, self.hidden_size, init_state=init_state)


    def hidden2latent(self, hidden):
        # --------------- REPARAMETERIZATION ------------------
        batch_size = hidden.size(0)
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        return mean, logv, z


    def latent2pos_encoder_init_hidden(self, z):
        # z から pos encoder の初期隠れ層への写像
        pos_encoder_init_hidden = self.latent2pos_encoder_hidden(z)
        b, h = pos_encoder_init_hidden.shape
        return pos_encoder_init_hidden.view(1, b, h)


    def _reshape_hidden_for_bidirection(self, hidden, batch_size, hidden_size):
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            return hidden.view(batch_size, hidden_size*self.hidden_factor)
        else:
            return hidden.view(batch_size, hidden_size)


    def _decode_batch(self, latent, out_sequence, out_length,
        embedding, decoder_rnn, outputs2vocab, latent2hidden=None):
        # out_sequence: Decoderに対するInput
        # latent: padded. [batch_size, max_sentence_length]
        # out_length: [batch_size]
        # init_state: [batch_size, latent_size]

        batch_size = out_sequence.size(0)

        # -------------------- DECODER --------------------
        sorted_lengths, sorted_idx = torch.sort(out_length, descending=True)
        out_sequence, latent = out_sequence[sorted_idx], latent[sorted_idx]
        _, reversed_idx = torch.sort(sorted_idx)
        out_embedding = embedding(out_sequence)

        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(out_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(out_sequence.data - self.sos_idx) * (out_sequence.data - PAD_INDEX) == 0] = 1
            decoder_input_sequence = out_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            out_embedding = embedding(decoder_input_sequence)
        out_embedding = self.embedding_dropout(out_embedding)

        # pad(=0)を除去したinputに変換
        packed_input = rnn_utils.pack_padded_sequence(out_embedding, sorted_lengths.data.tolist(), batch_first=True)

        if latent2hidden is not None:
            hidden = latent2hidden(latent)
        else:
            hidden = latent

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder forward pass
        outputs, _ = decoder_rnn(packed_input, hidden)

        # process outputs
        # 0埋めして系列長を揃える → (batch_size, out_length.max(), hidden_size)
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        padded_outputs = padded_outputs[reversed_idx]
        batch_size, max_seq_length ,hidden_size = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(outputs2vocab(padded_outputs.view(-1, hidden_size)), dim=-1)
        logp = logp.view(batch_size, max_seq_length, embedding.num_embeddings)
        return logp


    def pos_decode_batch(self, latent, out_sequence, out_length):
        # Text Encode から POS Decode
        return self._decode_batch(latent, out_sequence, out_length, 
            self.pos_embedding, self.pos_decoder_rnn, self.outputs2pos,
            latent2hidden=self.latent2pos_decoder_hidden)


    def text_decode_batch(self, hidden, out_sequence, out_length):
        # POS Encode から Text Decode
        return self._decode_batch(hidden, out_sequence, out_length, 
            self.embedding, self.text_decoder_rnn, self.outputs2vocab)


    @staticmethod
    def _kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)


    @staticmethod
    def gaussian_kl_divergence(recog_mu, recog_logvar, prior_mu, prior_logvar):
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar) 
                    - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                    - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld


    @staticmethod
    def standard_gaussian_kl_divergence(mean, logv):
        return -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())


    @staticmethod
    def recon_loss(logp, target, target_length):
        assert EOS_INDEX in target and SOS_INDEX not in target
        # 余計なpadding(=0)を除去
        target = target[:, :torch.max(target_length).data].contiguous()
        logp = logp.view(-1, logp.size(2))
        NLL_loss = NLL(logp, target.view(-1))
        return NLL_loss


    def loss(self, out_dict, label_dict, step, args):
        batch_size = label_dict['tgt_target'].size(0)
        POS_NLL_loss = self.recon_loss(out_dict['pos_logp'], label_dict['pos_target'], label_dict['pos_length'])
        TEXT_NLL_loss = self.recon_loss(out_dict['text_logp'], label_dict['tgt_target'], label_dict['tgt_length'])
        KL_loss = self.standard_gaussian_kl_divergence(out_dict['mean'], out_dict['logv'])
        KL_weight = self._kl_anneal_function(args.anneal_function, step, args.k, args.x0)
        loss = (POS_NLL_loss + TEXT_NLL_loss + KL_weight * KL_loss)/batch_size
        return {
            'loss': loss,
            'POS_NLL_loss': POS_NLL_loss,
            'TEXT_NLL_loss': TEXT_NLL_loss,
            'KL_weight': KL_weight,
            'KL_loss': KL_loss,
        }


    @staticmethod
    def target2input(target):
        # EOSの除去
        batch_size, seq_size = target.shape
        target_no_eos = target.clone()
        target_no_eos[target_no_eos == EOS_INDEX] = 0
        # SOSの追加
        sos_ids = LongTensor(batch_size, 1).fill_(SOS_INDEX)
        new_input = torch.cat([sos_ids, target_no_eos], dim=1)[:, :seq_size]
        # lengthの算出
        col, idx = (target == EOS_INDEX).nonzero(as_tuple=True)
        lengths = LongTensor(batch_size).fill_(seq_size)
        lengths[col] = idx + 1
        return new_input, lengths


    def pos_inference(self, n=4, z=None):
        if z is None:
            batch_size = n
            z = to_var(torch.randn([batch_size, self.latent_size]))
        else:
            batch_size = z.size(0)
        hidden = self.latent2pos_decoder_hidden(z)
        return self._inference(batch_size, hidden, self.pos_embedding, self.pos_decoder_rnn, self.outputs2pos, self.pos_max_sequence_length)


    def text_inference(self, pos_sequence, pos_length, z):
        # z + pos_sequence → hidden
        # Encode
        assert pos_sequence.size(0) == pos_length.size(0) == z.size(0)
        batch_size = pos_sequence.size(0)
        pos_embedding = self.pos_embedding(pos_sequence)
        init_state = self.latent2pos_encoder_init_hidden(z)
        pos_hidden = self.pos_encode(pos_sequence, pos_length, init_state)
        # Decode
        return self._inference(batch_size, pos_hidden, self.embedding, self.text_decoder_rnn, self.outputs2vocab, self.tgt_max_sequence_length)


    def _inference(self, batch_size, hidden, decoder_embedding, decoder_rnn, outputs2vocab, max_sequence_length):
        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        hidden = hidden.unsqueeze(0)

        # required for dynamic stopping of sentence generation
        sequence_idx = torch.arange(0, batch_size, out=Tensor()).long() # all idx of batch
        sequence_running = torch.arange(0, batch_size, out=Tensor()).long() # all idx of batch which are still generating
        sequence_mask = torch.ones(batch_size, out=Tensor()).bool()

        running_seqs = torch.arange(0, batch_size, out=Tensor()).long() # idx of still generating sequences with respect to current loop

        generations = Tensor(batch_size, max_sequence_length).fill_(PAD_INDEX).long()

        t=0
        while(t<max_sequence_length and len(running_seqs)>0):
            if t == 0:
                input_sequence = to_var(torch.Tensor(batch_size).fill_(SOS_INDEX).long())

            input_sequence = input_sequence.unsqueeze(1)

            input_embedding = decoder_embedding(input_sequence)

            output, hidden = decoder_rnn(input_embedding, hidden)

            logits = outputs2vocab(output)

            input_sequence = self._sample(logits)

            # save next input
            generations = self._save_sample(generations, input_sequence, sequence_running, t)

            # update gloabl running sequence
            sequence_mask[sequence_running] = (input_sequence != EOS_INDEX).data
            sequence_running = sequence_idx.masked_select(sequence_mask)

            # update local running sequences
            running_mask = (input_sequence != EOS_INDEX).data
            running_seqs = running_seqs.masked_select(running_mask)

            # prune input and hidden state according to local update
            if len(running_seqs) > 0:
                input_sequence = input_sequence.view(-1)[running_seqs]
                hidden = hidden[:, running_seqs]

                running_seqs = torch.arange(0, len(running_seqs), out=Tensor()).long()

            t += 1

        return generations

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