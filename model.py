import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from utils import to_var
from model_utils import dynamic_rnn

class SentenceVAE(nn.Module):

    def __init__(self, vocab_size, embedding_size, rnn_type, hidden_size, word_dropout, embedding_dropout, latent_size,
                sos_idx, eos_idx, pad_idx, unk_idx, max_sequence_length, num_layers=1, bidirectional=False, 
                cond_vocab_size=None, cond_embedding_size=None, cond_hidden_size=None,
                ):
        # WARNING: 分離できるのに共通なものがあるので注意
        # Enc, Decで共通          ：hidden_size
        # Enc, Cond-Encで共通     ：latent_size, 
        # Enc, Dec, Cond-Encで共通：num_layers, bidirectional

        super().__init__()
        self.tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

        self.max_sequence_length = max_sequence_length
        self.sos_idx = sos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx

        self.latent_size = latent_size

        self.is_conditional = self.is_conditional(cond_vocab_size, cond_embedding_size, cond_hidden_size)
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

        if rnn_type == 'rnn':
            rnn = nn.RNN
        elif rnn_type == 'gru':
            rnn = nn.GRU
        # elif rnn_type == 'lstm':
        #     rnn = nn.LSTM
        else:
            raise ValueError()

        self.encoder_rnn = rnn(embedding_size, self.hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)
        self.decoder_rnn = rnn(embedding_size, self.hidden_size, num_layers=num_layers, bidirectional=self.bidirectional, batch_first=True)

        # scalar for bidirectional mode
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
        self.latent2hidden = nn.Linear(self.dec_before_input_size , self.hidden_size * self.hidden_factor)

        self.outputs2vocab = nn.Linear(self.hidden_size * (2 if bidirectional else 1), vocab_size)


    def is_conditional(self, cond_vocab_size, cond_embedding_size, cond_hidden_size):
        param_dict = {k:v for k,v in locals().items() if 'cond' in k}
        valid_param_count = sum([bool(v) for k,v in param_dict.items()])
        if valid_param_count == len(param_dict):
            return True
        elif valid_param_count == 0:
            return False
        else:
            raise ValueError(f'invalid conditional params: {param_dict}')


    def encode(self, input_sequence, length):
        batch_size = input_sequence.size(0)
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_sequence = input_sequence[sorted_idx]

        # ENCODER
        input_embedding = self.embedding(input_sequence)

        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.data.tolist(), batch_first=True)

        _, hidden = self.encoder_rnn(packed_input)

        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            hidden = hidden.view(batch_size, self.hidden_size*self.hidden_factor)
        else:
            hidden = hidden.squeeze()

        # REPARAMETERIZATION
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)

        z = to_var(torch.randn([batch_size, self.latent_size]))
        z = z * std + mean
        return mean, std, z
        
    def _reshape_hidden_for_bidirection(self, hidden, batch_size, hidden_size):
        if self.bidirectional or self.num_layers > 1:
            # flatten hidden state
            return hidden.view(batch_size, hidden_size*self.hidden_factor)
        else:
            return hidden.squeeze()


    def _reparametarize(self, mean, std, batch_size, latent_size):
        # REPARAMETERIZATION
        z = to_var(torch.randn([batch_size, latent_size]))
        z = z * std + mean
        return z

        
    def forward(self, input_sequence, length, cond_sequence=None, cond_length=None):
        assert self.is_conditional == (cond_sequence is not None)
        if self.is_conditional:
            assert cond_length is not None, '必要'
            assert input_sequence.size(0) == cond_sequence.size(0), '同じバッチサイズ'
        batch_size = input_sequence.size(0)

        # --------------- ENCODER -------------------
        input_embedding = self.embedding(input_sequence)
        hidden = dynamic_rnn(self.encoder_rnn, input_embedding, length)
        hidden = self._reshape_hidden_for_bidirection(hidden, batch_size, self.hidden_size)        

        # Conditional-Encoder
        if self.is_conditional:
            cond_embedding = self.cond_embedding(cond_sequence)
            cond_hidden = dynamic_rnn(self.cond_encoder_rnn, cond_embedding, cond_length)
            cond_hidden = self._reshape_hidden_for_bidirection(cond_hidden, batch_size, self.cond_hidden_size)
            hidden = torch.cat([hidden, cond_hidden], dim=1)

        # Encoder Latent
        mean = self.hidden2mean(hidden)
        logv = self.hidden2logv(hidden)
        std = torch.exp(0.5 * logv)
        z = self._reparametarize(mean, std, batch_size, self.latent_size)

        # Conditional-Encoder Latent
        if self.is_conditional:
            cond_mean = self.cond_hidden2mean(cond_hidden)
            cond_logv = self.cond_hidden2logv(cond_hidden)
            cond_std = torch.exp(0.5 * cond_logv)
            cond_z = self._reparametarize(cond_mean, cond_std, batch_size, self.latent_size)

        # --------------- DECODER -------------------

        if self.is_conditional:
            z = torch.cat([z, cond_hidden], dim=1)

        hidden = self.latent2hidden(z)

        if self.bidirectional or self.num_layers > 1:
            # unflatten hidden state
            hidden = hidden.view(self.hidden_factor, batch_size, self.hidden_size)
        else:
            hidden = hidden.unsqueeze(0)

        # decoder input
        if self.word_dropout_rate > 0:
            # randomly replace decoder input with <unk>
            prob = torch.rand(input_sequence.size())
            if torch.cuda.is_available():
                prob=prob.cuda()
            prob[(input_sequence.data - self.sos_idx) * (input_sequence.data - self.pad_idx) == 0] = 1
            decoder_input_sequence = input_sequence.clone()
            decoder_input_sequence[prob < self.word_dropout_rate] = self.unk_idx
            input_embedding = self.embedding(decoder_input_sequence)

        input_embedding = self.embedding_dropout(input_embedding)
        
        sorted_lengths, sorted_idx = torch.sort(length, descending=True)
        input_embedding = input_embedding[sorted_idx]
        packed_input = rnn_utils.pack_padded_sequence(input_embedding, sorted_lengths.tolist(), batch_first=True)

        # decoder forward pass
        outputs, _ = self.decoder_rnn(packed_input, hidden)

        # process outputs
        padded_outputs = rnn_utils.pad_packed_sequence(outputs, batch_first=True)[0]
        padded_outputs = padded_outputs.contiguous()
        _,reversed_idx = torch.sort(sorted_idx)
        padded_outputs = padded_outputs[reversed_idx]
        b,s,_ = padded_outputs.size()

        # project outputs to vocab
        logp = nn.functional.log_softmax(self.outputs2vocab(padded_outputs.view(-1, padded_outputs.size(2))), dim=-1)
        logp = logp.view(b, s, self.embedding.num_embeddings)

        if self.is_conditional:
            return logp, mean, logv, z, cond_mean ,cond_logv, cond_z
        else:
            return logp, mean, logv, z


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

            input_embedding = self.embedding(input_sequence)

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
