import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict
from ptb import SOS_INDEX, UNK_INDEX, PAD_INDEX, EOS_INDEX

class AttributeDict(object):
    def __init__(self, obj):
        self.obj = obj

    def __getstate__(self):
        return self.obj.items()

    def __setstate__(self, items):
        if not hasattr(self, 'obj'):
            self.obj = {}
        for key, val in items:
            self.obj[key] = val

    def __getattr__(self, name):
        if name in self.obj:
            return self.obj.get(name)
        else:
            return None

    def fields(self):
        return self.obj

    def keys(self):
        return self.obj.keys()
    
    def __repr__(self):
        return f'<AttrDict{self.obj}>'

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:

            if word_id.item() == pad_idx:
                break
            sent_str[i] += i2w[str(word_id.item())] + " "

        sent_str[i] = sent_str[i].strip()


    return sent_str


def to_tensor(arr_like, cuda=True):
    tensor = torch.Tensor(arr_like)
    return tensor.cuda() if cuda and torch.cuda.is_available() else tensor


def ids2text(id_list, i2w, sep=''):
    return sep.join([i2w[f'{i}'] for i in id_list])


def ids2ptext(id_list, i2w, sep=''):
    text = ids2text(id_list, i2w, sep)
    return text.replace('<eos>', '').replace('<pad>', '')


def words2ids(words, w2i):
    assert type(words) == list
    return [w2i.get(word, UNK_INDEX) for word in words]


def words2input(words, w2i):
    id_list = [SOS_INDEX] + words2ids(words, w2i)
    sample_input = to_tensor(id_list).view(1,-1).to(dtype=torch.int64)
    sample_length = to_tensor([len(id_list)]).to(dtype=torch.int64)
    return {'input': sample_input, 'length': sample_length}


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T

def experiment_name(args, ts):

    exp_name = str()
    if args.experiment_name is not None:
        exp_name += f'{args.experiment_name}_'
        
    exp_name += "BS=%i_"%args.batch_size
    exp_name += "LR={}_".format(args.learning_rate)
    exp_name += "EB=%i_"%args.embedding_size
    exp_name += "%s_"%args.rnn_type.upper()
    exp_name += "HS=%i_"%args.hidden_size
    exp_name += "L=%i_"%args.num_layers
    exp_name += "BI=%i_"%args.bidirectional
    exp_name += "LS=%i_"%args.latent_size
    exp_name += "WD={}_".format(args.word_dropout)
    exp_name += "ANN=%s_"%args.anneal_function.upper()
    exp_name += "K={}_".format(args.k)
    exp_name += "X0=%i_"%args.x0
    exp_name += "TS=%s"%ts
    
    return exp_name