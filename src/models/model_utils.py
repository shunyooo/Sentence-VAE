import torch
from constant import SOS_INDEX
from torch.autograd import Variable

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def to_tensor(arr_like, cuda=True):
    tensor = torch.Tensor(arr_like)
    return tensor.cuda() if cuda and torch.cuda.is_available() else tensor


def words2input(words, w2i):
    id_list = [SOS_INDEX] + words2ids(words, w2i)
    sample_input = to_tensor(id_list).view(1,-1).to(dtype=torch.int64)
    sample_length = to_tensor([len(id_list)]).to(dtype=torch.int64)
    return {'input': sample_input, 'length': sample_length}