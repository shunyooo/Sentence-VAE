import numpy as np

import torch.nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import torch
import torch.nn as nn


def dynamic_rnn(cell, inputs, sequence_length, max_len=None, init_state=None):
    # 参考：https://github.com/shunyooo/kg-cvae-generator/blob/master/model/model_utils.py#L25-L71
    # ミニバッチでLSTMに学習させる場合, パディングして長さでソートする必要がある.
    # 出力はソート前に戻す

    sorted_lens, sorted_idx = sequence_length.sort(0, descending=True)

    sorted_inputs = inputs[sorted_idx].contiguous()
    if init_state is not None:
        sorted_init_state = init_state[:, sorted_idx].contiguous()

    packed_inputs = pack_padded_sequence(sorted_inputs, sorted_lens.data.tolist(), batch_first=True)


    if init_state is not None:
        _, state = cell(packed_inputs, sorted_init_state)
    else:
    	_, state = cell(packed_inputs)

    state = state.squeeze()
    _, reversed_idx = torch.sort(sorted_idx)
    return state[reversed_idx].contiguous()

    # Decoder用。既存の実装に任せる。
    # # Reshape *final* output to (batch_size, hidden_size)
    # outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=max_len)

    # # Add back the zero lengths
    # if zero_num > 0:
    #     outputs = torch.cat(
    #         [outputs, outputs.new_zeros(zero_num, outputs.size(1), outputs.size(2))], 0)
    #     if init_state is not None:
    #         state = torch.cat([state, sorted_init_state[:, valid_num:]], 1)
    #     else:
    #         state = torch.cat([state, state.new_zeros(state.size(0), zero_num, state.size(2))], 1)

    # # Reorder to the original order
    # new_outputs = outputs[inv_ix].contiguous()
    # new_state = state[:, inv_ix].contiguous()

    # # compensate the last last layer dropout, necessary????????? need to check!!!!!!!!
    # new_new_state = F.dropout(new_state, cell.dropout, cell.training)
    # new_new_outputs = F.dropout(new_outputs, cell.dropout, cell.training)

    # if output_fn is not None:
    #     new_new_outputs = output_fn(new_new_outputs)

    # return new_new_outputs, new_new_state
