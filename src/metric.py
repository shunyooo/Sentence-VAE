from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk
import torch
import numpy as np
from constant import PAD_INDEX, SOS_INDEX
from utils import ids2ptext


def write_tensorboard_valid_metric(writer, valid_tgt_id_list, decoded_id_list, train_tgt_id_list, train_tgt_ptb, split, epoch):
    # 各種指標の計測（BLEU, Distinct-1, 2, full, Train_contains_decode）
    # BLEU
    bleu = calculate_bleu(valid_tgt_id_list, decoded_id_list)
    writer.add_scalar(f"{split.upper()}/BLEU", bleu, epoch)

    # Distinct-1, 2, full
    distinct_dict, _ = flat_and_cal_diverse(decoded_id_list, [1, 2, 3, 'full'])
    for k, v in distinct_dict.items():
        if 'diversity' in k:
            writer.add_scalar(f"{split.upper()}/{k}", v, epoch)

    # コピー率
    train_tgt_id_list = remove_pad_index(train_tgt_id_list)
    contain_dict = summarize_full_contains(train_tgt_id_list, decoded_id_list)
    writer.add_scalar(f"{split.upper()}/novelty", contain_dict['decode_novelty'], epoch)

    # 3. デコードテキストのサンプル保存
    dlen, n_sample = len(decoded_id_list), 5
    save_decode_index_list = list(range(0, dlen, int(dlen / n_sample)))
    for i in save_decode_index_list:
        save_ids = valid_tgt_id_list[i] if epoch == 0 else decoded_id_list[i]
        writer.add_text(f'test-decode-{i}', f'```{ids2ptext(save_ids, train_tgt_ptb.i2w)}```', epoch)


# utils
def to_list(obj):
    if type(obj) == torch.Tensor:
        obj = obj.detach().tolist()
    if type(obj) == np.ndarray:
        obj = obj.tolist()

    if type(obj) == int:
        return obj
    if type(obj) == str:
        return obj
    if type(obj) == float:
        return obj

    if len(obj) > 0:
        return [to_list(item) for item in obj]


def get_item_dtype(obj):
    if type(obj) == torch.Tensor:
        obj = obj.detach().tolist()

    if type(obj) == int:
        return int
    if type(obj) == str:
        return str
    if type(obj) == float:
        return str

    if len(obj) > 0:
        return get_item_dtype(obj[0])


def remove_pad_index(obj):
    obj = to_list(obj)

    if type(obj) == list and type(obj[0]) == list:
        return [remove_pad_index(item) for item in obj]
    else:
        return [index for index in obj if index != PAD_INDEX and index != SOS_INDEX]


# ------------------- 新規性（Novelty）-------------------
def count_contains(inner_list, outer_list):
    return sum([inner in outer_list for inner in inner_list])


def summarize_full_contains(train_lines, decode_lines):
    if get_item_dtype(train_lines) == int:
        train_lines = remove_pad_index(train_lines)
        train_lines = [tuple(ids) for ids in train_lines]
    if get_item_dtype(decode_lines) == int:
        decode_lines = remove_pad_index(decode_lines)
        decode_lines = [tuple(ids) for ids in decode_lines]

    decode_line_set = set(decode_lines)
    train_line_set = set(train_lines)
    # 生成結果中 コピっていないカウント
    decode_in_train_count = count_contains(decode_lines, train_lines)
    return {
        'decode_line_count': len(decode_lines),
        'decode_set_count': len(decode_line_set),
        'train_line_count': len(train_lines),
        'train_set_count': len(train_line_set),
        'train_set&decode_set': len(decode_line_set & train_line_set),
        'train_set&decode_set/decode_set': len(decode_line_set & train_line_set) / len(decode_line_set),
        # 新規性（Novelty）
        'decode_novel_count': len(decode_lines) - decode_in_train_count,
        'decode_novelty': (len(decode_lines) - decode_in_train_count) / len(decode_lines)
    }


# BLEU
def calculate_bleu(references, hypotheses, sm_func=None):
    if get_item_dtype(references) == int:
        references = remove_pad_index(references)
    if get_item_dtype(hypotheses) == int:
        hypotheses = remove_pad_index(hypotheses)
    if sm_func is None:
        sm_func = SmoothingFunction().method7
    list_of_references = [[reference] for reference in references]
    return corpus_bleu(list_of_references, hypotheses, smoothing_function=sm_func)


# ------------------- DIVERSITY/DISTINCT -------------------
def ngram(words, n):
    # ngramに分割
    if n == 'full':
        return [tuple(words)]
    return list(zip(*(words[i:] for i in range(n))))


def flat_list(_llist, n=1):
    # _llist: word_list の list
    # 単語の愚直なリストに。ngram で flat にする
    all_list = []
    for _list in _llist:
        all_list += ngram(_list, n)
    return all_list


def cal_mean(_list):
    # 平均算出
    return sum(_list) / len(_list)


def cal_diverse(token_list):
    # 重複なし生成単語リスト
    unique_count = len(set(token_list))
    # 多様性
    diversity = unique_count / len(token_list)
    return {
        'unique_count': unique_count,
        'diversity': diversity,
    }


def flat_and_cal_diverse(text_list, ngrams):
    # text_listから多様性を算出
    # 重複あり生成単語リスト
    res_dict = {}
    data_dict = {}
    for ngram in ngrams:
        token_list = flat_list(text_list, ngram)
        suffix = f'{ngram}gram' if ngram != 'full' else 'full'
        d = cal_diverse(token_list)
        d[f'unique_count_{suffix}'] = d.pop('unique_count')
        d[f'diversity_{suffix}'] = d.pop('diversity')
        d[f'duplicate_count_{suffix}'] = len(token_list)
        res_dict.update(d)
        data_dict.update({f'token_list_{suffix}': token_list})
        data_dict.update({f'token_set_{suffix}': set(token_list)})
    return res_dict, data_dict