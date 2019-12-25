import os
import json
from utils import AttributeDict, words2input, ids2ptext, to_var
import torch
from model_kwbase import SentenceVAE
from constant import SOS_INDEX, UNK_INDEX, PAD_INDEX, EOS_INDEX

# ローカルのみ
from flask import Flask, jsonify
app = Flask(__name__)


model, src_vocab, tgt_vocab = None, None, None


model_filename = 'model.pytorch'
meta_filename = 'model_meta.json'
src_vocab_filename = 'src.vocab.json'
tgt_vocab_filename = 'tgt.vocab.json'

save_dir = './tmp'
model_path = os.path.join(save_dir, model_filename)
meta_path = os.path.join(save_dir, meta_filename)
src_vocab_path = os.path.join(save_dir, src_vocab_filename)
tgt_vocab_path = os.path.join(save_dir, tgt_vocab_filename)


# from google.cloud import storage
# from oauth2client.client import GoogleCredentials
# cred = GoogleCredentials.get_application_default()
# storage_client = storage.Client()
# bucket = storage_client.get_bucket(bucket_name)
# データのダウンロード
# bucket_name = 'kawamoto-ramiel'
# data_name = 'hr.mediatag'
# model_url = f'experiments_v1_kw2copy_20191220/models/{data_name}/E9.pytorch'
# meta_url = f'experiments_v1_kw2copy_20191220/models/{data_name}/model_meta.json'
# src_vocab_url = f'experiments_v1_kw2copy_20191220/data/{data_name}/src/ptb.vocab.json'
# tgt_vocab_url = f'experiments_v1_kw2copy_20191220/data/{data_name}/tgt/ptb.vocab.json'


def load_json(path):
    return json.load(open(path, 'r'))


def download_gs_data(url, save_path):
    blob = bucket.blob(url)
    blob.download_to_filename(save_path)


def load_model(path, meta_path):
        # モデルの読み込み
    with open(meta_path) as f:
        meta_dict = AttributeDict(json.load(f))

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    state_dict = torch.load(path, map_location=device)
    model_shapes = {k: v.shape for k, v in state_dict.items()}
    ext_kwargs = {}

    # BOW Loss
    bow_hidden_shape = model_shapes.get('latent2bow.0.weight')
    use_bow_loss = bow_hidden_shape is not None
    print(f'BOW Loss: {use_bow_loss}')
    if use_bow_loss:
        ext_kwargs['bow_hidden_size'] = bow_hidden_shape[0]
    else:
        ext_kwargs['use_bow_loss'] = False

    # Latent size
    latent_size = model_shapes.get('hidden2logv.bias')[0]
    print(f'Latent size: {latent_size}')

    # Gumbel
    gumbel_vocab_size, gumbel_embedding_size = model_shapes.get(
        'hidden2gumbel.weight', [None, None])
    is_gumbel = gumbel_vocab_size is not None
    print(f'Gumbel: {is_gumbel}')
    if is_gumbel:
        ext_kwargs['is_gumbel'] = is_gumbel
        ext_kwargs['gumbel_tau'] = tau_dict[name]
    print(ext_kwargs)

    vocab_size = model_shapes['embedding.weight'][0]

    # KW Base
    out_vocab_size, _ = model_shapes.get(
        'decoder_embedding.weight', [None, None])
    if out_vocab_size is not None:
        ext_kwargs['out_vocab_size'] = out_vocab_size

    model = SentenceVAE(
        vocab_size=vocab_size,
        sos_idx=SOS_INDEX,
        eos_idx=EOS_INDEX,
        pad_idx=PAD_INDEX,
        unk_idx=UNK_INDEX,
        max_sequence_length=meta_dict.max_sequence_length,
        embedding_size=meta_dict.embedding_size,
        rnn_type=meta_dict.rnn_type,
        hidden_size=meta_dict.hidden_size,
        word_dropout=meta_dict.word_dropout,
        embedding_dropout=meta_dict.embedding_dropout,
        latent_size=latent_size,
        num_layers=meta_dict.num_layers,
        bidirectional=meta_dict.bidirectional,
        **ext_kwargs,
    )
    print(model)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    model.load_state_dict(state_dict)
    print("Model loaded from %s" % (path))

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return model


def sample_z_from_gaussian(mean, logv):
    std = torch.exp(0.5 * logv)
    z = to_var(torch.randn([logv.shape[1]]))
    z = z * std + mean
    return z


def encode_z_list(model, sample_input, sample_length, n):
    z_list = []
    with torch.no_grad():
        hidden = model.encode(sample_input, sample_length)
        mean, logv, z = model.hidden2latent(hidden)
        z_list.append(z)
        for _ in range(n - 1):
            z = sample_z_from_gaussian(mean, logv)
            z_list.append(z)
    return torch.cat(z_list)


# 推論関数
def inference(word_list, model, src_vocab, tgt_vocab, n=10):
    # Inputへ整形
    sample = words2input(word_list, src_vocab['w2i'])
    sample_input, sample_length = sample['input'], sample['length']
    # Encode
    z_list = encode_z_list(model, sample_input, sample_length, n)
    # Decode
    generations, _ = model.inference(z=z_list)
    # Outputへ整形
    decode_texts = [ids2ptext(g, tgt_vocab['i2w']) for g in generations]
    return decode_texts


def main(request):
    # データのダウンロード
    # if not os.path.exists(model_path):
    #     download_gs_data(model_url, model_path)
    #     download_gs_data(src_vocab_url, src_vocab_path)
    #     download_gs_data(tgt_vocab_url, tgt_vocab_path)
    #     download_gs_data(meta_url, meta_path)

    # インスタンス化
    global model, src_vocab, tgt_vocab
    if model is None or src_vocab is None or tgt_vocab is None:
        model = load_model(model_path, meta_path)
        src_vocab, tgt_vocab = load_json(
            src_vocab_path), load_json(tgt_vocab_path)

    if not (request.args and 'keyword' in request.args):
        return jsonify({'message': 'no keyword'}), 500

    keyword_list = request.args.getlist('keyword')
    n = request.args.get('n', 100)
    decoded_texts = inference(keyword_list, model, src_vocab, tgt_vocab, n=n)
    return jsonify({'message': 'success', 'decoded': decoded_texts}), 200


if __name__ == "__main__":
    from flask import Flask, request
    app = Flask(__name__)

    @app.route('/')
    def index():
        return main(request)

    app.run('0.0.0.0', 8000, debug=True)
