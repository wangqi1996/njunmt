# coding=utf-8

import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.models import build_model
from src.task.nmt import prepare_configs, load_model_parameters, prepare_data


def reverse_text(filename="/home/user_data55/wangdq/data/ccmt/zh-en/parallel/train.en"):
    lines = 0
    contents = []
    trg_reverse = filename + ".reverse"
    with open(filename, 'r') as f:
        with open(trg_reverse, 'w') as f2:
            for line in f:
                tokens = line.split()
                tokens.reverse()
                contents.append(' '.join(tokens) + '\n')
                lines += 1
                if lines == 500:
                    f2.writelines(contents)
                    contents = []
                    lines = 0
            if len(contents) > 0:
                f2.writelines(contents)
    return trg_reverse


class _Rerank(object):
    def __init__(self, config_path, model_path, model_type):
        print(config_path)
        print(model_path)
        print(model_type)
        self.model_type = model_type
        configs = prepare_configs(config_path)

        data_configs = configs['data_configs']
        model_configs = configs['model_configs']

        vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
        vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad,
                                **model_configs)

        params = load_model_parameters(model_path, map_location="cpu")
        nmt_model.load_state_dict(params)
        nmt_model.cuda()
        nmt_model.eval()

        self.model = nmt_model
        self.data_configs = data_configs
        self.model_configs = model_configs
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt

    def _compute_scores(self, src_filename, trg_filename):
        valid_bitext_dataset = ZipDataset(
            TextLineDataset(data_path=src_filename,
                            vocabulary=self.vocab_src,
                            is_train_dataset=False,
                            max_len=100
                            ),
            TextLineDataset(data_path=trg_filename,
                            vocabulary=self.vocab_tgt,
                            is_train_dataset=False,
                            max_len=100
                            )
        )

        valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                      batch_size=20,
                                      use_bucket=True,
                                      buffer_size=1000,
                                      numbering=True,
                                      shuffle=False
                                      )

        valid_iter = valid_iterator.build_generator()
        score_result = dict()

        self.model.eval()
        eidx = 0
        uidx = 0
        training_progress_bar = tqdm(desc=' - (Epc {}, Upd {}) '.format(eidx, uidx),
                                     total=len(valid_iterator),
                                     unit="sents"
                                     )

        with torch.no_grad():
            for batch in valid_iter:
                seq_numbers, seqs_x, seqs_y = batch
                x, y = prepare_data(seqs_x, seqs_y, cuda=True)
                y_inp = y[:, :-1].contiguous()
                y_label = y[:, 1:].contiguous()  # [batch_size, seq_len]
                log_probs = self.model(x, y_inp, log_probs=True)  # [batch_size, seq_len, vocab_size]

                batch_size, seq_len = y_label.shape
                log_probs = log_probs.view(-1, self.vocab_tgt.max_n_words)
                y_label = y_label.view(-1)

                loss = F.nll_loss(log_probs, y_label, reduce=False, ignore_index=self.vocab_tgt.pad)
                loss = loss.view(batch_size, seq_len)
                loss = loss.sum(-1)

                y_label = y_label.view(batch_size, seq_len)
                valid_token = (y_label != self.vocab_tgt.pad).sum(-1)
                loss = loss.double().div(valid_token.double())

                for seq_num, l in zip(seq_numbers, loss):
                    assert seq_num not in score_result
                    score_result.update({seq_num: l.item()})

                training_progress_bar.update(batch_size)
                training_progress_bar.set_description(' - (Epc {}, Upd {}) '.format(eidx, uidx))
                # for i1, y_l in enumerate(y_label):
                #     score = 0
                #     for i2, y_index in enumerate(y_l):
                #         if y_index.item() == 0:
                #             break
                #         score += log_probs[i1][i2][y_index.item()].item()
                #     valid_token = (y_label != self.vocab_tgt.pad).long().sum().item()
                #     score = -1 * score / valid_token
                #     score_result.update({seq_numbers[i1]: score})

        return score_result

    def compute_scores(self, src_filename, trg_filename):
        score = self._compute_scores(src_filename, trg_filename)
        score_str = [str(s) + '\n' for s in score]
        # 写入文件
        with open(trg_filename + ".score", "w") as f:
            f.writelines(score_str)


class T2SRerank(_Rerank):

    def _compute_scores(self, src_filename, trg_filename):
        return super()._compute_scores(trg_filename, src_filename)


class R2LRerank(_Rerank):

    def _compute_scores(self, src_filename, trg_filename):
        trg_reverse = reverse_text(trg_filename)
        return super()._compute_scores(src_filename, trg_reverse)


class L2RRerank(_Rerank):
    pass


MAPPING = {
    't2s': T2SRerank,
    'r2l': R2LRerank,
    'l2r': _Rerank
}

if __name__ == '__main__':
    # rerank_max(beam_size=5, outputs="/home/user_data55/wangdq/code/njunmt/log/zh2en/outputmt03.a")
    rerank_config = [
        {"config_path": "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/transformer_uy2zh_bpe_new4.yaml",
         "model_path": "/home/user_data55/wangdq/code/njunmt_dist/save/uy2zh/uy2zh_base_d4/transformer.best.final",
         "model_type": "l2r"},
        {"config_path": "/home/user_data55/wangdq/code/njunmt_dist/configs/zh2uy/transformer_zh2uy_bpe_new2.yaml",
         "model_path": "/home/user_data55/wangdq/code/njunmt_dist/save/zh2uy/zh2uy_base_a/transformer.best.final",
         "model_type": "t2s"},
    ]
    src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual/all/uy.token"
    trg_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual/all/zh.token"
    # src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/parallel/uy.token"
    # trg_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/parallel/zh.token"
    import sys

    if sys.argv[1] == "l2r":
        print("l2r!!!!")
        r = L2RRerank(**rerank_config[0])
        r.compute_scores(src_filename, trg_filename)
    elif sys.argv[1] == "t2s":
        print("t2s!!!!!!")
        r = T2SRerank(**rerank_config[1])
        r.compute_scores(src_filename, trg_filename)
