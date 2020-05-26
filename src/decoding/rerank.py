# coding=utf-8
from contextlib import ExitStack

import torch

from scripts.process_data import reverse_text
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.metric.bleu_scorer import SacreBLEUScorer
from src.models import build_model
from src.task.nmt import prepare_configs, load_model_parameters, prepare_data


class _Rerank(object):
    def __init__(self, config_path, model_path, model_type):
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
                            ),
            TextLineDataset(data_path=trg_filename,
                            vocabulary=self.vocab_tgt,
                            is_train_dataset=False
                            )
        )

        valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                      batch_size=20,
                                      use_bucket=True,
                                      buffer_size=100000,
                                      numbering=True,
                                      shuffle=False
                                      )

        valid_iter = valid_iterator.build_generator()
        score_result = dict()

        self.model.eval()
        with torch.no_grad():
            for batch in valid_iter:
                seq_numbers, seqs_x, seqs_y = batch
                x, y = prepare_data(seqs_x, seqs_y, cuda=True)
                y_inp = y[:, :-1].contiguous()
                y_label = y[:, 1:].contiguous()
                log_probs = self.model(x, y_inp)
                for i1, y_l in enumerate(y_label):
                    score = 0
                    for i2, y_index in enumerate(y_l):
                        if y_index.item() == 0:
                            break
                        score += log_probs[i1][i2][y_index.item()].item()
                    score_result.update({seq_numbers[i1]: score})

        return score_result

    def compute_scores(self, src_filename, trg_filename_list):
        score_list = []
        for trg_filename in trg_filename_list:
            score = self._compute_scores(src_filename, trg_filename)
            score_list.append(score)
        return score_list


class T2SRerank(_Rerank):

    def _compute_scores(self, src_filename, trg_filename):
        return super()._compute_scores(trg_filename, src_filename)


class R2LRerank(_Rerank):

    def _compute_scores(self, src_filename, trg_filename):
        trg_reverse = reverse_text(trg_filename)
        return super()._compute_scores(src_filename, trg_reverse)


MAPPING = {
    't2s': T2SRerank,
    'r2l': R2LRerank,
    'l2r': _Rerank
}


class Rerank(object):

    def __init__(self, rerank_configs):
        self.rerank_model = []
        self.weight_dict = dict()

        for rerank in rerank_configs:
            model_config = rerank['config']
            model_path = rerank['path']
            model_type = rerank['type']
            model = MAPPING.get(model_type, None)(model_config, model_path, model_type)
            self.rerank_model.append(model)
            self.weight_dict[model_type] = rerank['weight']

    def compute_scores(self, src_filename, trg_filename_list):
        result = dict()
        all_score_dict = dict()

        for model in self.rerank_model:
            # 统计每个beam的分数
            score_list = model.compute_scores(src_filename, trg_filename_list)
            all_score_dict.update({model.model_type: score_list})

            # 统计max_value
            result[model.model_type] = {i: 0 for i in range(len(trg_filename_list))}
            max_value = {}
            max_index = {}
            for indx, score_dict in enumerate(score_list):
                for key, value in score_dict.items():
                    max_value.setdefault(key, -10000000)
                    if max_value[key] < value:
                        max_value[key] = value
                        max_index[key] = indx
            for _, value in max_index.items():
                result[model.model_type][value] += 1

        print(result)

        self.ave(trg_filename_list, all_score_dict, self.weight_dict)

    def norm_data(self, all_score_dict, weight_dict):
        for model_type, score_list in all_score_dict.items():
            weight = weight_dict.get(model_type, 1)
            beam_size = len(score_list)
            sentences_number = len(score_list[0])
            for i in range(sentences_number):
                _sum = 0
                for j in range(beam_size):
                    _sum += score_list[j][i]
                for j in range(beam_size):
                    score_list[j][i] = (score_list[j][i] / _sum * weight)
        return all_score_dict

    def ave(self, trg_filename_list, all_score_dict, weight_dict):
        # all_score_dict = self.norm_data(all_score_dict, weight_dict)
        # 选最大的
        final_dict = {}
        for model_type, score_list in all_score_dict.items():
            for idx, score in enumerate(score_list):
                final_dict.setdefault(idx, dict())  # beam-level
                for number, score in score.items():
                    final_dict[idx].setdefault(number, 0)
                    final_dict[idx][number] += score

        beam_size = len(trg_filename_list)
        sentence_numbers = len(final_dict[0])
        final_index = dict()
        for i in range(sentence_numbers):
            max_value = -10000000
            for j in range(beam_size):
                value = final_dict[j][i]
                if value > max_value:
                    final_index[i] = j
                    max_value = value

        result = {i: 0 for i in range(len(trg_filename_list))}
        for _, value in final_index.items():
            result[value] += 1

        print(result)
        import os
        vote_filename = os.path.join(os.path.dirname(trg_filename_list[0]), "ave.final")
        with ExitStack() as stack:
            handlers = [stack.enter_context(open(filename)) for filename in trg_filename_list]
            contents = []
            for handle in handlers:
                contents.append(handle.readlines())
        best_contents = []
        for i in range(sentence_numbers):
            index = final_index[i]
            best_contents.append(contents[index][i])
        with open(vote_filename, 'w') as f:
            f.writelines(best_contents)
        with open(vote_filename, 'r') as f:
            compute_bleu(f)


def compute_bleu(f):
    ref_path = '/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/dev.en'
    num_ref = 1
    bleu_scorer = SacreBLEUScorer(reference_path=ref_path,
                                  num_refs=num_ref,
                                  lang_pair="zh-en",
                                  sacrebleu_args="--tokenize none -lc",
                                  postprocess=False
                                  )

    bleu_v = bleu_scorer.corpus_bleu(f)
    print(" bleu: " + str(bleu_v))


def rerank_max(beam_size, outputs):
    # 计算每一句话翻译得分，选取最高的计算\

    ref_path = '/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/dev.en'
    num_ref = 1
    bleu_scorer = SacreBLEUScorer(reference_path=ref_path,
                                  num_refs=num_ref,
                                  lang_pair="zh-en",
                                  sacrebleu_args="--tokenize none -lc",
                                  postprocess=False
                                  )

    max_index_dict = dict()
    with open(outputs + "final", 'w') as f:
        with ExitStack() as stack:
            handles = [stack.enter_context(open(outputs + str(index))) for index in range(beam_size)]
            for index, contents in enumerate(zip(*handles)):
                max_score, max_index = 0, -1
                for beam, content in zip(range(beam_size), contents):
                    score = bleu_scorer.corpus_single_bleu(content, index)
                    if score > max_score:
                        max_score = score
                        max_index = beam
                max_content = contents[max_index]
                max_index_dict.setdefault(max_index, 0)
                max_index_dict.update({max_index: max_index_dict[max_index] + 1})
                f.write(max_content)
    with open(outputs + "final", 'r') as f:
        score = bleu_scorer.corpus_bleu(f)

    print(score)
    print(max_index_dict)


if __name__ == '__main__':
    # rerank_max(beam_size=5, outputs="/home/user_data55/wangdq/code/njunmt/log/zh2en/outputmt03.a")
    rerank_config = [
        {"config": "/home/user_data55/wangdq/code/njunmt_dist/configs/en2zh/transformer_en2zh_bpe.yaml",
         "path": "/home/user_data55/wangdq/code/njunmt_dist/save/en2zh_baseline/transformer.best.final",
         "type": "t2s",
         "weight": 1.5},
        {"config": "/home/user_data55/wangdq/code/njunmt_dist/configs/zh2en/transformer_zh2en_r2l_bpe.yaml",
         "path": "/home/user_data55/wangdq/code/njunmt_dist/save/zh2en_r2l/transformer.best.final",
         "type": "r2l",
         "weight": 1.5},
        {"config": "/home/user_data55/wangdq/code/njunmt_dist/configs/zh2en/transformer_zh2en_bpe.yaml",
         "path": "/home/user_data55/wangdq/code/njunmt_dist/save/zh2en_baseline/transformer.best.final",
         "type": "l2r",
         "weight": 1},

    ]
    src_filename = "/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/dev.zh"
    trg_filename = "/home/user_data55/wangdq/code/njunmt/log/zh2en/outputmt03."
    r = Rerank(rerank_config)
    r.compute_scores(src_filename, [trg_filename + str(i) for i in range(5)])
