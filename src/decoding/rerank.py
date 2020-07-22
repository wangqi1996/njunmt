# coding=utf-8

#  使用fast_align 计算前向和后向概率
import torch
import torch.nn.functional as F

# from scripts.process_data import reverse_text
from scripts.process.merge import merge
from src.data.data_iterator import DataIterator
from src.data.dataset import TextLineDataset, ZipDataset
from src.data.vocabulary import Vocabulary
from src.decoding.bleu_helper import reverse_text
from src.decoding.force_align import force_align_main
from src.models import build_model
from src.task.nmt import prepare_configs, load_model_parameters, prepare_data

"""
语言模型：
/home/user_data55/put/kenlm/build/bin/lmplz -o 5 --verbose_header --text /home/user_data55/wangdq/data/ccmt/uy-zh/sample/zh.token --arpa /home/wangdq/log.arpa --vocab_file /home/wangdq/log.vocab
"""


class _Rerank(object):
    def __init__(self, config_path, model_path, model_type, model_id):
        self.model_type = model_type
        self.model_id = model_id
        configs = prepare_configs(config_path)

        data_configs = configs['data_configs']
        model_configs = configs['model_configs']

        vocab_src = Vocabulary.build_from_file(**data_configs['vocabularies'][0])
        vocab_tgt = Vocabulary.build_from_file(**data_configs['vocabularies'][1])

        nmt_model = build_model(n_src_vocab=vocab_src.max_n_words,
                                n_tgt_vocab=vocab_tgt.max_n_words, padding_idx=vocab_src.pad,
                                vocab_src=vocab_src, vocab_tgt=vocab_tgt,
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
        # 其实好像还是会打乱
        valid_iterator = DataIterator(dataset=valid_bitext_dataset,
                                      batch_size=40,
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
                log_probs = self.model(x, y_inp)  # [batch_size, seq_len, vocab_size]

                batch_size, seq_len = y_label.shape
                log_probs = log_probs.view(-1, self.vocab_tgt.max_n_words)
                y_label = y_label.view(-1)

                loss = F.nll_loss(log_probs, y_label, reduce=False, ignore_index=self.vocab_tgt.pad)  # 越小越好
                loss = loss.view(batch_size, seq_len)
                loss = loss.sum(-1)

                y_label = y_label.view(batch_size, seq_len)
                valid_token = (y_label != self.vocab_tgt.pad).sum(-1)
                norm_loss = loss.double().div(valid_token.double())

                for seq_num, l, nl in zip(seq_numbers, loss, norm_loss):
                    score_result.update({seq_num: (l.item(), nl.item())})
                # for i1, y_l in enumerate(y_label):
                #     score = 0
                #     for i2, y_index in enumerate(y_l):
                #         if y_index.item() == 0:
                #             break
                #         score += log_probs[i1][i2][y_index.item()].item()
                #     score_result.update({seq_numbers[i1]: score})

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


class WP(_Rerank):
    def __init__(self, model_config, model_path, model_type, model_id):
        self.model_id = model_id
        self.model_type = model_type

    # 计算候选项的长度
    def _compute_scores(self, src_filename, trg_filename):
        score_result = {}
        with open(trg_filename, 'r') as f:
            for index, line in enumerate(f):
                length = len(line.split(' '))
                score_result.update({
                    index: (length,)
                })
        return score_result


class LengthRatio(_Rerank):
    def __init__(self, model_config, model_path, model_type, model_id):
        self.model_id = model_id
        self.model_type = model_type

    # 计算候选项的长度
    def _compute_scores(self, src_filename, trg_filename):
        score_result = {}
        with open(trg_filename, 'r') as f_trg, open(src_filename, 'r') as f_src:
            for index, (line, src) in enumerate(zip(f_trg, f_src)):
                length = len(line.split(' '))
                length2 = len(src.split(' '))
                score_result.update({
                    index: (abs(length - length2), length / length2)
                })
        return score_result


class Alignment(_Rerank):
    """https://github.com/clab/fast_align/issues/33"""

    def __init__(self, model_config, model_path, model_type, model_id):
        self.model_id = model_id
        self.model_type = model_type

    # 计算候选项的长度
    def _compute_scores(self, src_filename, trg_filename):
        token = "/home/user_data55/wangdq/data/ccmt/uy-zh/sample/rerank/align.token"
        merge(src_filename, trg_filename, token)
        base = "/home/user_data55/wangdq/data/ccmt/uy-zh/sample/rerank/"
        fwd_params, fwd_err, rev_params, rev_err = base + "uy2zh_params", base + "uy2zh_error", \
                                                   base + "zh2uy_params", base + "zh2uy_error"

        index = trg_filename[-2:]
        score_result = force_align_main(fwd_params, fwd_err, rev_params, rev_err, token, index)
        return score_result


class LM(_Rerank):
    # https://www.cnblogs.com/zidiancao/p/6067147.html
    def __init__(self, model_config, model_path, model_type, model_id):
        self.model_id = model_id
        self.model_type = model_type
        import kenlm
        print("loading LM")
        self.model = kenlm.Model('/home/user_data55/wangdq/model/LM')
        print("loading end!")

    def _compute_scores(self, src_filename, trg_filename):
        score_dict = {}
        with open(trg_filename, 'r') as f:
            for seq_num, line in enumerate(f):
                score = self.model.score(line, bos=True, eos=True)
                score_dict.update({
                    seq_num: (score,)
                })
        return score_dict


MAPPING = {
    't2s': T2SRerank,
    'r2l': R2LRerank,
    'l2r': _Rerank,
    'WP': WP,
    'LengthRatio': LengthRatio,
    'Alignment': Alignment,
    'LM': LM
}


class Rerank(object):

    def __init__(self, rerank_configs):
        self.rerank_model = []
        self.weight_dict = dict()

        for rerank in rerank_configs:
            model_config = rerank['config']
            model_path = rerank['path']
            model_type = rerank['type']
            model_id = rerank['id']
            model = MAPPING.get(model_type, None)(model_config, model_path, model_type, model_id)
            self.rerank_model.append(model)
            # self.weight_dict[model_type] = rerank['weight']

    def compute_scores_for_mira(self, src_filename, trg_filename_list, save_filename, bleu_filename, model_ids):
        all_score_dict = self.compute_scores(src_filename, trg_filename_list)
        final_dict = {}
        for model_id in model_ids:  # 不同的模型
            score_list = all_score_dict[model_id]
            for beam_id, scores in enumerate(score_list):  # 不同的candidate
                final_dict.setdefault(beam_id, dict())  # beam-level
                for number, score in scores.items():
                    final_dict[beam_id].setdefault(number, [])
                    final_dict[beam_id][number].extend([str(s) for s in score])  # 所有rerank模型的打分拼接起来

        contents = []
        beam_size = len(trg_filename_list)

        with open(save_filename, 'w') as f:
            if bleu_filename is not None:
                with open(bleu_filename, 'r') as bleu_list:
                    for sentence_index, bleu in enumerate(bleu_list):
                        candi_bleu_list = bleu.split(' ')
                        for beam_index, candi_bleu in enumerate(candi_bleu_list):
                            c = str(candi_bleu.strip()) + ' ' + ' '.join(final_dict[beam_index][sentence_index]) + '\n'
                            contents.append(c)
                        contents.append('\n')
            else:
                sentences_num = len(final_dict[0])
                for sentence_index in range(sentences_num):
                    for beam_index in range(beam_size):
                        c = ' '.join(final_dict[beam_index][sentence_index]) + '\n'
                        contents.append(c)
                    contents.append('\n')

            if len(contents) > 0:
                f.writelines(contents)

    def compute_scores(self, src_filename, trg_filename_list):
        all_score_dict = dict()

        for model in self.rerank_model:
            # 统计每个beam的分数
            score_list = model.compute_scores(src_filename, trg_filename_list)
            all_score_dict.update({model.model_id: score_list})

        # self.ave(trg_filename_list, all_score_dict, self.weight_dict)
        return all_score_dict


if __name__ == '__main__':
    # 暂时这样写吧，反正不常用
    # program_name = "compute_bleu_for_mira"
    program_name = 'compute_rerank_score'
    if program_name == "compute_rerank_score":
        # 对某个模型计算rerank
        csample1 = "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/transformer_uy2zh_sample_big.yaml"
        csample2 = "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/transformer_uy2zh_sample2_big.yaml"
        crelative1 = "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/sample_big_relative.yaml"
        crelative2 = "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/sample2_big_relative.yaml"
        cave = "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/sample_big_ave.yaml"
        cdynamic_cnn = "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/sample1_dynamic_cnn.yaml"

        sample1 = "/home/user_data55/wangdq/save/uy2zh/final/tune/sample1_tune_train/transformer.best.final"
        sample2 = "/home/user_data55/wangdq/save/uy2zh/final/tune/sample2_tune_train/transformer.best.final"
        relative1 = "/home/user_data55/wangdq/save/uy2zh/final/tune/relative1_tune_train/transformer_relative.best.final"
        relative2 = "/home/user_data55/wangdq/save/uy2zh/final/tune/relative2_tune_train/transformer.best.final"
        ave = "/home/user_data55/wangdq/save/uy2zh/final/tune/ave_tune_train/transformer.best.final"
        dynamic_cnn = "/home/user_data55/wangdq/save/uy2zh/final/tune/dynamic_cnn_tune_train/dynamicCnn.best.final"

        rerank_config = [
            {
                "id": 1,  # sample1
                "config": csample1,
                "path": sample1,
                "type": "l2r"},
            {
                "id": 2,  # sample2
                "config": csample2,
                "path": sample2,
                "type": "l2r"},
            {
                "id": 3,  # relative1
                "config": crelative1,
                "path": relative1,
                "type": "l2r"},
            {
                "id": 4,  # relative2
                "config": crelative2,
                "path": relative2,
                "type": "l2r"},
            {
                "id": 5,
                "config": "/home/user_data55/wangdq/code/njunmt_dist/configs/zh2uy/transformer_zh2uy_bpe_new2.yaml",
                "path": "/home/user_data55/wangdq/model/zh2uy/transformer.best.final",
                "type": "t2s"},
            {
                "id": 6,
                "type": "WP",
                "config": "",
                "path": "",
            },
            {
                "id": 7,
                "type": "LengthRatio",
                "config": "",
                "path": "",
            },
            {
                "id": 8,
                "type": "Alignment",
                "config": "",
                "path": "",
            },
            {
                "id": 9,
                "type": "LM",
                "config": "",
                "path": "",
            },
            {
                "id": 10,
                "config": "/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/rerank/sample_r2l.yaml",
                "path": "/home/user_data55/liuzh/ccmt_model/uy2zh/rerank/uy2zh_sample_r2l/transformer.best.final",
                "type": "r2l"},
            {
                "id": 11,  # dynamicCNN
                "config": cdynamic_cnn,
                "path": dynamic_cnn,
                "type": "l2r"},
            {
                "id": 12,
                "config": cave,
                "path": ave,
                "type": "l2r",
            }

        ]

        rerank_model = Rerank(rerank_config)
        task = "test"
        if task == "test":
            # src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/test/test.txt"
            # save_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/rerank/mira.input.testset"
            trg_filename_list = [
                "/home/user_data55/wangdq/data/ccmt/uy-zh/output/test_rerank/dev24.out." + str(
                    i) for i in range(24)]
            model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/test/test_2020.txt"
            save_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/test_rerank/mira.input.test"
            rerank_model.compute_scores_for_mira(src_filename, trg_filename_list, save_filename, None, model_ids)
        else:
            beam_size = 24
            filename_prefix = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/test.out."
            trg_filename_list = [filename_prefix + str(i) for i in range(beam_size)]
            model_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.uy.token"
            save_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/mira.input.test"
            bleu_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/dev.score"

            # compute_bleu_for_MIRA(filename_prefix, beam_size, bleu_filename)

            rerank_model.compute_scores_for_mira(src_filename, trg_filename_list, save_filename, bleu_filename,
                                                 model_ids)
