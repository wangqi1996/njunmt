# coding=utf-8
from contextlib import ExitStack

# from scripts.process_data import reverse_text
# from scripts.process_data import reverse_text
from src.metric.bleu_scorer import SacreBLEUScorer


def compute_bleu_for_MIRA(outputs, beam_size, save_score_filename=None):
    ref_path = "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh"
    if save_score_filename is None:
        save_score_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/rerank/dev.score"
    num_ref = 1
    bleu_scorer = SacreBLEUScorer(reference_path=ref_path,
                                  num_refs=num_ref,
                                  lang_pair="uy-zh",
                                  sacrebleu_args="--tokenize zh",
                                  postprocess=False
                                  )

    result = []
    sum = 0
    with open(save_score_filename, 'w') as f:
        with ExitStack() as stack:
            if beam_size != 1:
                handles = [stack.enter_context(open(outputs + str(index))) for index in range(beam_size)]
            else:
                handles = [stack.enter_context(open(outputs))]
            for index, contents in enumerate(zip(*handles)):
                # print(index)
                candidate_score = []
                for beam, content in zip(range(beam_size), contents):
                    score = bleu_scorer.corpus_single_bleu(content, index)
                    if beam == 0:
                        sum += score
                    candidate_score.append(str(score))
                result.append(' '.join(candidate_score) + '\n')
            if len(result) > 0:
                f.writelines(result)

    print(sum / 1000)


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


def compute_bleu(f):
    ref_path = "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh"
    num_ref = 1
    bleu_scorer = SacreBLEUScorer(reference_path=ref_path,
                                  num_refs=num_ref,
                                  lang_pair="uy-zh",
                                  sacrebleu_args="--tokenize zh",
                                  postprocess=False
                                  )
    bleu_v = bleu_scorer.corpus_bleu(f)
    print(" bleu: " + str(bleu_v))


def rerank_max(beam_size, outputs, ref_path="/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh"):
    # 计算每一句话翻译得分，选取最高的计算\
    num_ref = 1
    bleu_scorer = SacreBLEUScorer(reference_path=ref_path,
                                  num_refs=num_ref,
                                  lang_pair="uy-zh",
                                  sacrebleu_args="--tokenize zh",
                                  postprocess=False
                                  )

    max_index_dict = dict()
    select_index = {}
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
                select_index[index] = max_index
                max_content = contents[max_index]
                max_index_dict.setdefault(max_index, 0)
                max_index_dict.update({max_index: max_index_dict[max_index] + 1})
                f.write(max_content)
    with open(outputs + "final", 'r') as f:
        score = bleu_scorer.corpus_bleu(f)

    print(score)
    print(max_index_dict)
    print(select_index)


def check(filename):

    beam = 24
    with open(filename, 'r') as f:
        for index, line in enumerate(f):
            if len(line.strip()) == 0:
                continue
            c = line.split(' ')
            if len(c) != 21:
                print(index, ' ', index / beam, ' ', index % beam)


#
if __name__ == '__main__':
    # pass
    # beam_size = 24
    # outputs = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/test.out."
    # rerank_max(beam_size, outputs)
    # program_name = "compute_bleu_for_mira"
    # if program_name == "compute_bleu_for_mira":
    #     compute_bleu_for_MIRA(
    #         outputs="/home/user_data55/wangdq/data/ccmt/uy-zh/output/rerank/ensemble_allsample_relative_avebest_rerank.out.",
    #         beam_size=24)
    # outputs = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/rerank/ensemble_allsample_relative_avebest_rerank.out.0"
    # compute_bleu(open(outputs))

    check( "/home/user_data55/wangdq/data/ccmt/uy-zh/output/test_rerank/mira.input.test")
#
