# coding=utf-8

# from scripts.process_data import reverse_text
# from scripts.process_data import reverse_text
from src.metric.bleu_scorer import SacreBLEUScorer


def compute_bleu(f, ref_path):
    num_ref = 1
    bleu_scorer = SacreBLEUScorer(reference_path=ref_path,
                                  num_refs=num_ref,
                                  lang_pair="uy-zh",
                                  sacrebleu_args="--tokenize zh",
                                  postprocess=False
                                  )
    bleu_v = bleu_scorer.corpus_bleu(f)
    return bleu_v
    # print(" bleu: " + str(bleu_v))


#
if __name__ == '__main__':
    outputs = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/test.out."
    ref = "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh"
    # for i in range(24):
    #     output = outputs + str(i)
    #     print(str(i), ' : ',str(compute_bleu(open(output), ref)))

    output = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/test.out.final.result"
    print('final: ', str(compute_bleu(open(output), ref)))
