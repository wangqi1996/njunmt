from contextlib import ExitStack

from src.decoding.compute_bleu import compute_bleu


def read_score(score_filename, beam_size, weight, is_use, contain_bleu):
    score_dict = {}  # {sequence_number: {beam_0: beam_0_score, beam_1: beam_1_score}}
    beam_index = 0
    sentence_number = 0
    score_dict[sentence_number] = {}
    with open(score_filename, 'r') as f:
        for line in f:
            if line == '\n':
                assert beam_index == beam_size
                beam_index = 0
                sentence_number += 1
                score_dict[sentence_number] = {}
                continue

            feature_list = line.split(' ')
            if contain_bleu:
                feature_list = feature_list[1:]
            assert len(feature_list) == len(weight)
            _score = 0
            for feature_index, feature in enumerate(feature_list):
                _score += float(feature) * int(is_use[feature_index]) * float(weight[feature_index])

            score_dict[sentence_number][beam_index] = _score
            beam_index += 1
    if len(score_dict.get(len(score_dict) - 1, {})) == 0:
        score_dict[len(score_dict) - 1] = {}
        score_dict.pop(len(score_dict) - 1)
    return score_dict


def select(trg_filename_list, score_filename, weights, is_use, save_filename, contain_bleu=False):
    # 选最大的
    beam_size = len(trg_filename_list)
    score_dict = read_score(score_filename, beam_size, weights, is_use, contain_bleu)
    sentence_numbers = len(score_dict)
    select_index = dict()
    for i in range(sentence_numbers):
        max_score = -10000000
        for j in range(beam_size):
            score = score_dict[i][j]
            if score > max_score:
                select_index[i] = j
                max_score = score

    # 统计最终结果均出自于哪一个beam_size
    index_count = {i: 0 for i in range(beam_size)}
    for _, value in select_index.items():
        index_count[value] += 1
    print(index_count)

    with ExitStack() as stack:
        handlers = [stack.enter_context(open(filename)) for filename in trg_filename_list]
        contents = []
        for handle in handlers:
            contents.append(handle.readlines())

    select_content = []
    for i in range(sentence_numbers):
        index = select_index[i]
        select_content.append(contents[index][i])

    with open(save_filename, 'w') as f:
        f.writelines(select_content)


if __name__ == '__main__':
    trg_filename_list = [
        "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/test.out." + str(
            i) for
        i in range(24)]

    src_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.uy.token"
    score_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/mira.input.dev"
    weight_list = [1]
    is_use = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    weight_index = 0
    weights = []
    for use in is_use:
        if use == '1' or use == 1:
            weights.append(weight_list[weight_index])
            weight_index += 1
        else:
            weights.append(0)
    assert len(weights) == len(is_use)

    save_filename = "/home/user_data55/wangdq/data/ccmt/uy-zh/output/dev_rerank/test.out.final"
    select(trg_filename_list, score_filename, weights, is_use, contain_bleu=True, save_filename=save_filename)

    ref = "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh"
    print('final: ', str(compute_bleu(open(save_filename), ref)))
