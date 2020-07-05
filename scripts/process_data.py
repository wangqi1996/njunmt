# coding=utf-8
import copy
import os
import random
from contextlib import ExitStack


def split_en():
    # tokenize english text
    import sacremoses
    tokenizer = sacremoses.MosesTokenizer(lang="en")
    lines = 0
    contents = []
    filename = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/train.en"
    with open(filename, 'r') as f:
        with open(filename + '2', 'w') as f2:
            for line in f:
                tokens = tokenizer.tokenize(line, aggressive_dash_splits=True, return_str=True, escape=False)
                contents.append(tokens + '\n')
                lines += 1
                if lines == 500:
                    f2.writelines(contents)
                    contents = []
                    lines = 0
            if len(contents) > 0:
                f2.writelines(contents)


def split_test(src_filename, trg_filename, size=1000, times=3):
    src_contents = [[] for _ in range(times)]
    trg_contents = [[] for _ in range(times)]
    # count the size of file
    from src.data.dataset import get_num_of_lines
    lines = get_num_of_lines(src_filename)  # line = 656759
    import random
    numbers = random.sample(range(lines), times * size)
    file_index = 0
    with open(src_filename, 'r') as f_src, open(trg_filename, 'r') as f_trg:
        for index, (src, trg) in enumerate(zip(f_src, f_trg)):
            if index in numbers:
                src_contents[file_index % 3].append(src)
                trg_contents[file_index % 3].append(trg)
                file_index += 1
    with ExitStack() as stack:
        src_names = [os.path.join(os.path.dirname(src_filename), 'test.src.' + str(i)) for i in range(times)]
        handlers = [stack.enter_context(open(name, 'w')) for name in src_names]
        for src, handler in zip(src_contents, handlers):
            handler.writelines(src)

        trg_names = [os.path.join(os.path.dirname(src_filename), 'test.trg.' + str(i)) for i in range(times)]
        handlers = [stack.enter_context(open(name, 'w')) for name in trg_names]
        for trg, handler in zip(trg_contents, handlers):
            handler.writelines(trg)


def mlm(filename, max_length=150):
    """ filename: monolingual file
    for example:
    src: 团购 [MASK] 一旦 出现 质量 问题
    trg:  0    商品   0    0   0    0
    只能处理经过bpe处理的文件。
    """
    src_filename = os.path.join(os.path.dirname(filename), "mlm.src")
    trg_filename = os.path.join(os.path.dirname(filename), "mlm.trg")
    src_contents = []
    trg_contents = []
    with open(filename, 'r') as f, open(src_filename, 'w') as f_src, open(trg_filename, 'w') as f_trg:
        for line in f:
            trg_tokens = line.strip().split()[:max_length]
            tokens = copy.deepcopy(trg_tokens)
            masked_tokens = []
            flag = False
            for i in range(0, len(tokens)):
                if i >= max_length:
                    break
                token = tokens[i]
                rd = random.uniform(0, 1)
                if rd < 0.15:
                    flag = True
                    rd2 = random.uniform(0, 1)
                    if rd2 < 0.8:
                        masked_tokens.append('[MASK]')
                    elif 0.8 < rd2 < 0.9:
                        j = random.randint(0, len(tokens) - 1)
                        if j == i:
                            i = i - 1 if i != 0 else i + 1
                        masked_tokens.append(tokens[j])
                    else:
                        masked_tokens.append(token)
                else:
                    masked_tokens.append(token)
                    trg_tokens[i] = "<PAD>"
            if flag is False:
                continue
            src_contents.append(' '.join(masked_tokens) + '\n')
            trg_contents.append(' '.join(trg_tokens) + '\n')

            if len(src_contents) > 300:
                f_src.writelines(src_contents)
                f_trg.writelines(trg_contents)
                src_contents = []
                trg_contents = []
        if len(src_contents) > 0:
            f_src.writelines(src_contents)
            f_trg.writelines(trg_contents)
        print("end!!")


def DAE(filename, max_length=150):
    src_filename = os.path.join(os.path.dirname(filename), "DAE/dae.src")
    trg_filename = os.path.join(os.path.dirname(filename), "DAE/dae.trg")
    src_contents = []
    trg_contents = []
    import numpy as np
    with open(filename, 'r') as f, open(src_filename, 'w') as f_src, open(trg_filename, 'w') as f_trg:
        for line in f:
            trg_tokens = line.strip().split()[:max_length]
            masked_tokens = []

            j = 0
            for i in range(0, len(trg_tokens)):
                assert j >= i, u" j = %s, i = %s" % (j, i)
                if i >= max_length:
                    break
                if i < j:
                    continue
                token = trg_tokens[i]
                rd = random.uniform(0, 1)
                if rd <= 0.1:
                    span_len = np.random.poisson(lam=3)  # BART
                    j = i + span_len
                    masked_tokens.append('[MASK]')
                    if span_len == 0:
                        j += 1
                        masked_tokens.append(token)
                else:
                    masked_tokens.append(token)
                    j += 1

            src_contents.append(' '.join(masked_tokens) + '\n')
            trg_contents.append(' '.join(trg_tokens) + '\n')

            if len(src_contents) > 300:
                f_src.writelines(src_contents)
                f_trg.writelines(trg_contents)
                src_contents = []
                trg_contents = []
        if len(src_contents) > 0:
            f_src.writelines(src_contents)
            f_trg.writelines(trg_contents)
        print("end!!")


def strQ2B(ustring):
    ss = []
    for s in ustring:
        rstring = ""
        for uchar in s:
            inside_code = ord(uchar)
            if inside_code == 12288:  # 全角空格直接转换
                inside_code = 32
            elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
                inside_code -= 65248
            rstring += chr(inside_code)
        ss.append(rstring)
    return ''.join(ss)


if __name__ == '__main__':
    pass

# tttt()
# src = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/zh.token"
# trg = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/en.token"
# out = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/token"
# merge(src, trg, out)
# f1 = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/zh.token"
# f2 = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/en.token"
#
# cut_by_length(f1, f2)
# process_dev_as_text("/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/ref")
# process("/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh","/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.uy", )
# count_length(f1, f2)
# cut_by_length(f1, f2, 110, 90)
# count()

# pretrain
# mlm_filename = "/home/user_data55/wangdq/data/ccmt/processed-uy-zh/monolingual/monolingual.token.bpe"
#     mlm(mlm_filename)
#     src = "/home/user_data55/wangdq/data/ccmt/processed-uy-zh/monolingual/mlm.src"
#     trg = "/home/user_data55/wangdq/data/ccmt/processed-uy-zh/monolingual/mlm.trg"
# src = "/home/user_data55/wangdq/data/ccmt/processed-uy-zh/monolingual/DAE/dae.src"
# trg = "/home/user_data55/wangdq/data/ccmt/processed-uy-zh/monolingual/DAE/dae.trg"
# split_test(src, trg, 1000, 3)
# DAE(mlm_filename, 150)
# process_dev_as_text("/home/user_data55/wangdq/data/ccmt/uy-zh/CCMT2019_UC_test_src.xml")
