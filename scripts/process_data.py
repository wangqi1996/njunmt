# coding=utf-8
import copy
import os
import random
import re
from contextlib import ExitStack

from xml.dom.minidom import parse

from sacremoses import MosesDetokenizer, MosesDetruecaser


def get_sentence(file):
    # 读取文件
    rst = {}
    domTree = parse(file)
    docs = domTree.getElementsByTagName("doc")

    for doc in docs:
        # print("docid",doc.getAttribute('docid'))
        docid = doc.getAttribute('docid')
        rst.setdefault(doc.getAttribute('docid'), {})
        segs = doc.getElementsByTagName('seg')
        for seg in segs:
            # print("id:%d,sentence:%s"%(int(seg.getAttribute('id')),seg.childNodes[0].data))
            rst[docid].setdefault(seg.getAttribute('id'), seg.childNodes[0].data)

    return rst


def process_dev(zh_file_name, en_file_name):
    en_src = get_sentence(en_file_name)
    cn_ref = get_sentence(zh_file_name)

    en_fn = os.path.join(os.path.dirname(en_file_name), 'dev.en')
    zh_fn = os.path.join(os.path.dirname(zh_file_name), 'dev.zh')
    with open(en_fn, "w") as en_fp:
        with open(zh_fn, "w") as cn_fp:
            for docid in en_src:
                for id in en_src[docid]:
                    if (docid in cn_ref) and (id in cn_ref[docid]):
                        en_fp.writelines(en_src[docid][id] + "\n")
                        cn_fp.writelines(cn_ref[docid][id] + "\n")


def process_dev_as_text(zh_file_name):
    # precess file as text file line by line
    import re
    index = 0
    last_index = 0
    index_pattern = re.compile(r'id=\"(\d+)\"')
    seg_pattern = re.compile(r'<seg .*>(.*?)</seg>')
    contents = []
    with open(zh_file_name, 'r') as f:
        for line in f:
            if "<seg" in line:
                index += 1
                r = re.findall(index_pattern, line)
                if int(r[0]) == 1:
                    last_index = index - 1
                text_index = int(r[0]) + last_index
                assert text_index == index, u"line = %s, index = %s, text_index=%s" % (
                    line, str(index), str(text_index))
                new_line = re.findall(seg_pattern, line)
                contents.append(new_line[0] + '\n')
    zh_fn = os.path.join(os.path.dirname(zh_file_name), 'dev.uy.raw')
    with open(zh_fn, 'w') as f:
        f.writelines(contents)


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


def process_mono():
    filename = "/home/user_data55/wangdq/data/ccmt/zh-en/monolingual/xmu.txt"
    processed_name = os.path.join(os.path.dirname(filename), 'xmu.txt2')
    import re
    start_pattern = re.compile(r'<text')
    end_pattern = re.compile(r'</text>')
    content_pattern = re.compile(r'<text .*>(.*?)</text>')
    contents = []
    raw_text = ''
    start = False
    with open(filename, 'r') as f:
        with open(processed_name, 'w') as fw:
            for line in f:
                end_search = re.search(end_pattern, line)
                start_search = re.search(start_pattern, line)
                if start is True:
                    assert start_search is None, ""
                    raw_text += (line.strip('\n').strip())
                    if end_search is not None:
                        text = re.findall(content_pattern, raw_text)
                        contents.append(text[0] + '\n')
                        start = False
                        if len(contents) > 300:
                            fw.writelines(contents)
                            contents = []
                else:
                    assert end_search is None, u''
                    if start_search is not None:
                        raw_text = line.strip('\n').strip(' ')
                        start = True

            if len(contents) >= 0:
                fw.writelines(contents)


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


def count():
    src_filename = "/home/user_data55/wangdq/data/ccmt/zh-en/monolingual/test.src.0"
    trg_filename = "/home/user_data55/wangdq/data/ccmt/zh-en/monolingual/test.trg.0"
    with open(src_filename) as f_src, open(trg_filename) as f_trg:
        for l1, l2 in zip(f_src, f_trg):
            len1 = len(l1.split())
            len2 = len(l2.split())
            if len1 != len2:
                print(l1, l2)
            if len1 != 150:
                print(len1, len2)


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


def process(filename1, filename2):
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

    l1_list = []
    l2_list = []
    raw1 = {}
    raw2 = {}
    amp = r"&amp;"
    lt = r"&lt;"
    gt = r"&gt;"
    quot = r"&quot;"
    apos = r"&apos;"
    delete = 0
    re_space = re.compile(r"(?<![a-zA-Z])\s(?![a-zA-Z])", flags=re.UNICODE)
    re_final_comma = re.compile("\.$")
    pattern_list = {amp: "&", lt: ">", gt: "<", quot: "\"", apos: "\'"}
    with open(filename1 + ".new", 'w') as f1_w, open(filename2 + ".new", 'w') as f2_w:
        with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
            for index, (l1, l2) in enumerate(zip(f1, f2)):
                # 重复句的删除
                h1 = hash(l1)
                h2 = hash(l2)
                if h1 in raw1 and h2 in raw2:
                    if raw1[h1] == raw2[h2]:
                        # print(raw1[h1])
                        print(raw2[h2])
                        print(index)
                        # print(l1)
                        # print(l2)
                        # print(h1)
                        # print(h2)
                        delete += 1
                        continue
                raw1[h1] = index
                raw2[h2] = index
                # 语种检测
                # from langdetect import detect
                # if detect(l1) != 'zh-cn' and detect(l2) != 'en':
                #     print(l1)
                #     print(l2)
                #     continue
                l1 = l1.strip().strip('\n').strip()
                l2 = l2.strip().strip('\n').strip()
                # 全角转半角
                l1 = strQ2B(l1)
                l2 = strQ2B(l2)
                # 转义字符处理
                for pattern, replace in pattern_list.items():
                    l1 = l1.replace(pattern, replace)
                    l2 = l2.replace(pattern, replace)
                # # 大小写转换
                # l1 = l1.lower()
                # l2 = l2.lower()
                # wmt的处理
                l1 = re_space.sub("", l1)
                l1 = l1.replace(",", u"\uFF0C")
                l1 = re_final_comma.sub(u"\u3002", l1)


                l1_list.append(l1.strip() + '\n')
                l2_list.append(l2.strip() + '\n')

                if len(l1_list) > 300:
                    f1_w.writelines(l1_list)
                    f2_w.writelines(l2_list)
                    l1_list = []
                    l2_list = []

        if len(l1_list) > 0:
            f1_w.writelines(l1_list)
            f2_w.writelines(l2_list)
    print("delete: ", delete)


def tttt():
    filename = "/home/user_data55/wangdq/code/njunmt/log/uy2zh/outputmt01.0"
    new_filename = "/home/user_data55/wangdq/code/njunmt/log/uy2zh/outputmt01.00"
    c = []
    detokenizer = MosesDetokenizer(lang="zh")
    detrucaser = MosesDetruecaser()
    with open(filename, 'r') as f1, open(new_filename, 'w') as f2:
        for l in f1:
            # t = l.strip().split(' ')
            # c.append(''.join(t))
            l = detrucaser.detruecase(l, return_str=False)
            l = detokenizer.detokenize(l, return_str=True)
            c.append(l + '\n')
            if len(c) > 300:
                f2.writelines(c)
                c = []

        if len(c) > 0:
            f2.writelines(c)

def count_length(filename1, filename2):
    len1 = {}
    _len1 = 0
    len2 = {}
    _len2 = 0
    count = 0
    l = {}

    with open(filename1, 'r') as f1, open(filename2, 'r') as f2:
        for l1, l2 in zip(f1, f2):
            ll1 = len(l2.split(' '))
            ll2 = len(l1.split(' '))
            # _len1 += ll1
            # _len2 += ll2
            # len1.setdefault(int(ll1/10), 0)
            # len1[int(ll1/10)] += 1
            # len2.setdefault(int(ll2/10), 0)
            # len2[int(ll2/10)] += 1
            count += 1
            l.setdefault(int(ll1 / ll2), 0)
            l[int(ll1 / ll2)] += 1
    print(len1)
    print(len2)
    print(count)
    print(_len1)
    print(_len2)
    print(_len1 / count)
    print(_len2 / count)
    print(l)


def cut_by_length(f1, f2, len1=100, len2=100, ld1=3, ld2=6):
    l1_list = []
    l2_list = []
    delete = 0
    delete2 = 0
    with open(f1 + ".new", 'w') as f1_w, open(f2 + ".new", 'w') as f2_w:
        with open(f1, 'r') as f1, open(f2, 'r') as f2:
            for l1, l2 in zip(f1, f2):
                l1_len = len(l1.split(" "))
                l2_len = len(l2.split(" "))

                if l1_len > len1 or  l2_len > len2:
                    delete += 1
                    continue
                if l1_len/l2_len > ld1 or l2_len/l1_len > ld2:
                    delete2 += 1
                    continue

                l1_list.append(l1)
                l2_list.append(l2)

                if len(l1_list) > 100:
                    f1_w.writelines(l1_list)
                    f2_w.writelines(l2_list)
                    l1_list = []
                    l2_list = []

            if len(l1_list) > 0:
                f1_w.writelines(l1_list)
                f2_w.writelines(l2_list)
    print(delete)
    print(delete2)


if __name__ == '__main__':
    tttt()
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
