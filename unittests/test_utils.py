# MIT License

# Copyright (c) 2018 the NJUNMT-pytorch authors.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

import argparse
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import subprocess


def build_test_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true", help="Running test on GPU")

    return parser


def get_model_name(path):
    return os.path.basename(path).strip().split(".")[0]


def clean_tmp_dir(path):
    subprocess.run("rm -rf {0}/*".format(path), shell=True)


def rm_tmp_dir(path):
    subprocess.run("rm -rf {0}".format(path), shell=True)


from xml.dom.minidom import parse


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
    zh_fn = os.path.join(os.path.dirname(zh_file_name), 'dev.en')
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

def reverse_text():
    lines = 0
    contents = []
    filename = "/home/user_data55/wangdq/data/ccmt/zh-en/parallel/train.en"
    with open(filename, 'r') as f:
        with open(filename + '.reverse', 'w') as f2:
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

# if __name__ == '__main__':
#     #     zh_file_name = '/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/newstest2019-zhen-src.zh.sgm'
#     #     en_file_name = '/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/newstest2019-zhen-ref.en.sgm'
#     #     # process_dev(zh_file_name, en_file_name)
#     #     process_dev_as_text(zh_file_name)
#     #     process_dev_as_text(en_file_name)
#     split_en()
