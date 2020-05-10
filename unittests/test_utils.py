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

# if __name__ == '__main__':
# #     #     zh_file_name = '/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/newstest2019-zhen-src.zh.sgm'
# #     #     en_file_name = '/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/newstest2019-zhen-ref.en.sgm'
# #     #     # process_dev(zh_file_name, en_file_name)
# #     #     process_dev_as_text(zh_file_name)
# #     #     process_dev_as_text(en_file_name)
# #     split_en()
#     process_mono()
