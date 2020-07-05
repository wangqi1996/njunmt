# coding=utf-8
# !/bin/python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# LASER  Language-Agnostic SEntence Representations
# is a toolkit to calculate multilingual sentence embeddings
# and to use them for document classification, bitext filtering
# and mining
#
# --------------------------------------------------------
#
# Tool to extract subset of mined bitexts in a tsv.gz file

import argparse
import gzip
import os
import sys

from langconv import *

###############################################################################
#
# Main
#
###############################################################################

parser = argparse.ArgumentParser(description='Tool to extract bitext from the WikiMatrix')
parser.add_argument('--encoding', default='utf-8',
                    help='character encoding for input/output')
parser.add_argument('--tsv', type=str, required=False,
                    default="/home/user_data55/wangdq/data/wmt2020/paralle/data/matrix.tsv",
                    help='File with mined bitexts')
parser.add_argument('--src-lang', type=str, required=False, default="en",
                    help='Source language')
parser.add_argument('--trg-lang', type=str, required=False, default="zh",
                    help='Traget language')
parser.add_argument('--threshold', type=float, default=1.05,
                    help='Threshold on margin score')
parser.add_argument('--nb-sents', type=int, default=999999999,
                    help='Maximal number of sentences')
args = parser.parse_args()

print('Tool to extract bitext from the WikiMatrix')

nl = 0
nw_src = 0
nw_trg = 0
print('Processing {}'.format(args.tsv))
with open(args.tsv, 'r', encoding=args.encoding) as tsv:
    with open(args.tsv + '.' + args.src_lang, 'w', encoding=args.encoding) as fsrc:
        with open(args.tsv + '.' + args.trg_lang, 'w', encoding=args.encoding) as ftrg:
            while nl < args.nb_sents:
                line = tsv.readline()
                if not line:
                    break
                fields = line.split('\t')
                cur_src = len(fields[1].split())
                cur_trg = len(fields[2].split())
                if float(fields[0]) < args.threshold:
                    break
                fsrc.write(fields[1].strip() + '\n')
                t = Converter('zh-hans').convert(fields[2].strip())
                ftrg.write(t + '\n')
                nw_src += cur_src
                nw_trg += cur_trg

                nl += 1
                if nl % 100000 == 0:
                    print('\r - {:d} lines read'.format(nl), end='')

print('\r - wrote {:d} lines'.format(nl))
print(' - with {:d} source and {:d} target words'.format(nw_src, nw_trg))
print(' - last threshold is {:.4f}'.format(float(fields[0])))
