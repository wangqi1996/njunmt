#!/usr/bin/env bash

lang=zh
filename=$lang.token
nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#python $nmt_script/reverse.py $filename

#mv $filename.reverse reverse/
#cd reverse
#python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input $lang.token.reverse --output $lang.reverse.code
#python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input $lang.token.reverse --codes $lang.code --output $lang.token.reverse.bpe
#python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input $lang.token.bpe --output $lang.reverse.vocab
