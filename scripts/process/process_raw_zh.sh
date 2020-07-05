#!/usr/bin/env bash
# 4. 调用mosedir脚本
mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process

# 转义
perl $mosedir/deescape-special-chars.perl  < zh.txt > zh.txt.n
mv zh.txt.n zh.txt

 # normalize-punctuation.perl  汉语和维语统一处理
perl $mosedir/normalize-punctuation.perl -l zh < zh.txt > zh.txt.n
mv zh.txt.n zh.txt

# tokenizer 只用中文
python $nmt_script/split_zh.py zh.txt zh.token