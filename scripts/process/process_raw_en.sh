#!/usr/bin/env bash
# 4. 调用mosedir脚本
mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process

en_file=en.txt
perl $mosedir/deescape-special-chars.perl -l en < $en_file > $en_file.new
mv $en_file.new $en_file

#  normalize-punctuation.perl
cat $en_file | perl $mosedir/replace-unicode-punctuation.perl  | perl $mosedir/normalize-punctuation.perl -l en > $en_file.new
mv $en_file.new $en_file

# tokenizer 7745215
en_token=en.token
perl $mosedir/tokenizer.perl -a -l en -threads 10 < $en_file > $en_token
perl $mosedir/deescape-special-chars.perl -l en < $en_token > $en_token.new
mv $en_token.new $en_token