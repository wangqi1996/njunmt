#!/usr/bin/env bash

# 1. 拼接
#cat Book1_en.txt Book2_en.txt Book3_en.txt Book4_en.txt Book5_en.txt Book6_en.txt Book7_en.txt Book8_en.txt Book9_en.txt Book10_en.txt Book11_en.txt Book12_en.txt Book13_en.txt Book14_en.txt Book15_en.txt Book16_en.txt Book17_en.txt Book18_en.txt Book19_en.txt Book20_en.txt  > Book_en.txt
#cat Book1_cn.txt Book2_cn.txt Book3_cn.txt Book4_cn.txt Book5_cn.txt Book6_cn.txt Book7_cn.txt Book8_cn.txt Book9_cn.txt Book10_cn.txt Book11_cn.txt Book12_cn.txt Book13_cn.txt Book14_cn.txt Book15_cn.txt Book16_cn.txt Book17_cn.txt Book18_cn.txt Book19_cn.txt Book20_cn.txt  > Book_cn.txt
#cd /home/user_data55/wangdq/data/ccmt/zh-en/parallel/data
#cat casia2015/casia2015_en.txt casict2011/casict-A_en.txt casict2011/casict-B_en.txt casict2015/casict2015_en.txt datum2015/datum_en.txt  datum2017/Book_en.txt neu2017/NEU_en.txt > ../en.txt
#cat casia2015/casia2015_ch.txt casict2011/casict-A_ch.txt casict2011/casict-B_ch.txt casict2015/casict2015_ch.txt datum2015/datum_ch.txt  datum2017/Book_cn.txt neu2017/NEU_ch.txt > ../zh.txt
#cd ../


nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#data_path=/home/user_data55/wangdq/data/ccmt/zh-en/parallel/
zh_file=zh.txt
en_file=en.txt

# 2. 去掉只有中文 or 只有英文的
python $nmt_script/cut_mono.py $zh_file $en_file
mv $zh_file.new $zh_file
mv $en_file.new $en_file

# 调用python函数 去重、转义
python $nmt_script/unique.py $zh_file $en_file
mv $zh_file.new $zh_file
mv $en_file.new $en_file

# \x1e不能用作bpe，不过貌似没啥影响
#mv zh.txt.new zh.txt
#mv en.txt.new en.txt
#recsep=$(echo -e '\x1e')
#paste zh.txt en.txt  | grep -v "$recsep" > tsv
#cut -f 1 tsv > zh.txt.n
#cut -f 2 tsv > en.txt.n

mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
# 分开运行一下命令
bash process_raw_en.sh &
bash process_raw_zh.sh &
wait
# 转义
#perl $mosedir/deescape-special-chars.perl -l zh < $zh_file > $zh_file.new
#perl $mosedir/deescape-special-chars.perl -l en < $en_file > $en_file.new
#mv $zh_file.new $zh_file
#mv $en_file.new $en_file
#
##  normalize-punctuation.perl
#cat $zh_file | perl $mosedir/replace-unicode-punctuation.perl | perl $mosedir/normalize-punctuation.perl -l zh  > $zh_file.new
#cat $en_file | perl $mosedir/replace-unicode-punctuation.perl  | perl $mosedir/normalize-punctuation.perl -l en > $en_file.new
#mv $zh_file.new $zh_file
#mv $en_file.new $en_file
#
## tokenizer 7745215
#zh_token=zh.token
#en_token=en.token
#perl $mosedir/tokenizer.perl -a -l en -threads 10 < $en_file > $en_token
#perl $mosedir/deescape-special-chars.perl -l en < $en_token > $en_token.new
#mv $en_token.new $en_token

#python $nmt_script/split_zh.py $zh_file $zh_token
 # -----------------------------------------------------------------------

zh_token=zh.token
en_token=en.token
# 长度筛选 7740379
#python $nmt_script/length.py count  $zh_token $en_token
python $nmt_script/length.py cut $zh_token $en_token 110 120 3 4
mv $zh_token.new $zh_token
mv $en_token.new $en_token


# fast-align 词对齐 https://github.com/clab/fast_align 首先运行merge函数，合并  不好，只删除了8个
#python $nmt_script/merge.py $zh_token $en_token token
#fast_align_dir=/home/user_data55/wangdq/code/fast_align/build
#$fast_align_dir/fast_align -i token -d -o -v > zh_en_align
#$fast_align_dir/fast_align -i token -d -o -v -r > en_zh_align
##$fast_align_dir/atools -i zh_en_align -j en_zh_align -c grow-diag-final-and 这个不用
#python $nmt_script/align.py $zh_token $en_token zh_en_align en_zh_align 0.6
#mv $zh_token.new $zh_token
#mv $en_token.new $en_token

# 合并平行语料
cat zh.token /home/user_data55/wangdq/data/ccmt/zh-en/parallel/zh.token > zh.token.new &
cat en.token /home/user_data55/wangdq/data/ccmt/zh-en/parallel/en.token > en.token.new &
wait
mv zh.token.new zh.token
mv en.token.new en.token

python $nmt_script/o_unique.py $zh_file $en_file
mv $zh_file.new $zh_file
mv $en_file.new $en_file

{
python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input zh.token --output zh.code
python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input zh.token --codes zh.code --output zh.token.bpe
python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input zh.token.bpe --output zh.vocab
} &

{
python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input en.token --output en.code &&
python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input en.token --codes en.code --output en.token.bpe &&
python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input en.token.bpe --output en.vocab
} &

wait
echo "done"

# dev && test
#mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
#cd /home/user_data55/wangdq/data/ccmt/zh-en/dev2020
#perl $mosedir/../ems/support/input-from-sgm.perl  < newstest2019-zhen-ref.en.sgm > dev.en
#perl $mosedir/../ems/support/input-from-sgm.perl  < newstest2019-zhen-src.zh.sgm > dev.zh
#
#ref=/home/user_data55/wangdq/data/ccmt/zh-en/multi-reference/CWMT2008/ec_ref.txt
#src=/home/user_data55/wangdq/data/ccmt/zh-en/multi-reference/CWMT2008/ec_src.txt
#
## 转义
#perl $mosedir/deescape-special-chars.perl < $src > $src.new
#perl $mosedir/deescape-special-chars.perl  < $ref > $ref.new
#mv $src.new $src
#mv $ref.new $ref
#
## 标点符号
#cat $src | perl $mosedir/replace-unicode-punctuation.perl | perl $mosedir/normalize-punctuation.perl -l zh  > $src.new
#cat $ref | perl $mosedir/replace-unicode-punctuation.perl  | perl $mosedir/normalize-punctuation.perl -l en > $ref.new
#mv $src.new $src
#mv $ref.new $ref
#
#en=$src
#zh=$ref
## tokenizer
#perl $mosedir/tokenizer.perl -a -l en -threads 10 < $en > $en.token
#perl $mosedir/deescape-special-chars.perl -l en < $en.token > $en.token.new
#mv $en.token.new $en.token
#python $nmt_script/split_zh.py $zh $zh.token
