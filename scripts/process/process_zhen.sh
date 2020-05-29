#!/usr/bin/env bash

#cat Book1_en.txt Book2_en.txt Book3_en.txt Book4_en.txt Book5_en.txt Book6_en.txt Book7_en.txt Book8_en.txt Book9_en.txt Book10_en.txt Book11_en.txt Book12_en.txt Book13_en.txt Book14_en.txt Book15_en.txt Book16_en.txt Book17_en.txt Book18_en.txt Book19_en.txt Book20_en.txt  > Book_en.txt

#cat Book1_cn.txt Book2_cn.txt Book3_cn.txt Book4_cn.txt Book5_cn.txt Book6_cn.txt Book7_cn.txt Book8_cn.txt Book9_cn.txt Book10_cn.txt Book11_cn.txt Book12_cn.txt Book13_cn.txt Book14_cn.txt Book15_cn.txt Book16_cn.txt Book17_cn.txt Book18_cn.txt Book19_cn.txt Book20_cn.txt  > Book_cn.txt

# 拼接
cd /home/user_data55/wangdq/data/ccmt/zh-en/parallel/data
cat casia2015/casia2015_en.txt casict2011/casict-A_en.txt casict2011/casict-B_en.txt casict2015/casict2015_en.txt datum2015/datum_en.txt  datum2017/Book_en.txt neu2017/NEU_en.txt > ../en.txt
cat casia2015/casia2015_ch.txt casict2011/casict-A_ch.txt casict2011/casict-B_ch.txt casict2015/casict2015_ch.txt datum2015/datum_ch.txt  datum2017/Book_cn.txt neu2017/NEU_ch.txt > ../zh.txt
cd ../
# 1050000 936654 999979 2036833 1000003
nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
data_path=/home/user_data55/wangdq/data/ccmt/zh-en/parallel/
# 9023453

# 去掉只有中文 or 只有英文的 7740370
python $nmt_script/cut_mono.py zh.token en.token

# 调用python函数 去重、转义 -1278238=7745215
python $nmt_script/unique.py $data_path/zh.txt $data_path/en.txt
mv zh.txt.new zh.txt
mv en.txt.new en.txt

# \x1e不能用作bpe，不过貌似没啥影响
#mv zh.txt.new zh.txt
#mv en.txt.new en.txt
#recsep=$(echo -e '\x1e')
#paste zh.txt en.txt  | grep -v "$recsep" > tsv
#cut -f 1 tsv > zh.txt.n
#cut -f 2 tsv > en.txt.n

mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
# 转义
perl $mosedir/deescape-special-chars.perl -l zh < zh.txt > zh.txt.n
perl $mosedir/deescape-special-chars.perl -l en < en.txt > en.txt.n

#  normalize-punctuation.perl
mv zh.txt.n zh.txt
mv en.txt.n en.txt
cat zh.txt | perl $mosedir/replace-unicode-punctuation.perl | perl $mosedir/normalize-punctuation.perl -l zh  > zh.txt2
cat en.txt | perl $mosedir/replace-unicode-punctuation.perl  | perl $mosedir/normalize-punctuation.perl -l en > en.txt2
mv zh.txt2 zh.txt
mv en.txt2 en.txt
# tokenizer 7745215
perl $mosedir/tokenizer.perl -a -l en -threads 10 < en.txt > en.token
perl $mosedir/deescape-special-chars.perl -l en < en.token > en.token.new
mv en.token.new en.token

python $nmt_script/split_zh.py zh.txt zh.token
 # -----------------------------------------------------------------------

# 长度筛选 7740379
python $nmt_script/length.py count  zh.token en.token
python $nmt_script/length.py cut zh.token en.token 110 120 3 4
mv zh.token.new zh.token
mv en.token.new en.token

# 去掉只有中文 or 只有英文的 7740370
python $nmt_script/cut_mono.py zh.token en.token


# fast-align 词对齐 https://github.com/clab/fast_align 首先运行merge函数，合并  不好，只删除了8个
python $nmt_script/merge.py zh.token en.token token
fast_align_dir=/home/user_data55/wangdq/code/fast_align/build
$fast_align_dir/fast_align -i token -d -o -v > zh_en_align
$fast_align_dir/fast_align -i token -d -o -v -r > en_zh_align
$fast_align_dir/atools -i zh_en_align -j en_zh_align -c grow-diag-final-and
python $nmt_script/align.py zh.token en.token zh_en_align en_zh_align
mv zh.token.new zh.token
mv en.token.new en.token

echo "using bpe"
export lang=zh
subword-nmt learn-bpe -s 32000 < $data_path/${lang}.token > $data_path/${lang}.code
subword-nmt apply-bpe -c $data_path/${lang}.code < $data_path/${lang}.token > $data_path/${lang}.token.bpe
subword-nmt get-vocab --input $data_path/${lang}.token.bpe --output $data_path/${lang}.vocab

export lang=en
subword-nmt learn-bpe -s 32000 < $data_path/${lang}.token > $data_path/${lang}.code
subword-nmt apply-bpe -c $data_path/${lang}.code < $data_path/${lang}.token > $data_path/${lang}.token.bpe
subword-nmt get-vocab --input $data_path/${lang}.token.bpe --output $data_path/${lang}.vocab


# dev && test
mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
cd /home/user_data55/wangdq/data/ccmt/zh-en/dev2020
perl $mosedir/../ems/support/input-from-sgm.perl  < newstest2019-zhen-ref.en.sgm > dev.en
perl $mosedir/../ems/support/input-from-sgm.perl  < newstest2019-zhen-src.zh.sgm > dev.zh

ref=/home/user_data55/wangdq/data/ccmt/zh-en/multi-reference/CWMT2008/ec_ref.txt
src=/home/user_data55/wangdq/data/ccmt/zh-en/multi-reference/CWMT2008/ec_src.txt

# 转义
perl $mosedir/deescape-special-chars.perl < $src > $src.new
perl $mosedir/deescape-special-chars.perl  < $ref > $ref.new
mv $src.new $src
mv $ref.new $ref

# 标点符号
cat $src | perl $mosedir/replace-unicode-punctuation.perl | perl $mosedir/normalize-punctuation.perl -l zh  > $src.new
cat $ref | perl $mosedir/replace-unicode-punctuation.perl  | perl $mosedir/normalize-punctuation.perl -l en > $ref.new
mv $src.new $src
mv $ref.new $ref

en=$src
zh=$ref
# tokenizer
perl $mosedir/tokenizer.perl -a -l en -threads 10 < $en > $en.token
perl $mosedir/deescape-special-chars.perl -l en < $en.token > $en.token.new
mv $en.token.new $en.token
python $nmt_script/split_zh.py $zh $zh.token
