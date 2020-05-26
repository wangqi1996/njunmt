#!/usr/bin/env bash

#cat Book1_en.txt Book2_en.txt Book3_en.txt Book4_en.txt Book5_en.txt Book6_en.txt Book7_en.txt Book8_en.txt Book9_en.txt Book10_en.txt Book11_en.txt Book12_en.txt Book13_en.txt Book14_en.txt Book15_en.txt Book16_en.txt Book17_en.txt Book18_en.txt Book19_en.txt Book20_en.txt  > Book_en.txt

#cat Book1_cn.txt Book2_cn.txt Book3_cn.txt Book4_cn.txt Book5_cn.txt Book6_cn.txt Book7_cn.txt Book8_cn.txt Book9_cn.txt Book10_cn.txt Book11_cn.txt Book12_cn.txt Book13_cn.txt Book14_cn.txt Book15_cn.txt Book16_cn.txt Book17_cn.txt Book18_cn.txt Book19_cn.txt Book20_cn.txt  > Book_cn.txt

# 拼接
cd /home/user_data55/wangdq/data/ccmt/zh-en/parallel/data
cat casia2015/casia2015_en.txt casict2011/casict-A_en.txt casict2011/casict-B_en.txt casict2015/casict2015_en.txt datum2015/datum_en.txt  datum2017/Book_en.txt neu2017/NEU_en.txt > ../en.txt

cat casia2015/casia2015_ch.txt casict2011/casict-A_ch.txt casict2011/casict-B_ch.txt casict2015/casict2015_ch.txt datum2015/datum_ch.txt  datum2017/Book_cn.txt neu2017/NEU_ch.txt > ../zh.txt
cd ../

# 调用python函数 去重、转义

# \x1e不能用作bpe
mv zh.txt.new zh.txt
mv en.txt.new en.txt
recsep=$(echo -e '\x1e')
paste zh.txt en.txt  | grep -v "$recsep" > tsv
cut -f 1 tsv > zh.txt.n
cut -f 2 tsv > en.txt.n

mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
# 转义
mv zh.txt.n zh.txt
mv en.txt.n en.txt
perl $mosedir/deescape-special-chars.perl -l zh < zh.txt > zh.txt.n
perl $mosedir/deescape-special-chars.perl -l en < en.txt > en.txt.n

#  normalize-punctuation.perl
mv zh.txt.n zh.txt
mv en.txt.n en.txt
perl $mosedir/normalize-punctuation.perl -l zh < zh.txt > zh.token
perl $mosedir/normalize-punctuation.perl -l en < en.txt > en.token

# tokenizer
perl $mosedir/tokenizer.perl -a -l en -threads 10 < en.token > en.token.new
perl $mosedir/deescape-special-chars.perl -l en < en.token > en.token.new




# giza++ 词对齐
 ./plain2snt.out /home/user_data55/wangdq/data/ccmt/zh-en/parallel/zh.token /home/user_data55/wangdq/data/ccmt/zh-en/parallel/en.token
base=/home/user_data55/wangdq/data/ccmt/zh-en/parallel
./snt2cooc.out $base/zh.token.vcb $base/en.token.vcb $base/zh.token_en.token.snt > $base/zh_en.cooc
./snt2cooc.out $base/en.token.vcb $base/zh.token.vcb $base/en.token_zh.token.snt > $base/en_zh.cooc

 ./GIZA++ -S  $base/zh.token.vcb -T $base/en.token.vcb -C $base/zh.token_en.token.snt  -CoocurrenceFile $base/zh_en.cooc -o zh2en -OutputPath zh2en
 ./GIZA++ -S $base/en.token.vcb  -T $base/zh.token.vcb -C $base/en.token_zh.token.snt  -CoocurrenceFile $base/en_zh.cooc -o en2zh -OutputPath en2zh

echo "using bpe"
export lang=$1
subword-nmt learn-bpe -s 32000 < /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.txt > /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.code
subword-nmt get-vocab --input /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.txt --output /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.vocab


# dev && test
cd /home/user_data55/wangdq/data/ccmt/zh-en/dev2020
perl $mosedir/../ems/support/input-from-sgm.perl  < newstest2019-zhen-ref.en.sgm > dev.en
perl $mosedir/../ems/support/input-from-sgm.perl  < newstest2019-zhen-src.zh.sgm > dev.zh

# 调用python函数去重
mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
# 转义
perl $mosedir/deescape-special-chars.perl -l zh < dev.zh > dev.zh.n
perl $mosedir/deescape-special-chars.perl -l en < dev.en > dev.en.n

# 标点符号
mv dev.zh.n dev.zh
mv dev.en.n dev.en
perl $mosedir/normalize-punctuation.perl -l zh < dev.zh  > dev.zh.token2
perl $mosedir/normalize-punctuation.perl -l en <  dev.en  > dev.en.token2
mv dev.zh.token2 dev.zh
mv dev.en.token2 dev.en

# tokenizer
perl $mosedir/tokenizer.perl -a -l en -threads 10 < dev.en > dev.en.token
perl $mosedir/deescape-special-chars.perl -l en < dev.en.token > dev.en.n