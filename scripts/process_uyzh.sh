#!/usr/bin/env bash

cat 2015/XJTIPC-2015-3w-ch.txt 2015/XJTIPC-3w-ch.txt 2017/ch.txt 2018/ch.txt ICT-UC-corpus-CWMT2017/Ch.txt > zh.txt


cat 2015/XJTIPC-2015-3w-uy.txt 2015/XJTIPC-3w-uy.txt 2017/uy.txt 2018/ch.txt ICT-UC-corpus-CWMT2017/Uy.txt > uy.txt

echo "using bpe"

export lang=$1


subword-nmt learn-bpe -s 32000 < /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.txt > /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.code

subword-nmt get-vocab --input /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.txt --output /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/${lang}.vocab

perl wrap-xml.perl /home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/data/d/CWMT2018-uc-news-test-ref.xml zh < /home/user_data55/wangdq/code/njunmt/log/uy2zh/outputmt01.00 > /home/user_data55/wangdq/code/njunmt/log/uy2zh/outputmt01.00.xml
