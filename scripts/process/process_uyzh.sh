#!/usr/bin/env bash
# 1. 解压缩数据并且cat
#cat 2015/XJTIPC-2015-3w-ch.txt 2015/XJTIPC-3w-ch.txt 2017/ch.txt 2018/ch.txt ICT-UC-corpus-CWMT2017/Ch.txt > ../zh.txt
#cat 2015/XJTIPC-2015-3w-uy.txt 2015/XJTIPC-3w-uy.txt 2017/uy.txt 2018/uy.txt ICT-UC-corpus-CWMT2017/Uy.txt > ../uy.txt

# 2. 运行unique文件, 去重，可选全角转半角
nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#python $nmt_script/unique.py zh.txt uy.txt
#mv zh.txt.new zh.txt
#mv uy.txt.new uy.txt
#
## 3. \x1e会影响bpe，维汉没影响
##mv zh.txt.new zh.txt
##mv en.txt.new en.txt
##recsep=$(echo -e '\x1e')
##paste zh.txt uy.txt  | grep -v "$recsep" > tsv
##cut -f 1 tsv > zh.txt.n
##cut -f 2 tsv > en.txt.n
#
#mv zh.txt.new zh.txt
#mv uy.txt.new uy.txt
## 4. 调用mosedir脚本
mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
#
## 转义
#{
#perl $mosedir/deescape-special-chars.perl < uy.txt > uy.txt.n
#mv uy.txt.n uy.txt
#
# # normalize-punctuation.perl  汉语和维语统一处理
#perl $mosedir/normalize-punctuation.perl -l uy < uy.txt > uy.txt.n
#mv uy.txt.n uy.txt
#
#mv uy.txt uy.token
#} &
### tokenizer 只用中文
##
#bash $nmt_script/process_raw_zh.sh &
#wait
##
### 长度和长度比例提出
##python $nmt_script/length.py count zh.token uy.token
#python $nmt_script/length.py cut zh.token uy.token  100 100 6 4 # 设置长度、长度比例！！否则用默认的
#
#mv zh.token.new zh.token
#mv uy.token.new uy.token
#
## fast-align 词对齐 https://github.com/clab/fast_align 首先运行merge函数，合并 没有消失
#python $nmt_script/merge.py zh.token uy.token token
#fast_align_dir=/home/user_data55/wangdq/code/fast_align/build
#$fast_align_dir/fast_align -i token -d -o -v > zh_uy_align
#$fast_align_dir/fast_align -i token -d -o -v -r > uy_zh_align
#$fast_align_dir/atools -i zh_uy_align -j uy_zh_align -c grow-diag-final-and
#python $nmt_script/align.py zh.token uy.token zh_uy_align uy_zh_align 0.6

## 合并平行语料
#cat zh.token /home/user_data55/wangdq/data/ccmt/uy-zh/parallel/zh.token > zh.token.new &
#cat uy.token /home/user_data55/wangdq/data/ccmt/uy-en/parallel/uy.token > uy.token.new &
#wait
#mv zh.token.new zh.token
#mv uy.token.new uy.token
#

cd /home/user_data55/wangdq/data/ccmt/uy-zh/sample3
cat ../monolingual/sample3/uy.token ../parallel/uy.token > uy.token &
cat ../monolingual/sample3/zh.token ../parallel/zh.token > zh.token &
wait

python $nmt_script/o_unique.py zh.token uy.token
mv zh.token.new zh.token
mv uy.token.new uy.token


cd  /home/user_data55/wangdq/data/ccmt/uy-zh/sample_all
cat ../sample/zh.token ../sample2/zh.token ../sample3/zh.token > zh.token &
cat ../sample/uy.token ../sample2/uy.token ../sample3/uy.token > uy.token   &
wait

python $nmt_script/o_unique.py zh.token uy.token
mv zh.token.new zh.token
mv uy.token.new uy.token


{
python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input zh.token --output zh.code
python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input zh.token --codes zh.code --output zh.token.bpe
python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input zh.token.bpe --output zh.vocab
} &

{
python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input uy.token --output uy.code &&
python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input uy.token --codes uy.code --output uy.token.bpe &&
python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input uy.token.bpe --output uy.vocab
} &
#
wait
echo "done"

# -----------------------------------------------------------------------------------------------------------------------

## dev && test
#mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
#nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#
#cd /home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/data/CWMT2018-TestSet-UC
#perl $mosedir/../ems/support/input-from-sgm.perl  < CWMT2018-uc-news-test-ref.xml > dev.zh
#perl $mosedir/../ems/support/input-from-sgm.perl  < CWMT2018-uc-news-test-src.xml > dev.uy
#
## 转义
#perl $mosedir/deescape-special-chars.perl < dev.zh > dev.zh.n
#perl $mosedir/deescape-special-chars.perl  < dev.uy > dev.uy.n
#mv dev.zh.n dev.zh
#mv dev.uy.n dev.uy
#
# # normalize-punctuation.perl  汉语和维语统一处理，有必要做吗？
#perl $mosedir/normalize-punctuation.perl -l zh < dev.zh > dev.zh.n
#perl $mosedir/normalize-punctuation.perl -l uy < dev.uy > dev.uy.n
#mv dev.zh.n dev.zh
#mv dev.uy.n dev.uy
#
## tokenizer 只用中文
#python $nmt_script/split_zh.py dev.zh dev.zh.token

# -----------------------------------------------------------------------------------------------------------------------

## 计算bleu-sgm
#predict_dir=/home/user_data55/wangdq/code/njunmt_dist/valid
#src_xml=/home/user_data55/wangdq/data/ccmt/uy-zh/test/CCMT2019_UC_test_src.xml
#mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
#nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#output=base_d3
#predict_dir=/home/user_data55/wangdq/data/ccmt/uy-zh/output
#mv test.out.0 $output.txt
## 分词逆操作 好像作用不大
#python $nmt_script/postprocess.py $predict_dir/$output.txt
#mv $predict_dir/$output.txt.new $predict_dir/$output.txt
#
## 转义回原始字符
#perl $mosedir/escape-special-chars.perl < $predict_dir/$output.txt > $predict_dir/$output.txt.new
#mv $predict_dir/$output.txt.new $predict_dir/$output.txt
#
#perl $mosedir/wrap-xml.perl $src_xml zh <  $predict_dir/$output.txt > $predict_dir/$output.xml
#
## 转移到window计算bleu-sbp
##1. 如果使用src_xml, 更新为tstset
#cd D:\Download\Mteval Scoring Tool(BLEU-SBP)\Mteval Scoring Tool(BLEU-SBP)\Tool for Windows\
#mteval_sbp.exe -s example\test_src.xml -r example\test_ref.xml  -t example\test.xml

# -----------------------------------------------------------------------------------------------------------------------
## 单语数据
#cd /home/user_data55/wangdq/data/ccmt/uy-zh/monolingual
#python $nmt_script/extract_mono.py xmu.txt
#
## 分割单句 split函数
#mv xmu.txt.new xmu.txt
## 去重
#mono_dir=/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual
#nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#python $nmt_script/s_unique.py $mono_dir/xmu.txt
#mv xmu.txt.new xmu.txt
#
## 转义
#perl $mosedir/deescape-special-chars.perl < xmu.txt > xmu.txt.new
#mv xmu.txt.new xmu.txt
#
## normalize-punctuation.perl
#perl $mosedir/normalize-punctuation.perl -l zh < xmu.txt > xmu.txt.new
#mv xmu.txt.new xmu.txt
#
## # tokenizer 只用中文
#python $nmt_script/split_zh.py xmu.txt xmu.token
#
## 长度剔除
#python $nmt_script/s_length.py count xmu.token
#python $nmt_script/s_length.py cut xmu.token 120
#mv xmu.token.new xmu.token

# -----------------------------------------------------------------------------------------------------------------------
## BT数据处理
#
## 2. 运行unique文件, 去重，可选全角转半角
#nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process
#data_path=/home/user_data55/wangdq/data/ccmt/uy-zh/BT/
#python $nmt_script/unique.py  $data_path/zh.token $data_path/uy.token
#
#mv zh.token.new zh.token
#mv uy.token.new uy.token
#
#python $nmt_script/split_zh.py zh.txt zh.token
## 长度
#python $nmt_script/length.py count zh.token uy.token
#python $nmt_script/length.py cut zh.token uy.token  100 100 8 8