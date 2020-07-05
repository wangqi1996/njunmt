#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------------------
# BT数据处理
# 合并

# 2. 运行unique文件, 去重，可选全角转半角
mosedir=/home/user_data55/wangdq/code/mosesdecoder/scripts/tokenizer
nmt_script=/home/user_data55/wangdq/code/njunmt_dist/scripts/process

data_path=.
file1=zh.txt
file2=uy.txt

python $nmt_script/o_unique.py  $data_path/$file1 $data_path/$file2
mv $file1.new $file1
mv $file2.new $file2

# 转义
perl $mosedir/deescape-special-chars.perl  < $file1 > $file1.new
perl $mosedir/deescape-special-chars.perl < $file2 > $file2.new
mv $file1.new $file1
mv $file2.new $file2

 # normalize-punctuation.perl  汉语和维语统一处理
perl $mosedir/normalize-punctuation.perl -l zh < $file1 > $file1.new
perl $mosedir/normalize-punctuation.perl -l uy < $file2 > $file2.new
mv $file1.new $file1
mv $file2.new $file2

# tokenizer
python $nmt_script/split_zh.py $file1 $file1.token
mv

# 长度
file1=zh.token
file2=en.token
python $nmt_script/length.py count $file1 $file2
python $nmt_script/length.py cut $file1 $file2 100 100 3 2
mv $file1.new $file1
mv $file2.new $file2

# 对齐
python $nmt_script/merge.py zh.token uy.token token
fast_align_dir=/home/user_data55/wangdq/code/fast_align/build
$fast_align_dir/fast_align -i token -d -o -v > zh_uy_align
$fast_align_dir/fast_align -i token -d -o -v -r > uy_zh_align
$fast_align_dir/atools -i zh_uy_align -j uy_zh_align -c grow-diag-final-and
python $nmt_script/align.py zh.token uy.token zh_uy_align uy_zh_align
mv $file1.new $file1
mv $file2.new $file2


# ---------------------------------------------------------------------------------------------------------------------------------------

# 拼接
cat ../monolingual/all/zh.token ../parallel/zh.token > zh.token
cat ../monolingual/all/uy.token ../parallel/uy.token > uy.token

python $nmt_script/o_unique.py zh.token uy.token

mv $file1.new $file1
mv $file2.new $file2
echo "using bpe"
export lang=zh
python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input zh.token --output zh.code
python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input zh.token --codes zh.code --output zh.token.bpe
python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input zh.token.bpe --output zh.vocab

python  /home/user_data55/wangdq/code/subword-nmt/learn_bpe.py -s 32000 --input en.token --output en.code &&
python  /home/user_data55/wangdq/code/subword-nmt/apply_bpe.py  --input en.token --codes en.code --output en.token.bpe &&
python /home/user_data55/wangdq/code/subword-nmt/get_vocab.py --input en.token.bpe --output en.vocab

subword-nmt learn-bpe -s 32000 < $data_path/${lang}.token > $data_path/${lang}.code
subword-nmt apply-bpe -c $data_path/${lang}.code < $data_path/${lang}.token > $data_path/${lang}.token.bpe
subword-nmt get-vocab --input $data_path/${lang}.token.bpe --output $data_path/${lang}.vocab

export lang=uy
subword-nmt learn-bpe -s 32000 < $data_path/${lang}.token > $data_path/${lang}.code
subword-nmt apply-bpe -c $data_path/${lang}.code < $data_path/${lang}.token > $data_path/${lang}.token.bpe
subword-nmt get-vocab --input $data_path/${lang}.token.bpe --output $data_path/${lang}.vocab

