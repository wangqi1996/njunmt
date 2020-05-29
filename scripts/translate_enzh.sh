#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="en"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export CON="configs/en2zh/transformer_en2zh_bpe_new.yaml"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/zh-en/output/"
mkdir -p $SAVETODIR

#python -m src.bin.translate \
#    --model_name "transformer" \
#    --source_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/${MT}.src" \
#    --ref_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/${MT}.ref" \
#    --model_path "/home/user_data55/wangdq/code/njunmt_dist/save/zh2en_baseline/transformer.best.final" \
#    --config_path "${CON}" \
#    --batch_size 20 \
#    --beam_size 5 \
#    --saveto "${SAVETODIR}/output${MT}" \
#    --use_gpu  \
#    --num_refs 1

python -m src.bin.translate \
    --model_name "transformer" \
    --source_path "/home/user_data55/wangdq/data/ccmt/zh-en/test/en2zh/test.en.token" \
    --model_path "/home/wangdq/save/en2zh/base/transformer.best.final" \
    --config_path "${CON}" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "${SAVETODIR}/en2zh.out" \
    --use_gpu
