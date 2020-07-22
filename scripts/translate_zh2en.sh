#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="zh"
export TGT="en"
export LAN="${SRC}2${TGT}"
export CON="configs/zh2en/transformer_zh2en_bpe_new_d2.yaml"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/zh-en/output/zh2en/"
mkdir -p $SAVETODIR

python -m src.bin.translate \
    --model_name "transformer" \
    --source_path "/home/user_data55/wangdq/data/ccmt/zh-en/test/zh2en/test.zh.token" \
    --model_path "/home/wangdq/save/zh2en/base2/transformer.best.final" \
    --config_path "${CON}" \
    --batch_size 40 \
    --beam_size 5 \
    --alpha 1.3 \
    --saveto "${SAVETODIR}/dev.out" \
    --use_gpu  \
    --num_refs 1
