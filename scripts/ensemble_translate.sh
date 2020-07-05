#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="uy"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/uy-zh/output/"
export CON="/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final"
export MODEL="/home/user_data55/liuzh/ccmt_model/uy2zh"
export Post="transformer.best.final"
mkdir -p $SAVETODIR

python -m  src.bin.ensemble_translate \
    --model_name "transformer" \
    --config_path "${CON}/transformer_uy2zh_sample_big.yaml,${CON}/transformer_uy2zh_sample2_big.yaml,${CON}/transformer_uy2zh_sample3_big.yaml,${CON}/sample2_char.yaml,${CON}/sample_big_relative.yaml"  \
    --source_path "/home/user_data55/wangdq/data/ccmt/uy-zh/test/test.txt" \
    --model_path "${MODEL}/sample_big/${Post},${MODEL}/sample2_big/${Post},${MODEL}/sample3_big/${Post},${MODEL}/uy2zh_sample2_char/${Post},${MODEL}/sample_big_relative/transformer_relative.best.final"  \
    --batch_size 20 \
    --beam_size 5 \
    --alpha 1.5 \
    --use_gpu  \
    --saveto "${SAVETODIR}/all_sample2.out"