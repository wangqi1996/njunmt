#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="uy"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export CON="/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final/sample_big_ave.yaml"
#export CON="configs/uy2zh/transformer_uy2zh_sample_big.yaml"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/uy-zh/output/"
mkdir -p $SAVETODIR


python -m src.bin.translate \
    --model_name "transformer" \
    --source_path "/home/user_data55/wangdq/data/ccmt/uy-zh/test/test.txt" \
    --model_path "/home/user_data55/wangdq/save/uy2zh/final/uy2zh_sample_big_ave/transformer.best.final"  \
    --config_path "${CON}" \
    --batch_size 20 \
    --beam_size 5 \
    --alpha 1.5 \
    --use_gpu  \
    --saveto "${SAVETODIR}/sample_ave_final.out" \
    --num_refs 1  \
#    --sample_search_k 2