#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export num=$2
export SRC="uy"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export CON="configs/uy2zh/transformer_uy2zh_bpe_new.yaml"
export MT="mt0${num}"
export SAVETODIR="/home/user_data55/wangdq/code/njunmt/log/${LAN}/"
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
    --source_path "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.uy" \
    --ref_path "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh.raw" \
    --model_path "/home/user_data55/wangdq/code/njunmt_dist/save/uy2zh/uy2zh_base/transformer.best.final" \
    --config_path "${CON}" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "${SAVETODIR}/output${MT}" \
    --use_gpu  \
    --num_refs 1
