#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export num=$2
export SRC="zh"
export TGT="en"
export LAN="${SRC}2${TGT}"
export CON="configs/transformer_nist_zh2en_bpe.yaml"
export MT="mt0${num}"
export SAVETODIR="/home/user_data55/wangdq/code/njunmt/log/${LAN}/"
mkdir -p $SAVETODIR

python -m src.bin.translate \
    --model_name "transformer" \
    --source_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/${MT}.src" \
    --ref_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/${MT}.ref" \
    --model_path "/home/user_data55/wangdq/model/nist_zh2en/save/transformer.best.final" \
    --config_path "${CON}" \
    --batch_size 20 \
    --beam_size 5 \
    --saveto "${SAVETODIR}/output${MT}" \
    --use_gpu  \
    --num_refs 4 \
    --sample_K $3  \
    --copy_head
