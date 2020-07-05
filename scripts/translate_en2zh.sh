#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="en"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export CON="configs/en2zh/transformer_en2zh_bpe_new.yaml"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/zh-en/output/en2zh/"
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
#    --ref_path "/home/user_data55/wangdq/data/ccmt/zh-en/dev2020/dev.zh"  \
#python -m src.distributed.launch  \
#	--nproc_per_node=4 --nnodes=1 --node_rank=0 \
#	--master_addr="127.0.0.1" --master_port=11479 \
#    src.bin.translate \
python -m src.bin.translate \
    --model_name "transformer" \
    --source_path  "/home/user_data55/wangdq/data/ccmt/zh-en/test/en2zh/test.en.token" \
    --model_path "/home/wangdq/save/en2zh/base/transformer.best.final" \
    --config_path "${CON}" \
    --batch_size 40 \
    --beam_size 5 \
    --alpha 1.3 \
    --saveto "${SAVETODIR}/dev.out" \
    --use_gpu  \
    --num_refs 1
