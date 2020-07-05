#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
# export num=$2
export SRC="zh"
export TGT="uy"
export LAN="${SRC}2${TGT}"
export CON="/home/user_data55/wangdq/code/njunmt_dist/configs/zh2uy/transformer_zh2uy_bpe_new2.yaml"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual/"
mkdir -p $SAVETODIR

python -m src.distributed.launch  \
	--nproc_per_node=4 --nnodes=1 --node_rank=0 \
	--master_addr="127.0.0.1" --master_port=11479 \
    src.bin.translate \
    --model_name "transformer" \
    --source_path "/home/user_data55/wangdq/data/ccmt/uy-zh/monolingual/all/zh.token" \
    --model_path "/home/user_data55/wangdq/model/zh2uy/transformer.best.final" \
    --config_path "${CON}" \
    --batch_size 128 \
    --alpha 0.8 \
    --beam_size 1 \
    --saveto "${SAVETODIR}/sample3.txt" \
    --use_gpu  \
    --sample_search_k 10
    # --num_refs 4 \
    # --copy_head
