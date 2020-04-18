#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.distributed.launch  \
	--nproc_per_node=4 --nnodes=1 --node_rank=0 \
	--master_addr="127.0.0.1" --master_port=11234
    src.bin.train \
    --model_name "transformer" \
    --reload \
    --config_path "./configs/transformer_nist_zh2en_bpe.yaml" \
    --log_path "./log/1/" \
    --saveto "./save/1/" \
    --use_gpu