#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
    --model_name "transformer" \
    --config_path "./configs/uy2zh/transformer_uy2zh_bpe_new.yaml" \
    --log_path "./log/1/" \
    --saveto "./save/1/" \
    --use_gpu