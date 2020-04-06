#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
	--model_name "transformer" \
	--config_path "configs/transformer_nist_zh2en_bpe_ema.yaml" \
	--log_path "/home/user_data55/wangdq/model/nist_zh2en_ema/log"\
	--saveto "/home/user_data55/wangdq/model/nist_zh2en_ema/save/" \
	--use_gpu	
