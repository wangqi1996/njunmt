#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1

echo "Using GPU $CUDA_VISIBLE_DEVICES..."

python -m src.bin.train \
	--model_name "transformer" \
	--config_path "configs/transformer_wmt14_en2de.yaml" \
	--log_path "./log" \
	--saveto "./save/" \
	--use_gpu	
