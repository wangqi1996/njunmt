#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="uy"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/uy-zh/output/"
export CON="/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final"
export MODEL="/home/user_data55/wangdq/save/uy2zh/final/tune"
export Post="transformer.best.final"
mkdir -p $SAVETODIR

export csample1=$CON/transformer_uy2zh_sample_big.yaml
export csample2=$CON/transformer_uy2zh_sample2_big.yaml
export crelative1=$CON/sample_big_relative.yaml
export crelative2=$CON/sample2_big_relative.yaml
export cave=$CON/sample_big_ave.yaml
export cdynamic_cnn=$CON/sample1_dynamic_cnn.yaml

export sample1=$MODEL/sample1_tune_train/transformer.best.final
export sample2=$MODEL/sample2_tune_train/transformer.best.final
export relative1=$MODEL/relative1_tune_train/transformer_relative.best.final
export relative2=$MODEL/relative2_tune_train/transformer.best.final
export ave=$MODEL/ave_tune_train/transformer.best.final
export dynamic_cnn=$MODEL/dynamic_cnn_tune_train/dynamicCnn.best.final

python -m src.bin.ensemble_translate \
  --model_name "transformer" \
  --source_path "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.uy.token" \
  --config_path "$csample1,$csample2,$crelative1,$cave,$cdynamic_cnn" \
  --model_path "$sample1,$sample2,$relative1,$ave,$dynamic_cnn" \
  --batch_size 20 \
  --beam_size 5 \
  --alpha 1.5 \
  --use_gpu \
  --saveto "${SAVETODIR}/test.out" \
  --ref_path "/home/user_data55/wangdq/data/ccmt/uy-zh/dev2020/dev.zh" \
  --num_refs 1

#

#/home/user_data55/wangdq/save/uy2zh/final/tune
#
#sample1: sample1_tune_train/transformer.best.final   46.96
#sample2: sample2_tune_train/transformer.best.final   46.76
#relative1:
#relative2: relative2_tune_train/transformer.best.final 47.43
#ave: ave_tune_train/transformer.best.final   47.26
#dynamic_cnn: dynamic_cnn_tune_train/transformer.best.final 46.82
#
#
#config_path: /home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/final
#
#sample1:   transformer_uy2zh_sample_big.yaml
#sample2:  transformer_uy2zh_sample2_big.yaml
#relative1:  sample_big_relative.yaml
#relative2:  sample2_big_relative.yaml
#ave:    sample_big_ave.yaml
#dynamic_cnn:  sample1_dynamic_cnn.yaml
