#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=$1
export SRC="uy"
export TGT="zh"
export LAN="${SRC}2${TGT}"
export CON="/home/user_data55/wangdq/code/njunmt_dist/configs/uy2zh/transformer_uy2zh_sample_big.yaml"
#export CON="configs/uy2zh/transformer_uy2zh_sample_big.yaml"
export SAVETODIR="/home/user_data55/wangdq/data/ccmt/uy-zh/output/"
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
#    --model_path "/home/jiangqn/save/uy2zh/sample_big_tag/transformer.best.final" \

python -m src.bin.translate \
    --model_name "transformer" \
    --source_path "/home/user_data55/wangdq/data/ccmt/uy-zh/test/test.txt" \
    --model_path "/home/jiangqn/save/uy2zh/uy2zh_sample_big/transformer.best.final"  \
    --config_path "${CON}" \
    --batch_size 20 \
    --beam_size 5 \
    --alpha 1.5 \
    --use_gpu  \
    --saveto "${SAVETODIR}/relative.out" \
    --num_refs 1  \
#    --sample_search_k 2
