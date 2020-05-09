#!/usr/bin/env bash
export num=$1
export SRC="zh"
export TGT="en"
export LAN="${SRC}2${TGT}"
export CON="configs/transformer_nist_zh2en_bpe.yaml"
export MT="mt0${num}"
export SAVETODIR="/home/user_data55/wangdq/code/njunmt/log/${LAN}/"
mkdir -p $SAVETODIR
python -m src.distributed.launch  \
        --nproc_per_node=3 --nnodes=1 --node_rank=0 \
        --master_addr="127.0.0.1" --master_port=12245  \
	    src.bin.ensemble_translate \
        --model_name "transformer" \
        --source_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/${MT}.src" \
        --model_path '/home/user_data55/wangdq/code/njunmt_dist/save/transformer.best.final,/home/user_data55/wangdq/code/njunmt_dist/save/bpe_dropout2/transformer.best.final,/home/user_data55/wangdq/code/njunmt_dist/save/char_enc2/transformer.best.final'  \
        --ref_path "/home/user_data/weihr/NMT_DATA_PY3/NIST-ZH-EN/test/${MT}.ref" \
        --config_path "${CON}" \
        --batch_size 20 \
        --beam_size 5 \
        --saveto "${SAVETODIR}/output${MT}" \
        --use_gpu  \
        --num_refs 4 \
        --multi_gpu
