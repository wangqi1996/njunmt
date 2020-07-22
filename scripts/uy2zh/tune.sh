python -m src.distributed.launch \
  --nproc_per_node=4 --nnodes=1 --node_rank=0 \
  --master_addr=127.0.0.1 --master_port=15255 \
  src.bin.train \
  --model_name "transformer_relative" \
  --use_gpu \
  --config_path "configs/uy2zh/final/tune.yaml" \
  --pretrain_path "/home/user_data55/liuzh/ccmt_model/uy2zh/sample_big_relative/transformer_relative.best.final" \
  --saveto "/home/wangdq/save/uy2zh/tune" \
  --log_path "./uy2zh/tune/" \
  --valid_path "/home/wangdq/valid/uy2zh/tune"  \
  --shared_dir /tmp/tune
