#!/bin/bash
# 4-GPU DDP AR training for V4 on HIMO (for server 109)
# Effective batch size: 4 GPUs x 64 = 256
#
# Usage:
#   bash train_himo_v4_ar_ddp.sh 2o    # Train 2-object
#   bash train_himo_v4_ar_ddp.sh 3o    # Train 3-object
#   bash train_himo_v4_ar_ddp.sh both  # Train 2o then 3o sequentially

set -e
cd /hdd0/lizhe/code/diffh2o

export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe
export PYTHONUNBUFFERED=1
export NCCL_DEBUG=WARN

MODE=${1:-both}

run_2o() {
  echo "=== Starting HIMO 2o V4 AR DDP training (4 GPUs) ==="
  echo "  Init: save/himo_2o_v4_ar/model_init.pt (200k non-AR)"
  echo "  Target: 200k AR steps"
  torchrun --nproc_per_node=4 --master_port=29500 \
    -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_2o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 2 \
    --multi_gpu \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 64 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_2o_v4_ar \
    --resume save/himo_2o_v4_ar/model_init.pt \
    --save_interval 10000 --log_interval 500 \
    --use_wandb --wandb_project himo-v4
  echo "=== HIMO 2o V4 AR complete ==="
}

run_3o() {
  echo "=== Starting HIMO 3o V4 AR DDP training (4 GPUs) ==="
  echo "  Init: save/himo_3o_v4_ar/model_init.pt (150k non-AR)"
  echo "  Target: 200k AR steps"
  torchrun --nproc_per_node=4 --master_port=29500 \
    -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_3o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 3 \
    --multi_gpu \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 64 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_3o_v4_ar \
    --resume save/himo_3o_v4_ar/model_init.pt \
    --save_interval 10000 --log_interval 500 \
    --use_wandb --wandb_project himo-v4
  echo "=== HIMO 3o V4 AR complete ==="
}

case $MODE in
  2o)   run_2o ;;
  3o)   run_3o ;;
  both) run_2o && run_3o ;;
  *)    echo "Usage: $0 {2o|3o|both}" && exit 1 ;;
esac
