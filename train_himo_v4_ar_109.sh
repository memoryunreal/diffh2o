#!/bin/bash
# V4 AR training on server 109
# Uses DataParallel (no NCCL needed, avoids container DDP hang)
#
# Usage:
#   bash train_himo_v4_ar_109.sh 2o    # Train 2-object on GPUs 1,2,3
#   bash train_himo_v4_ar_109.sh 3o    # Train 3-object on GPUs 4,5,6
#   bash train_himo_v4_ar_109.sh both  # Train both (sequentially)

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rog_env
cd /hdd0/lizhe/code/diffh2o

export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe
export PYTHONUNBUFFERED=1

MODE=${1:-both}

run_2o() {
  echo "=== HIMO 2o V4 AR — DataParallel on GPUs 1,2,3 ==="
  echo "  Init: save/himo_2o_v4_ar/model_init.pt"
  echo "  Target: 200k steps"
  echo "  Started: $(date)"

  rm -rf save/himo_2o_v4_ar/args.json save/himo_2o_v4_ar/wandb 2>/dev/null

  CUDA_VISIBLE_DEVICES=1,2,3 python -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_2o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 2 \
    --data_parallel 0,1,2 \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 192 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_2o_v4_ar \
    --resume save/himo_2o_v4_ar/model_init.pt \
    --save_interval 10000 --log_interval 100 \
    --use_wandb --wandb_project himo-v4

  echo "=== HIMO 2o V4 AR complete: $(date) ==="
}

run_3o() {
  echo "=== HIMO 3o V4 AR — DataParallel on GPUs 4,5,6 ==="
  echo "  Init: save/himo_3o_v4_ar/model_init.pt"
  echo "  Target: 200k steps"
  echo "  Started: $(date)"

  rm -rf save/himo_3o_v4_ar/args.json save/himo_3o_v4_ar/wandb 2>/dev/null

  CUDA_VISIBLE_DEVICES=4,5,6 python -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_3o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 3 \
    --data_parallel 0,1,2 \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 192 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_3o_v4_ar \
    --resume save/himo_3o_v4_ar/model_init.pt \
    --save_interval 10000 --log_interval 100 \
    --use_wandb --wandb_project himo-v4

  echo "=== HIMO 3o V4 AR complete: $(date) ==="
}

case $MODE in
  2o)   run_2o ;;
  3o)   run_3o ;;
  both) run_2o && run_3o ;;
  *)    echo "Usage: $0 {2o|3o|both}" && exit 1 ;;
esac
