#!/bin/bash
# V4 AR training on server 109
# Uses DataParallel (no NCCL needed, avoids container DDP hang)
#
# GPU layout (GPUs 1-6):
#   GPUs 1,2 — 2o ff2048
#   GPUs 3,4 — 3o ff2048
#   GPU 5   — 2o ff1024
#   GPU 6   — 3o ff1024
#
# Usage:
#   bash train_himo_v4_ar_109.sh all        # All 4 runs in parallel
#   bash train_himo_v4_ar_109.sh 2o         # 2o ff2048 only
#   bash train_himo_v4_ar_109.sh 3o         # 3o ff2048 only
#   bash train_himo_v4_ar_109.sh 2o_1024    # 2o ff1024 only
#   bash train_himo_v4_ar_109.sh 3o_1024    # 3o ff1024 only

set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate rog_env
cd /hdd0/lizhe/code/diffh2o

export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe
export PYTHONUNBUFFERED=1

MODE=${1:-all}

run_2o() {
  echo "=== HIMO 2o V4 AR ff2048 — DataParallel on GPUs 1,2 ==="
  echo "  Target: 200k steps"
  echo "  Started: $(date)"

  mkdir -p save/himo_2o_v4_ar_ff2048

  CUDA_VISIBLE_DEVICES=1,2 python -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_2o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 2 \
    --ff_dim 2048 \
    --data_parallel 0,1 \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 96 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_2o_v4_ar_ff2048 \
    --save_interval 10000 --log_interval 100 \
    --use_wandb --wandb_project himo-v4

  echo "=== HIMO 2o V4 AR ff2048 complete: $(date) ==="
}

run_3o() {
  echo "=== HIMO 3o V4 AR ff2048 — DataParallel on GPUs 3,4 ==="
  echo "  Target: 200k steps"
  echo "  Started: $(date)"

  mkdir -p save/himo_3o_v4_ar_ff2048

  CUDA_VISIBLE_DEVICES=3,4 python -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_3o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 3 \
    --ff_dim 2048 \
    --data_parallel 0,1 \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 96 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_3o_v4_ar_ff2048 \
    --save_interval 10000 --log_interval 100 \
    --use_wandb --wandb_project himo-v4

  echo "=== HIMO 3o V4 AR ff2048 complete: $(date) ==="
}

run_2o_1024() {
  echo "=== HIMO 2o V4 AR ff1024 — Single GPU 5 ==="
  echo "  Resume: save/himo_2o_v4_ar/model_init.pt"
  echo "  Target: 200k steps"
  echo "  Started: $(date)"

  CUDA_VISIBLE_DEVICES=5 python -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_2o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 2 \
    --ff_dim 1024 \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 64 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_2o_v4_ar \
    --resume save/himo_2o_v4_ar/model_init.pt \
    --save_interval 10000 --log_interval 100 \
    --use_wandb --wandb_project himo-v4

  echo "=== HIMO 2o V4 AR ff1024 complete: $(date) ==="
}

run_3o_1024() {
  echo "=== HIMO 3o V4 AR ff1024 — Single GPU 6 ==="
  echo "  Resume: save/himo_3o_v4_ar/model_init.pt"
  echo "  Target: 200k steps"
  echo "  Started: $(date)"

  CUDA_VISIBLE_DEVICES=6 python -m train.train_flow \
    --dataset_type himo --data_root dataset/HIMO_V4_3o \
    --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
    --obj_dyn_dim 9 --max_objects 3 \
    --ff_dim 1024 \
    --autoregressive --history_len 10 \
    --num_steps 200000 --batch_size 64 --lr 1e-4 \
    --num_frames 300 \
    --save_dir save/himo_3o_v4_ar \
    --resume save/himo_3o_v4_ar/model_init.pt \
    --save_interval 10000 --log_interval 100 \
    --use_wandb --wandb_project himo-v4

  echo "=== HIMO 3o V4 AR ff1024 complete: $(date) ==="
}

case $MODE in
  2o)      run_2o ;;
  3o)      run_3o ;;
  2o_1024) run_2o_1024 ;;
  3o_1024) run_3o_1024 ;;
  all)     run_2o & run_3o & run_2o_1024 & run_3o_1024 & wait ;;
  *)       echo "Usage: $0 {2o|3o|2o_1024|3o_1024|all}" && exit 1 ;;
esac
