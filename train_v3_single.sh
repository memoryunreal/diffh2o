#!/bin/bash
# Single-GPU training for CRR-Flow V3
# Batch size: 32, target 200K steps
#
# Usage: bash train_v3_single.sh

cd /hdd0/lizhe/code/diffh2o

export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe
export PYTHONUNBUFFERED=1

python -m train.train_flow \
  --model_version v3 \
  --device 0 \
  --num_steps 200000 \
  --batch_size 32 \
  --lr 1e-4 \
  --latent_dim 512 \
  --num_layers 8 \
  --num_heads 8 \
  --ff_dim 2048 \
  --num_frames 200 \
  --data_root dataset/OAKINK2_V3 \
  --save_dir save/v3_entity_200k_fixed \
  --save_interval 50000 \
  --log_interval 500 \
  --use_wandb \
  --wandb_project crr-flow
