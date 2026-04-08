#!/bin/bash
# 4-GPU DDP training for CRR-Flow V4 (velocity + BPS)
# Effective batch size: 4 GPUs × 64 = 256, target 200K steps
#
# Usage: bash train_v4_ddp.sh

cd /hdd0/lizhe/code/diffh2o

export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe
export PYTHONUNBUFFERED=1

# NCCL config (109 has working P2P)
export NCCL_DEBUG=WARN

torchrun --nproc_per_node=4 --master_port=29500 \
  -m train.train_flow \
  --model_version v4 \
  --multi_gpu \
  --num_steps 200000 \
  --batch_size 64 \
  --lr 1e-4 \
  --latent_dim 512 \
  --num_layers 8 \
  --num_heads 8 \
  --ff_dim 2048 \
  --num_frames 200 \
  --data_root dataset/OAKINK2_V4 \
  --save_dir save/v4_entity_ddp_200k \
  --save_interval 5000 \
  --log_interval 500 \
  --use_wandb \
  --wandb_project crr-flow
