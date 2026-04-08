#!/bin/bash
# 4-GPU DDP training for CRR-Flow V3 (entity-centric DiT)
# Train from step 0, target 200K steps
# Effective batch size: 4 GPUs × 64 = 256
#
# Usage: bash train_flow_ddp.sh

source /root/miniconda3/bin/activate rog_env

# WandB
export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe

torchrun --nproc_per_node=4 --master_port=29500 \
  -m train.train_flow \
  --model_version v3 \
  --multi_gpu \
  --num_steps 200000 \
  --batch_size 64 \
  --lr 1e-4 \
  --latent_dim 512 \
  --num_layers 8 \
  --num_heads 8 \
  --ff_dim 2048 \
  --num_frames 200 \
  --data_root dataset/OAKINK2_V3 \
  --save_dir save/v3_entity_ddp_200k \
  --save_interval 50000 \
  --log_interval 500 \
  --use_wandb \
  --wandb_project crr-flow
