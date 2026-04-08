#!/bin/bash
# Single-GPU AR training for V4 on HIMO 2-object
# Fine-tunes from 200k non-AR checkpoint with autoregressive conditioning
#
# NOTE: We DON'T use --resume because it would set start_step=200000 and immediately stop.
# Instead, we manually load weights by copying the checkpoint as model000000000.pt in the new save_dir.
# The optimizer starts fresh (good for switching to AR training strategy).

cd /hhd4/lizhe/code/diffh2o

export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe
export PYTHONUNBUFFERED=1

# Prepare: copy pretrained weights as starting checkpoint
mkdir -p save/himo_2o_v4_ar
cp save/himo_2o_v4/model000200000.pt save/himo_2o_v4_ar/pretrained_init.pt

nohup conda run -n rog_env python -m train.train_flow \
  --dataset_type himo --data_root dataset/HIMO_V4_2o \
  --model_version v4 --body_dim 201 --lhand_dim 135 --rhand_dim 135 \
  --obj_dyn_dim 9 --max_objects 2 --num_frames 300 \
  --autoregressive --history_len 10 \
  --batch_size 64 --num_steps 200000 --save_interval 50000 --log_interval 100 \
  --save_dir save/himo_2o_v4_ar \
  --resume save/himo_2o_v4_ar/pretrained_init.pt \
  --use_wandb --wandb_project himo-v4 \
  --device 3 > /tmp/himo_2o_ar_train.log 2>&1 &

echo "HIMO 2o V4 AR training PID: $!"
echo "Log: /tmp/himo_2o_ar_train.log"
