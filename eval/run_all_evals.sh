#!/bin/bash
# Run evaluation for all trained model variants
# Uses GPU 5 (33GB free) for generation
# Results saved to eval_results/

source /root/miniconda3/bin/activate rog_env
export PYTHONUNBUFFERED=1
GPU=5
N=50  # samples per eval (0=all, use 50 for quick eval)
mkdir -p eval_results

echo "========================================"
echo "CRR-Flow Evaluation Suite"
echo "GPU: $GPU, Samples: $N"
echo "========================================"

# --- OakInk2 Models ---

echo ""
echo "=== V3 @150K on OakInk2 ==="
python -m eval.eval_crr_flow \
  --model_path save/v3_entity_200k_fixed/model000150000.pt \
  --model_version v3 --data_root dataset/OAKINK2_V3 \
  --num_samples $N --device $GPU --output_dir eval_results

echo ""
echo "=== V4 @130K on OakInk2 ==="
python -m eval.eval_crr_flow \
  --model_path save/v4_entity_200k/model000130000.pt \
  --model_version v4 --data_root dataset/OAKINK2_V4 \
  --num_samples $N --device $GPU --output_dir eval_results

echo ""
echo "=== V4 Autoreg @120K on OakInk2 ==="
python -m eval.eval_crr_flow \
  --model_path save/v4_autoreg_200k/model000120000.pt \
  --model_version v4 --data_root dataset/OAKINK2_V4 \
  --num_samples $N --device $GPU --output_dir eval_results

echo ""
echo "=== V4 AR+Release @60K on OakInk2 ==="
python -m eval.eval_crr_flow \
  --model_path save/v4_autoreg_release_200k/model000060000.pt \
  --model_version v4 --data_root dataset/OAKINK2_V4 \
  --state_dir dataset/OAKINK2_V4/state_arrays_fixed \
  --num_samples $N --device $GPU --output_dir eval_results

# --- GRAB Model ---

echo ""
echo "=== V4 GRAB @40K ==="
python -m eval.eval_crr_flow \
  --model_path save/v4_grab_200k/model000040000.pt \
  --model_version v4 --data_root dataset/GRAB_V4 \
  --num_samples $N --device $GPU --output_dir eval_results

# --- InterAct GRAB Model ---

echo ""
echo "=== V4 InterAct GRAB @20K ==="
python -m eval.eval_crr_flow \
  --model_path save/v4_interact_grab_200k/model000020000.pt \
  --model_version v4 --data_root dataset/INTERACT_GRAB_V4 \
  --num_samples $N --device $GPU --output_dir eval_results

# --- OMOMO Model ---

echo ""
echo "=== V4 OMOMO @40K ==="
python -m eval.eval_crr_flow \
  --model_path save/v4_omomo_200k/model000040000.pt \
  --model_version v4 --data_root dataset/INTERACT_OMOMO_V4 \
  --num_samples $N --device $GPU --output_dir eval_results

echo ""
echo "========================================"
echo "All evaluations complete!"
echo "Results in eval_results/"
echo "========================================"
