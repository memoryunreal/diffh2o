# CRR-Flow Marker Model Architecture (without object stream)

## Overview

FlowDiT is a single-backbone DiT (Diffusion Transformer) that generates 77 SMPL-X body marker positions via flow matching. It uses state-aware conditioning (per-frame entity state labels) and CLIP text conditioning through AdaLN-Zero modulation.

**Model class:** `FlowDiT` in `model/flow_dit.py`
**Training script:** `train/train_flow.py` (flow matching mode, no `--use_diffusion`)
**Dataset:** `dataset/OAKINK2_MARKERS/` (77 markers + 3 objects = 258D, 400 frames)

## Input Dimensions

### Motion Input: `x_t` [B, T, 258]

| Range | Feature | Dim | Description |
|-------|---------|-----|-------------|
| 0-231 | Marker Positions | 77 × 3 | 77 SMPL-X surface vertex positions (x,y,z) |
| 231-234 | Object 1 Pos | 3 | Object 1 translation |
| 234-240 | Object 1 Rot | 6 | Object 1 rotation (6D continuous repr) |
| 240-243 | Object 2 Pos | 3 | Object 2 translation |
| 243-249 | Object 2 Rot | 6 | Object 2 rotation (6D continuous repr) |
| 249-252 | Object 3 Pos | 3 | Object 3 translation |
| 252-258 | Object 3 Rot | 6 | Object 3 rotation (6D continuous repr) |
| **Total** | | **258** | |

**Note:** Unlike InterAct's HOIDiff (962D), this model has:
- No marker velocities (231D saved)
- No feet contact features (14D saved)
- No per-marker relative object representation (468D saved)
- No dual-backbone / cross-attention (simpler architecture)
- Objects are concatenated directly with markers, not processed in a separate stream

### 77 SMPL-X Markers

The 77 markers are **67 SSM body markers + 10 fingertips**, selected from SMPL-X's 10,475 vertices using indices from `/hhd4/lizhe/code/InterAct/process/markerset.py`:

```python
markerset_smplx = [5920, 5621, 5882, 3486, 4430, 3258, 2029, 5694, 5645, 9117, 4302,
                   4319, 707, 4788, 4198, 3998, 8846, 4726, 3682, 3688, 5890, 5901,
                   3961, 4722, 5698, 5868, 3892, 3712, 4479, 4088, 3477, 4839, 5787,
                   4291, 8576, 6248, 7166, 6021, 3067, 8250, 8388, 8339, 3116, 7040,
                   7099, 2198, 7524, 7115, 6265, 6746, 8634, 7462, 6443, 6449, 8584,
                   8595, 6709, 7458, 8392, 8527, 7215, 6832, 7575, 6503, 8481, 5945,
                   5947, 8079, 7669, 7794, 7905, 8022, 5361, 4933, 5058, 5169, 5286]
```

| Body Part | Marker Count | Indices |
|-----------|-------------|---------|
| Head | 6 | Forehead, temples, back of head |
| Torso/Spine | 19 | Shoulders, chest, spine, hips, pelvis |
| Left hand/wrist | 9 | Wrist, palm, knuckles |
| Right hand/wrist | 8 | Wrist, palm, knuckles |
| Left foot/ankle | 7 | Ankle, heel, sole |
| Right foot/ankle | 7 | Ankle, heel, sole |
| Left toes | 5 | Toe tips |
| Right toes | 6 | Toe tips |
| **SSM subtotal** | **67** | |
| Left fingertips | 5 | Thumb through pinky tips |
| Right fingertips | 5 | Thumb through pinky tips |
| **Total** | **77** | |

### Conditioning Inputs: `y` (dict)

| Key | Shape | Source | Description |
|-----|-------|--------|-------------|
| `text` | list of B strings | `texts/*.txt` | Natural language descriptions, CLIP-encoded inside model |
| `state_labels` | [B, T, 20] | `state_arrays/*.npy` | Per-frame per-entity state labels (7 states, padded to 20 entities) |
| `lengths` | [B] | Dataset | Actual sequence length |

**State label vocabulary:**

| ID | State | Applies to |
|----|-------|-----------|
| 0 | `[Idle]` | Hands (no annotation) |
| 1 | `[Reach]` | Hands (approaching target) |
| 2 | `[Grasped]` | Hands/Objects (rigid attachment) |
| 3 | `[Grasped_and_Interacting]` | Objects (grasped + tool acting) |
| 4 | `[Release]` | Hands (releasing target) |
| 5 | `[Static]` | Objects (at rest) |
| 6 | `[Interacting]` | Objects (being acted upon) |
| -1 | (padding) | Unused entity slots |

## Model Architecture

### Hyperparameters

| Parameter | Value | Source |
|-----------|-------|--------|
| `input_dim` | 258 | Auto-detected from dataset |
| `latent_dim` | 512 | `--latent_dim` |
| `ff_dim` | 1024 | `--ff_dim` |
| `num_layers` | 8 | `--num_layers` |
| `num_heads` | 8 | `--num_heads` |
| `dropout` | 0.1 | `--dropout` |
| `max_seq_len` | 400 | `--num_frames` |
| `cond_mask_prob` | 0.1 | Classifier-free guidance dropout |
| **Trainable params** | **~20M** | Excluding frozen CLIP |
| **CLIP params** | **151M** | Frozen ViT-B/32 |

### Architecture Diagram

```
Input: x_t [B, T=400, D=258] (noisy motion at flow time t)

  → Linear(258, 512)                           # input_proj
  + PositionalEncoding1D(512, max_len=400)     # 1D sinusoidal
  + StateEmbedding(7+1, 512)                   # per-frame entity states
      nn.Embedding with padding_idx=7
      Sum over active entities (state >= 0)
      Added to token sequence: h = h + state_emb

  Condition vector:
      t_emb = SinusoidalTimestepEmbedder(512)  # continuous t∈[0,1]
      text_emb = CLIP_ViT-B/32(text) → Linear(512, 512)
      text_emb = mask_cond(text_emb, p=0.1)    # CFG dropout
      c = MLP(cat(t_emb, text_emb))            # [B, 512]
          Linear(1024, 512) → SiLU → Linear(512, 512)

  → 8 × DiTBlock(d=512, heads=8, ff=1024)     # AdaLN-Zero transformer
      Each block:
          AdaLN modulation: c → SiLU → Linear(512, 3072) → chunk(6)
              → shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
          Pre-norm self-attention:
              x_norm = (LayerNorm(x) * (1+scale_msa) + shift_msa)
              x = x + gate_msa * MultiheadAttention(x_norm, x_norm, x_norm)
          Pre-norm FFN:
              x_norm = (LayerNorm(x) * (1+scale_mlp) + shift_mlp)
              x = x + gate_mlp * MLP(x_norm)
                  Linear(512, 1024) → GELU → Dropout → Linear(1024, 512)

  → FinalLayer(512, 258)                       # AdaLN-Zero output
      c → SiLU → Linear(512, 1024) → chunk(2) → shift, scale
      x = (LayerNorm(x) * (1+scale) + shift)
      x = Linear(512, 258)
      (Initialized to zeros for stable training start)

Output: v_t [B, T=400, D=258] (predicted velocity field)
```

### Key Differences from InterAct's HOIDiff

| Aspect | HOIDiff (InterAct) | FlowDiT (This) |
|--------|-------------------|-----------------|
| Architecture | Dual-backbone transformer + cross-attention | Single-backbone DiT |
| Input dim | 962D (markers+vel+feet+obj+contact) | 258D (markers+obj only) |
| Generation | DDPM diffusion (predict x₀) | Flow matching (predict velocity) |
| Timestep | Discrete [0, 1000] | Continuous [0, 1] |
| Conditioning | Timestep + BPS + CLIP text | Timestep + CLIP text + state labels |
| Object stream | Separate backbone + cross-attention | Concatenated with markers |
| Modulation | Standard TransformerEncoder | AdaLN-Zero (shift/scale/gate) |
| Sequence length | Variable (max ~300) | Fixed 400 frames |
| Trainable params | ~30M | ~20M |

## Training

### Flow Matching Loss

```python
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)

x0 = randn_like(x1)                                    # noise
t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)  # OT interpolation
vt_pred = model(xt, t, y=y)                             # predict velocity
loss = MSE(vt_pred, ut)                                 # velocity MSE
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=0.01) |
| Gradient clipping | 1.0 |
| Batch size | 4 |
| Training sequences | 20 (from 501 train split) |
| Total steps | 200K |
| Checkpoint interval | 50K |
| WandB project | `crr-flow` |

### Training Command

```bash
conda run -n rog_env python -m train.train_flow \
  --device 3 --num_steps 200000 --batch_size 4 \
  --latent_dim 512 --num_layers 8 --num_heads 8 --ff_dim 1024 \
  --num_frames 400 --split train_20 \
  --data_root dataset/OAKINK2_MARKERS \
  --save_dir save/markers_fm_200k \
  --save_interval 50000 \
  --use_wandb --wandb_project crr-flow
```

## Inference (ODE Integration)

```python
from torchdiffeq import odeint

x0 = randn(1, 400, 258)                        # initial noise
def ode_fn(t, x):
    return model(x, t.expand(1), y=y)
t_span = linspace(0, 1, 11)                     # 10 Euler steps
traj = odeint(ode_fn, x0, t_span, method='euler')
generated = traj[-1]                             # [1, 400, 258]
```

## Visualization Pipeline

### Path A: Direct Marker Rendering (quick)
77 markers rendered as colored spheres (blue=body, red=fingertips) using aitviewer.

### Path B: SMPL-X Inverse Fitting → Full Mesh (production)

Adapted from InterAct's `marker2smpl.py` for SMPL-X (10,475 vertices):

```
Generated markers [T, 77, 3]
    ↓
Optional: Savitzky-Golay smoothing (light w=5 or heavy w=21)
    ↓
LBFGS optimization (SMPL-X parameters):
    Stage 1: translation + global_orient (5 iterations)
    Stage 2: all params jointly (20 iterations)
    Loss = 100×marker_fit + 5×smoothness + 5×beta_reg
    ↓
SMPL-X forward pass → vertices [T, 10475, 3]
    ↓
Render mesh + objects → MP4 video
```

### Post-Processing (Mandatory per CLAUDE.md)

Three versions for every visualization:
1. **Raw** — no post-processing
2. **Medium** — Savitzky-Golay (window=11, polyorder=3)
3. **Heavy** — Savitzky-Golay (window=21, polyorder=3)

## Experiment Results (100K checkpoint)

| Metric | Value |
|--------|-------|
| MSE (markers) | 0.0066 |
| Gen marker Y range | [-1.15, 0.41] |
| GT marker Y range | [-1.15, 0.39] |
| Visualization | `output/markers_100k_vis/gt_mesh.mp4`, `gen_raw_mesh.mp4`, `gen_light_mesh.mp4`, `gen_heavy_mesh.mp4` |

## Data Preprocessing

**Script:** `prepare/preprocess_oakink2_flow.py --feature_mode markers`

**Pipeline:**
```
Raw OakInk2 annotation pickle
    ↓
SMPL-X forward pass (with 300 betas, body_shape from annotation)
    → vertices [T, 10475, 3]
    ↓
Select 77 markers: verts[:, markerset_smplx]
    → markers [T, 77, 3]
    ↓
Flatten: markers.reshape(T, 231)
    ↓
Concatenate object transforms: pos(3D) + rot(6D) × 3 objects = 27D
    ↓
Uniform subsample to 400 frames
    ↓
Save: [400, 258] float32
```

**Dataset:** `dataset/OAKINK2_MARKERS/` — 627 sequences, 501 train / 126 test

## Files

| File | Purpose |
|------|---------|
| `model/flow_dit.py` | FlowDiT model class |
| `train/train_flow.py` | Training script (flow matching + diffusion modes) |
| `sample/generate_flow.py` | Generation via ODE integration |
| `data_loaders/humanml/data/flow_dataset.py` | Dataset loader |
| `prepare/preprocess_oakink2_flow.py` | Preprocessing (markers mode) |
| `/hhd4/lizhe/code/InterAct/process/markerset.py` | 77 marker vertex indices |
