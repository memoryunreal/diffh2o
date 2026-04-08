# Model Architecture & I/O Specification

## 1. Generation Model (Single-Motion)

**Checkpoint**: `save/oakink2_full/model000400015.pt`
**Architecture**: UNet with AdaGN (Adaptive Group Normalization)

### Input

| Component | Shape | Description |
|-----------|-------|-------------|
| Noisy motion | `[B, 398, 1, T]` | Normalized 398D OakInk2 features, T=200 frames |
| Diffusion timestep | `[B]` | Integer timestep in [0, 1000) |
| Text embedding (CLIP) | `[B, 512]` | Frozen CLIP ViT-B/32 text encoder output |
| Object BPS encoding | `[B, 1024, 3]` | Basis Point Set encoding of target object geometry |
| Conditioning mask | `[B]` | Binary mask for classifier-free guidance (10% dropout) |

### Output

| Component | Shape | Description |
|-----------|-------|-------------|
| Predicted x₀ | `[B, 398, 1, T]` | Denoised motion prediction (same shape as input) |

### 398D Feature Vector Layout

```
Body root (9D):           world_tsl (3D) + world_rot (6D)           [0:9]
Body pose (126D):         21 body joints × 6D rotation              [9:135]
Left hand PCA (30D):      pos (3D) + orient (6D) + pca (21D)       [135:165]
Right hand PCA (30D):     pos (3D) + orient (6D) + pca (21D)       [165:195]
Left hand quat (67D):     tsl (3D) + 16 joints × 4D quaternion     [195:262]
Right hand quat (67D):    tsl (3D) + 16 joints × 4D quaternion     [262:329]
SDF left (21D):           signed distance per joint (placeholder)   [329:350]
SDF right (21D):          signed distance per joint (placeholder)   [350:371]
Object 1 pose (9D):       position (3D) + rotation (6D)            [371:380]
Object 2 pose (9D):       position (3D) + rotation (6D)            [380:389]
Object 3 pose (9D):       position (3D) + rotation (6D)            [389:398]
─────────────────────────────────────────────────────────────────────
Total: 398D
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | oakink2_primitive (2272 train / 568 test) |
| Training steps | 400,000 |
| Batch size | 64 |
| Max frames (T) | 224 (padded from 200) |
| Learning rate | 1e-4 |
| Noise schedule | cosine |
| Diffusion steps | 1000 |
| Predict target | x₀ (start) |
| Latent dim | 512 |
| UNet channels | [512, 1024, 1024, 1024, 1024] (dim_mults=[2,2,2,2]) |
| Layers | 8 |
| FF size | 1024 |
| Guidance scale (inference) | 2.5 |
| FP16 | Yes |
| Optimizer | AdamW (beta2=0.999, weight_decay=0.01) |
| Grad clip | 1.0 |

### Normalization

- Mean: `dataset/OAKINK2/Mean_oakink2_primitive.npy` (398,)
- Std: `dataset/OAKINK2/Std_oakink2_primitive.npy` (398,)
- Normalize: `x_norm = (x - mean) / std`
- Denormalize: `x = x_norm * std + mean`

### Key Files

- Model class: `model/mdm.py` (MDM UNet with AdaGN)
- Config: `configs/card.py` → `oakink2_full`
- Training: `train/train_oakink2.py`
- Generation: `sample/generate_oakink2.py`
- Dataset: `data_loaders/humanml/data/oakink2_dataset.py`

---

## 2. Transition Model (Switch Scheduler)

**Checkpoint**: `save/oakink2_transition/model000200000.pt`
**Architecture**: Transformer Encoder (8 layers)

### Input

| Component | Shape | Description |
|-----------|-------|-------------|
| Noisy transition | `[B, 398, 1, 60]` | Normalized 398D features, 60 transition frames |
| Diffusion timestep | `[B]` | Integer timestep in [0, 1000) |
| Boundary before | `[B, 10, 398]` | Last 10 frames of preceding segment (normalized) |
| Boundary after | `[B, 10, 398]` | First 10 frames of following segment (normalized) |
| Text source embedding | `[B, 22, 300]` | GloVe word embeddings of source sentence (22 max tokens) |
| Text target embedding | `[B, 22, 300]` | GloVe word embeddings of target sentence |
| Source object BPS | `[B, 1024, 3]` | BPS encoding of source object |
| Target object BPS | `[B, 1024, 3]` | BPS encoding of target object |

### Output

| Component | Shape | Description |
|-----------|-------|-------------|
| Predicted x₀ | `[B, 398, 1, 60]` | Denoised transition (same shape as input) |
| Blend weights | `[B, 1, 60, 1]` | Sigmoid-ramp blending weights for boundary smoothing |

### Internal Processing Flow

```
Input [B, 398, 1, 60]
  → permute to [B, 60, 398]
  → input_projection: Linear(398, 512)     → [B, 60, 512]
  + timestep_embedding: sinusoidal(t)       → [B, 512] broadcast
  + boundary_embedding: MLP([10*398*2])     → [B, 512] broadcast
  + text_embedding: MLP([300*2])            → [B, 512] broadcast
  + obj_embedding: MLP([3072*2])            → [B, 512] broadcast
  → positional_encoding                     → [B, 60, 512]
  → 8× TransformerEncoderLayer(d=512, h=4, ff=1024, dropout=0.1, GELU)
  → output_projection: Linear(512, 398)    → [B, 60, 398]
  → permute to [B, 398, 1, 60]
```

### Conditioning Details

**Boundary Encoder**:
- Input: concatenation of last 10 frames of segment A + first 10 frames of segment B
- Flattened: `[B, 10*398 + 10*398]` = `[B, 7960]`
- MLP: `7960 → 1024 → 512` with GELU and dropout

**Text Encoder**:
- Source + target GloVe embeddings pooled to sentence level (mean over tokens)
- Concatenated: `[B, 300 + 300]` = `[B, 600]`
- MLP: `600 → 512 → 512` with GELU

**Object BPS Encoder**:
- Source + target BPS flattened and concatenated: `[B, 3072 + 3072]` = `[B, 6144]`
- MLP: `6144 → 512 → 512` with GELU

**Classifier-Free Guidance**: 10% dropout on boundary, text, and object conditions during training.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Dataset | oakink2_transitions (1776 train / 445 test) |
| Training steps | 200,000 |
| Batch size | 4 |
| Transition frames | 60 |
| Boundary frames | 10 |
| Learning rate | 1e-4 |
| Noise schedule | cosine |
| Diffusion steps | 1000 |
| Predict target | x₀ (start) |
| Latent dim | 512 |
| Transformer layers | 8 |
| Attention heads | 4 |
| FF size | 1024 |
| Dropout | 0.1 |
| FP16 | Yes |

### Normalization

- Mean: `dataset/OAKINK2/Mean_transitions.npy` (398,)
- Std: `dataset/OAKINK2/Std_transitions.npy` (398,)

### Key Files

- Model class: `model/switch_scheduler.py` → `SwitchScheduler`
- Config: `configs/card.py` → `oakink2_transition`
- Training: `train/train_transition.py`
- Generation: `sample/generate_transition.py`
- Dataset: `data_loaders/humanml/data/transition_dataset.py`

---

## 3. End-to-End Pipeline

**Script**: `generate_long_motion_cli.py`

### Flow

```
Multi-sentence text prompt
  → TextOrchestrator: split sentences, extract objects, plan durations
  → For each sentence:
      → SegmentGenerator: DiffH2O backbone (generation model above)
      → Output: [T_i, 398] motion segment
  → For each adjacent pair (seg_i, seg_{i+1}):
      → TransitionGenerator: SwitchScheduler (transition model above)
      → Input: last 10 frames of seg_i + first 10 frames of seg_{i+1}
      → Output: [60, 398] transition
  → Concatenate: seg_1 + trans_1→2 + seg_2 + trans_2→3 + ... + seg_N
  → Post-process: velocity smoothing at boundaries
  → Output: full long-horizon motion sequence
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| Text Orchestrator | `sample/text_orchestration.py` | Parse multi-sentence prompts |
| Segment Generator | `sample/segment_generator.py` | Per-sentence DiffH2O generation |
| Transition Generator | `sample/transition_generator.py` | Bridge segments with diffusion |
| Motion Structures | `utils/motion_data_structures.py` | MotionSegment, TransitionSegment, MotionSequence |
| CLI | `generate_long_motion_cli.py` | End-to-end command-line interface |
