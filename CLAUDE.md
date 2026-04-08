# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DiffH2O is a diffusion-based synthesis system for generating hand-object interactions from textual descriptions. The codebase implements a two-stage generation pipeline with separate models for grasping and interaction phases.

## Environment Setup Commands

```bash
# Create and activate conda environment
conda config --append channels conda-forge
conda env create -f environment_diffh2o.yml
conda activate diffh2o
pip install -r requirements.txt
conda remove --force ffmpeg
pip install git+https://github.com/openai/CLIP.git

# Download data and pretrained models
bash prepare/download_representations.sh
bash prepare/download_pretrained_models.sh
```

## Common Development Commands

### Training Models

```bash
# Train grasp model
python -m train.train_grasp

# Train full model with simple text descriptions
python -m train.train_diffh2o

# Train full model with detailed text descriptions  
python -m train.train_diffh2o_detailed

# Train interaction-only model (for comparison with IMoS)
python -m train.train_interaction

# Optional parameters for all training commands:
# --device <gpu_id>
# --train_platform_type {ClearmlPlatform, TensorboardPlatform}
```

### Generating Samples

```bash
# Single-stage generation with simple annotations
python -m sample.generate --model_path ./save/diffh2o_full/model000200000.pt --num_samples 16

# Single-stage generation with detailed annotations
python -m sample.generate --model_path ./save/diffh2o_full_detailed/model000200000.pt --num_samples 16 --text_detailed

# Two-stage generation with guidance (simple annotations)
python -m sample.generate_2stage --model_path ./save/diffh2o_full/model000200000.pt --num_samples 16 --guidance

# Two-stage generation with guidance (detailed annotations)  
python -m sample.generate_2stage --model_path ./save/diffh2o_full_detailed/model000200000.pt --num_samples 16 --guidance --text_detailed

# Long-term generation from multi-sentence text
python generate_long_motion_cli.py \
  --prompt "The person picks up the apple, examines it, then places it down and waves." \
  --model_path ./save/diffh2o_full/model000200000.pt \
  --output output/long_motion
```

### Visualization

```bash
# Visualize generated sequences
python visualize/visualize_sequences.py --file_path save/diffh2o_full/samples_000400000/ --is_pca

# Save visualization as video (headless)
xvfb-run -s "-screen 0 1024x768x24" python visualize/visualize_sequences.py --file_path <path> --is_pca --save_video --resolution low
```

## Code Architecture

### Core Components

1. **Model Architecture** (`model/`)
   - `mdm.py`, `mdm_unet.py`, `mdm_dit.py`: Different diffusion model architectures
   - `cfg_sampler.py`: Classifier-free sampling implementation

2. **Training Pipeline** (`train/`)
   - `training_loop.py`: Core training loop implementation
   - `train_diffh2o.py`, `train_diffh2o_detailed.py`: Full model training scripts
   - `train_grasp.py`: Grasp-only model training
   - `train_interaction.py`: Interaction-only model training

3. **Data Processing** (`data_loaders/`)
   - `get_data.py`: Main data loading interface
   - `humanml/`: HumanML data processing utilities
   - Motion representation handling for GRAB dataset

4. **Diffusion Framework** (`diffusion/`)
   - `gaussian_diffusion.py`: Core diffusion process implementation
   - `respace.py`: Timestep respacing utilities
   - `losses.py`: Loss computation

5. **Generation Pipeline** (`sample/`)
   - `generate.py`: Single-stage generation
   - `generate_2stage.py`: Two-stage generation with inpainting
   - `condition_hands.py`: Hand conditioning utilities

6. **Configuration System** (`configs/`)
   - `card.py`: Model configuration presets (diffh2o_grasp, diffh2o_full, etc.)
   - `data.py`: Dataset configuration
   - `model.py`: Model architecture configurations

7. **Evaluation** (`eval/`)
   - `eval_grab.py`: GRAB dataset evaluation
   - `metrics/`: Physics and statistical metrics

## Key Implementation Details

- The system uses PCA representation for MANO hand model by default
- Motion sequences are represented in a special format combining hand poses and object transformations
- Two-stage generation involves:
  1. Grasp phase generation (hand approaching object)
  2. Interaction phase generation with inpainting of grasp
- Supports both simple and detailed text annotations for training/generation

## Dataset Structure

The processed GRAB dataset should be organized as:
```
dataset/GRAB_HANDS/
├── diffh2o_representation_full/      # Full sequences
├── diffh2o_representation_grasp/     # Grasp phase only
├── diffh2o_representation_interaction/ # Interaction phase only
├── texts_simple/                     # Simple text annotations
├── texts_detailed/                   # Detailed text annotations
└── texts_grasp/                      # Grasp-specific annotations
```

## Model Checkpoints

Pretrained models are saved in:
```
save/
├── diffh2o_grasp/          # Grasp model checkpoint
├── diffh2o_full/           # Full model (simple text)
├── diffh2o_full_detailed/  # Full model (detailed text)
└── oakink2_full/           # OakInk2 full model
```

---

## OakInk2 Dataset Integration

OakInk2 is a bimanual hand-object manipulation dataset (CVPR 2024) that has been integrated into DiffH2O. Unlike GRAB which only has hand data, OakInk2 includes full-body SMPL-X poses.

### OakInk2 Source Data Location

```
/hhd4/lizhe/dataset/OakInk2/data/
├── anno_preview/           # Annotation pickles (SMPL-X, MANO, object transforms)
├── program/
│   ├── program_info/       # Primitive task info (JSON)
│   └── desc_info/          # Text descriptions (JSON)
├── object_repair/align_ds/ # Object 3D meshes (OBJ files)
└── object_affordance/      # Affordance annotations
```

### OakInk2 Feature Vector Layout (398D)

The preprocessed OakInk2 representation extends GRAB's 117D to 398D:

```
Body root (9D):           world_tsl (3D) + world_rot (6D)
Body pose (126D):         21 body joints × 6D rotation
Left hand PCA (30D):      pos (3D) + orient (6D) + pca_pose (21D)
Right hand PCA (30D):     pos (3D) + orient (6D) + pca_pose (21D)
Left hand quat (67D):     tsl (3D) + pose_coeffs (16 joints × 4D)
Right hand quat (67D):    tsl (3D) + pose_coeffs (16 joints × 4D)
SDF left (21D):           signed distance to object (placeholder)
SDF right (21D):          signed distance to object (placeholder)
Object 1 pose (9D):       position (3D) + rotation (6D)
Object 2 pose (9D):       position (3D) + rotation (6D)
Object 3 pose (9D):       position (3D) + rotation (6D)
─────────────────────────────────────────────────────────
Total: 398D
```

### OakInk2 Processing Pipeline

1. **Preprocessing** (`prepare/preprocess_oakink2.py`):
   - Loads SMPL-X/MANO data from pickle files
   - Converts quaternions to axis-angle then PCA (21 components)
   - OakInk2 is already 30 FPS (no downsampling needed)
   - Supports two extraction modes:
     - **Primitive mode**: Extracts individual primitive segments (GRAB-like)
     - **Complex mode**: Extracts full sequences for long motion generation
   - Computes BPS encodings for objects

2. **Dataset Class** (`data_loaders/humanml/data/oakink2_dataset.py`):
   - `OakInk2Dataset`: Core dataset implementation
   - `OakInk2`: Wrapper class for DiffH2O compatibility

3. **Configuration** (`configs/data.py`, `configs/card.py`):
   - `oakink2_base`: Base OakInk2 config
   - `oakink2_full`: Full model config (398D features)

### OakInk2 Commands

```bash
# Preprocess OakInk2 - Primitive mode (short segments, like GRAB)
python -m prepare.preprocess_oakink2 --mode primitive --num_samples 10

# Preprocess OakInk2 - Complex mode (full sequences for long motion)
python -m prepare.preprocess_oakink2 --mode complex --num_samples 10

# Preprocess OakInk2 - Both modes
python -m prepare.preprocess_oakink2 --mode both --num_samples 0

# Train OakInk2 model (primitive mode)
python -m train.train_oakink2

# Optional: specify MANO model path and max frames
python -m prepare.preprocess_oakink2 --mano_path /path/to/mano/models --max_frames 1000
```

### OakInk2 Preprocessed Data Structure

```
dataset/OAKINK2/
├── oakink2_primitive/           # Primitive mode motion files (*.npy, shape: T×398)
├── texts_primitive/             # Primitive mode text annotations
├── oakink2_complex/             # Complex mode motion files (full sequences)
├── texts_complex/               # Complex mode texts (high-level + concatenated)
├── Mean_oakink2_primitive.npy   # Normalization mean for primitives
├── Std_oakink2_primitive.npy    # Normalization std for primitives
├── Mean_oakink2_complex.npy     # Normalization mean for complex
├── Std_oakink2_complex.npy      # Normalization std for complex
├── train_oakink2_primitive.txt  # Training split (primitive)
├── test_oakink2_primitive.txt   # Test split (primitive)
├── train_oakink2_complex.txt    # Training split (complex)
├── test_oakink2_complex.txt     # Test split (complex)
├── bps_enc_oakink2.npy          # Object BPS encodings (dict)
├── file_names_primitive.txt     # Sequence ID mapping (primitive)
└── file_names_complex.txt       # Sequence ID mapping (complex)
```

### Key Differences: GRAB vs OakInk2

| Aspect | GRAB | OakInk2 |
|--------|------|---------|
| Feature dim | 117D | 398D |
| Body data | Hands only | Full body (21 joints) |
| Hand repr | PCA only | PCA + Quaternions |
| Objects | Single | Multi-object (up to 3) |
| Frame rate | 30 FPS | 30 FPS (native) |
| Annotations | Simple/Detailed | Primitive + Complex |
| Modes | Single | Primitive (short) / Complex (long) |

### OakInk2 Key Files

- `prepare/preprocess_oakink2.py`: Main preprocessing script
- `data_loaders/humanml/data/oakink2_dataset.py`: Dataset class
- `data_loaders/get_data.py`: Dataset factory (supports 'oakink2')
- `configs/data.py`: `oakink2_base`, `oakink2_full` configs
- `configs/card.py`: `oakink2_full` model card
- `train/train_oakink2.py`: Training script

### OakInk2 TODO / Known Issues

1. **SDF Computation**: Currently placeholder (zeros). Need to implement:
   - Load MANO layer to get hand joint positions
   - Query distance from joints to object mesh surface

2. **Multi-object BPS**: Currently returns first object's BPS. Need to:
   - Track which objects are used per segment
   - Concatenate BPS for all objects

3. **Model Architecture**: 398D features may require:
   - Adjusted latent dimensions
   - Different UNet channel multipliers

4. **Visualization**: Need OakInk2-specific visualization that:
   - Renders full body (not just hands)
   - Supports multiple objects

### OakInk2 Text Annotation Format

OakInk2 uses the same format as GRAB:
```
caption#tokens#start_time#end_time
```

Example:
```
Press the button on the scale.##0.0#0.0
```

The text descriptions come from `program/desc_info/*.json` and describe primitive actions like "press_button", "place_onto", "pour", etc.

---

## ChangeLog

### 2026-01-07: OakInk2 Training Compatibility Fix

Fixed `'OakInk2' object has no attribute 't2m_dataset'` error when running `python -m train.train_oakink2`.

**Root Cause**: The DiffH2O training code expects the dataset to have a specific interface (`t2m_dataset` attribute with transform methods) that the original OakInk2 implementation was missing.

**Files Modified**:

1. **`data_loaders/humanml/data/oakink2_dataset.py`**:
   - Added `inv_transform(data)` method for numpy inverse normalization
   - Added `inv_transform_th(data, traject_only=None, use_rand_proj=None)` method for torch inverse normalization
   - Added `transform_th(data, traject_only=None, use_rand_proj=None)` method for torch forward normalization
   - Added `self.t2m_dataset = self.dataset` alias in `OakInk2` wrapper class for compatibility with training code that accesses `data.dataset.t2m_dataset.*`

2. **`utils/model_util.py`**:
   - Added `'oakink2'` to the list of datasets using text conditioning (line 50): `elif args.dataset in ['kit', 'humanml', 'grab', 'oakink2']`
   - Added handling for OakInk2's 398D feature dimension (lines 73-76):
     ```python
     elif args.dataset == 'oakink2':
         data_rep = 'hml_vec'
         nfeats = 1
         njoints = 398  # OakInk2 feature dimension
     ```

**Verification**: Training now runs successfully with:
- Correct 398D feature dimension (`dims: [398, 1024, 1024, 1024, 1024]`)
- Text-based conditioning (`EMBED TEXT`)
- Loss values being logged and models being saved

### 2026-01-07: OakInk2 Preprocessing Improvements

Improved OakInk2 preprocessing with corrected FPS handling and dual extraction modes.

**FPS Correction**:
- OakInk2 is natively 30fps (4 camera views are angles, not temporal oversampling)
- Removed incorrect 120→30fps downsampling

**New Features**:
- Added `--mode` argument: `primitive`, `complex`, or `both`
- **Primitive mode**: Extracts individual primitive segments (GRAB-like training)
- **Complex mode**: Extracts full sequences for long motion generation (up to 20 minutes)
- Both text formats stored for complex tasks:
  - High-level description from `task_target.json`
  - Concatenated primitive descriptions

**Files Modified**:
- `prepare/preprocess_oakink2.py`: Removed downsampling, added modes, dual text support
- `CLAUDE.md`: Updated documentation

**Output Structure**:
```
dataset/OAKINK2/
├── oakink2_primitive/           # Short primitive segments
├── texts_primitive/
├── oakink2_complex/             # Full sequences (2-20 minutes)
├── texts_complex/               # Both high-level and concatenated texts
├── Mean_oakink2_primitive.npy
├── Std_oakink2_primitive.npy
├── Mean_oakink2_complex.npy
├── Std_oakink2_complex.npy
└── ...
```

### 2026-01-08: WandB Integration for Training

Added Weights & Biases logging platform support for experiment tracking.

**New Features**:
- `WandbPlatform` class in `train/train_platforms.py`
- Supports local wandb server via environment variables
- Logs scalars, hyperparameters, and run configuration

**Usage**:
```bash
# Source wandb environment variables
source /path/to/wandb_init.sh

# Run training with WandB logging
python -m train.train_oakink2 --train_platform_type WandbPlatform
```

**Environment Variables**:
- `WANDB_BASE_URL`: WandB server URL (for local deployment)
- `WANDB_API_KEY`: API key
- `WANDB_ENTITY`: User/team name
- `WANDB_PROJECT`: Project name (default: `diffh2o`)
- `WANDB_RUN_NAME`: Run name (default: save_dir basename)

**Files Modified**:
- `train/train_platforms.py`: Added `WandbPlatform` class
- `train/train_oakink2.py`: Added `WandbPlatform` import


**Conda Environment**:
- Use conda environment rog_env for running training or testing or data preprocessing step

---

## IMPORTANT: Original DiffH2O vs OakInk2 Extension

**Original DiffH2O** (the published paper):
- Trained on **GRAB dataset** with **117D features** (hands + object, NO body)
- Training scripts: `train_grasp.py`, `train_diffh2o.py`, `train_diffh2o_detailed.py`
- Config card: `diffh2o_full`, `diffh2o_grasp`
- Dataset: `dataset/GRAB_HANDS/` (117D: hand PCA + SDF + object pose)
- Model: UNet with AdaGN, 117D input
- See README.md line 184: "DiffH2O is trained on the GRAB dataset"

**OakInk2 Extension** (our custom work):
- Extended to **OakInk2 dataset** with **398D features** (full body + hands + multi-object)
- Training script: `train_oakink2.py`
- Config card: `oakink2_full`
- Dataset: `dataset/OAKINK2/` (398D features, broken PCA due to chumpy issue)
- This is NOT the original paper's pipeline

**CRR-Flow** (new flow matching model):
- Training script: `train_flow.py`
- Dataset: `dataset/OAKINK2_FLOW/` (360D/315D/135D/60D variants)
- Model: FlowDiT with state labels
- Uses TorchCFM for flow matching OR standard diffusion loss

**Do NOT confuse these three pipelines.**

---

## Testing Guidelines

**All experiments MUST run in background and save both motion files (.npy) and visualization videos (.mp4).**

### Environment Notes
- **rog_env**: For generation (training, inference, data preprocessing)
- **mdm2**: For visualization (has aitviewer for SMPL-X mesh rendering)
- Each visualization must run in a **separate process** (aitviewer GL context reuse bug)

### Step 1: Test Single-Motion Model

```bash
# Generate motions (background, rog_env)
conda run -n rog_env python -m sample.generate_oakink2 \
  --model_path ./save/oakink2_full/model000400015.pt \
  --num_samples 4 --batch_size 4 \
  --output_dir output/oakink2_single_test &

# Extract individual .npy files + visualize (mdm2 env, one process per video)
for i in 0 1 2 3; do
  xvfb-run -s "-screen 0 1024x768x24" conda run -n mdm2 \
    python -m visualize.visualize_oakink2 \
    --motion_path output/oakink2_single_test/motion_${i}.npy \
    --output_path output/oakink2_single_test/motion_${i}.mp4 \
    --resolution low &
done
```

### Step 2: Test Transition Model

```bash
# Generate transitions (background, rog_env)
conda run -n rog_env python -m sample.generate_transition \
  --model_path save/oakink2_transition/model000200000.pt \
  --num_samples 4 --output_dir output/transition_test &

# Visualize (mdm2 env, one process per video)
for i in 0 1 2 3; do
  xvfb-run -s "-screen 0 1024x768x24" conda run -n mdm2 python -c "
from visualize.visualize_transitions import render_motion_file
render_motion_file('output/transition_test/full_00000${i}.npy',
                   'output/transition_test_vis/full_00000${i}.mp4',
                   resolution='low')
" &
done
```

### Step 3: End-to-End Test

```bash
conda run -n rog_env python generate_long_motion_cli.py \
  --prompt "Press the button on the scale. Place the bowl onto the platform of the scale." \
  --model_path save/oakink2_full/model000400015.pt \
  --transition_model_path save/oakink2_transition/model000200000.pt \
  --use_trained_transitions --dataset_type oakink2 \
  --output output/e2e_test &
```

### Visualization Standard (MANDATORY)

When visualizing any generated motion, **always produce three versions** with post-processing smoothing. All frames must be saved.

```python
from scipy.signal import savgol_filter

motion = np.load('generated_motion.npy')  # [T, D]

# Version 1: Raw (no post-processing)
motion_raw = motion

# Version 2: Medium smoothing (Savitzky-Golay, window=11, polyorder=3)
motion_medium = savgol_filter(motion, window_length=11, polyorder=3, axis=0)

# Version 3: Heavy smoothing (Savitzky-Golay, window=21, polyorder=3)
motion_heavy = savgol_filter(motion, window_length=21, polyorder=3, axis=0)
```

Save all three as `.npy` and render as `.mp4`:
```
output/<test_name>/
├── gen_raw.npy + gen_raw.mp4             # No post-processing
├── gen_medium.npy + gen_medium.mp4       # SG w=11, p=3
├── gen_heavy.npy + gen_heavy.mp4         # SG w=21, p=3
├── gen_raw_frames/frame_*.png            # All frames saved
├── gen_medium_frames/frame_*.png
├── gen_heavy_frames/frame_*.png
└── gt.npy + gt.mp4                       # Ground truth reference
```

Render each version in a **separate xvfb-run process** (aitviewer GL context reuse bug).

### Output Structure Convention

All test outputs go in `output/<test_name>/`:
```
output/<test_name>/
├── motion_*.npy     # Individual motion files [T, D] denormalized
├── motion_*.mp4     # Corresponding visualization videos
├── stats.txt        # Diagnostic statistics (GT vs generated ranges)
└── args.json        # Generation parameters used
```
## Wandb-local
export WANDB_API_KEY=local-875ae1f1e11ab3854264013801021d95d9fa03aa
export WANDB_BASE_URL=http://172.18.36.108:8080
export WANDB_ENTITY=lizhe

---

## Paper Visualization Pipeline

### Overview

For paper figures, we render SMPL-X body meshes + object meshes from AR chain generation results. There are two types of visualization: **generated motion** and **ground truth (GT) motion**, both frame-aligned for comparison.

### Key Issue: Object Slot Misalignment in AR Chain

The AR chain concatenates multiple preprocessed segments (each 200 frames). **Each segment has different objects in the 4 object slots** (sorted alphabetically per segment). When concatenated to 560 frames, the object slot identities change at segment boundaries (frame ~200, ~380). This means:

- **DO NOT** use the 36D object channels (indices 231-267) from `full_gt.npy` / `full_gen.npy` directly for multi-segment rendering with a single object mapping.
- **DO** use `rebuild_gt_aligned.py` to extract GT body + objects from raw OakInk2 annotations with proper per-frame alignment.

### Rendering Pipeline: GT Body + Objects (Properly Aligned)

```bash
# Step 1: Build aligned GT data (rog_env, needs smplx)
# This recovers the exact OakInk2 frame IDs for each of the 560 generated frames,
# then extracts raw SMPL-X body + ALL object transforms at those frames.
conda run -n rog_env python -m visualize.rebuild_gt_aligned \
  --mode fit --scene scoop_cheese \
  --output_dir output/ar_chain_scoop_cheese/gt_aligned \
  --device cuda:0  # or cpu

# Step 2: Render video (mdm2 env, headless)
xvfb-run -a -s "-screen 0 1280x960x24" conda run -n mdm2 \
  python -m visualize.rebuild_gt_aligned --mode render \
  --output_dir output/ar_chain_scoop_cheese/gt_aligned \
  --output_path output/ar_chain_scoop_cheese/gt_aligned.mp4 \
  --resolution medium
```

**Available scenes** (configured in `SCENE_MAP` dict in script):
- `scoop_cheese`: seg 0, 2, 3 of `scene_01__A003++...`
- `lightbulb`: seg 0, 1, 2 of `scene_02__A001++...`

### Rendering Pipeline: Generated Body + GT Objects

The generated body comes from the model's output (marker fitting to SMPL-X), while objects use GT transforms from the aligned data:

```bash
# Use render_with_objects/ for gen body, gt_aligned/ for GT objects
# See render_full.py:render_video() for the rendering function
```

### Rendering Single Frames (for debugging / paper figures)

Each frame must be rendered in a **separate process** (aitviewer GL context reuse bug):

```bash
# Render frame $fi with GT body + all objects
xvfb-run -a -s "-screen 0 1280x960x24" conda run -n mdm2 \
  python3 /tmp/render_gt_aligned_frame.py $fi

# Render 10 frames in parallel
for fi in 0 10 20 30 40 50 60 70 80 90; do
  xvfb-run -a -s "-screen 0 1280x960x24" conda run -n mdm2 \
    python3 render_script.py $fi &
done
wait
```

### Exporting OBJ Files for Blender

```bash
# Export frame-by-frame OBJ files (gen human, gen objects, gt human, gt objects)
# Sampled every 3 frames → ~187 OBJ files per mesh type
conda run -n rog_env python -m visualize.export_obj_frames \
  --scene scoop_cheese --step 3
```

Output structure:
```
output/ar_chain_<scene>/obj_export/
├── gen_human/        frame_0000.obj, frame_0003.obj, ...
├── gen_objects/<id>/ frame_0000.obj, ...
├── gt_human/         frame_0000.obj, ...
└── gt_objects/<id>/  frame_0000.obj, ...
```

### OakInk2 Scene Objects

Each OakInk2 sequence has up to 14 objects. To see all objects at a specific frame, load from raw annotations:

```python
# Save individual object OBJ at a specific frame (world-space positioned)
# See debug_vis/scoop_cheese/obj_*_frame0.obj for examples
```

Object ID prefixes: `C` = container (cups/bowls), `O02` = small manipulable objects, `O93` = food items, `S` = scene furniture (static).

### aitviewer Rendering Notes

- **Alpha must be 1.0**: Using alpha < 1.0 (e.g., 0.9) causes horizontal line artifacts due to internal SMPL-X geometry (mouth, eye sockets) showing through. Always use `color=(r, g, b, 1.0)`.
- **z_up=False**: SMPL-X data is Y-up. Pass `z_up=False` to `Meshes()`.
- **GL context reuse bug**: Each render must run in a separate process. Cannot create multiple `HeadlessRenderer` instances in one process.
- **Environments**: `rog_env` (Python 3.10) for fitting/SMPL-X, `mdm2` (Python 3.8) for aitviewer rendering. No `list[int]` type hints in mdm2-compatible code.

### Model Selection for AR Chain Generation

**IMPORTANT**: The AR chain (`generate_flow_autoreg.py`) requires a well-trained model. Testing revealed:

| Model | Steps | Max Movement | GT Movement | Status |
|-------|-------|-------------|-------------|--------|
| `v4_entity_200k/model000200000.pt` | 200k | 0.197m | 0.189m | **Working** |
| `v4_autoreg_release_200k/model000060000.pt` | 60k | 0.060m | 0.189m | **Undertrained** - near-static output |

The v4_autoreg_release model at 60k steps generates near-static motion. Use `v4_entity_200k` at 200k steps (or later checkpoints of autoreg) for proper generation.

### Per-Segment Object Slot Mapping

Each preprocessed segment has its own object set (sorted alphabetically, max 4). When rendering generated objects, use the correct per-segment mapping:

```python
# Example: scoop_cheese sequence segments
# seg 0 (000214): Slot 0=C11001 (cup), Slot 1=O02@0030@00002 (spoon)
# seg 2 (000216): Slot 0=O02@0039@00001 (microwave), Slot 1=O02@0039@00002 (gate), Slot 2=O02@0039@00004 (button)
# seg 3 (000217): Slot 0=C11001 (cup)
```

To find slot mapping for any segment, check `preprocess_v3.py:585`:
```python
obj_slot_names = sorted(list(seg_objects))[:4]
```

### Three-Way Comparison Pipeline (Paper Figures)

For each scene/segment, generate 3 comparison visualizations:

| Type | Body | Active Objects | Scene Objects | Script |
|------|------|---------------|---------------|--------|
| **GT aligned** | Raw SMPL-X from annotation | GT objects from annotation | GT from annotation | `rebuild_gt_aligned.py` |
| **Gen body + GT objects** | SMPL-X fitted from gen markers | GT objects from annotation | GT from annotation | render with `entity_gen_body_verts.npy` + `gt_aligned/obj_*` |
| **Gen body + Gen objects** | SMPL-X fitted from gen markers | Gen objects from 267D motion (per-segment slot mapping) | GT from annotation | render with gen body + 6D rot transform of gen object slots |

**Full workflow for one segment** (example: scoop_cheese seg 0 = sample 000214):

```bash
# 1. Generate motion with entity model (rog_env)
conda run -n rog_env python3 -c "
from sample.generate_flow import create_model, generate_motion
# ... generate entity_gen_000214.npy and gt_000214.npy
"

# 2. Build GT aligned body + all objects (rog_env)
conda run -n rog_env python -m visualize.rebuild_gt_aligned \
  --mode fit --scene scoop_cheese \
  --output_dir debug_vis/scoop_cheese/gt_aligned

# 3. Fit SMPL-X to generated markers (rog_env)
conda run -n rog_env python3 -c "
from visualize.marker2smplx import fit_smplx_to_markers
# ... save entity_gen_body_verts.npy
"

# 4. Render all 3 types (mdm2, parallel per-frame)
# Each frame in a separate process (aitviewer GL context bug)
for fi in 0 20 40 ...; do
  xvfb-run -a ... python3 render_gt_aligned_frame.py $fi &
  xvfb-run -a ... python3 render_entity_gen_frame.py $fi &
  xvfb-run -a ... python3 render_entity_gen_gen_obj_frame.py $fi &
done
wait
```

**Output naming convention** in `debug_vis/<scene>/`:
- `gt_aligned_frame{NNN}.png` — GT body + GT objects
- `entity_gen_gt_obj_frame{NNN}.png` — Gen body + GT objects
- `entity_gen_gen_obj_frame{NNN}.png` — Gen body + Gen objects

### Key Visualization Scripts

| Script | Purpose |
|--------|---------|
| `visualize/rebuild_gt_aligned.py` | Build frame-aligned GT body + ALL objects from raw annotations |
| `visualize/render_full.py` | Render body + objects from saved .npy mesh data to MP4 |
| `visualize/render_raw_smplx_gt_objects.py` | Render raw SMPL-X for full GT sequence (not AR-chain-aligned) |
| `visualize/export_obj_frames.py` | Export per-frame OBJ files for Blender import |
| `visualize/visualize_oakink2.py` | Original OakInk2 visualization (single sequences) |