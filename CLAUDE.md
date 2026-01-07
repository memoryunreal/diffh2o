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