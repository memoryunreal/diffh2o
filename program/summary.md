# DiffH2O Project Summary

## Table 1: Data Processing Methods

| Processing Method | Data Format (Dim × Frames) | Processing Script | Storage Location | Visualization Method |
|---|---|---|---|---|
| **GRAB Full** (original DiffH2O) | 117D × ~170f (hand PCA 30D×2 + SDF 42D + object 15D) | External (GRAB pipeline) | `dataset/GRAB_HANDS/diffh2o_representation_full/` | `visualize/visualize_sequences.py --is_pca` (MANO hand mesh + object) |
| **GRAB Grasp** | 117D × ~50f (grasp phase only) | External | `dataset/GRAB_HANDS/diffh2o_representation_grasp/` | Same as above |
| **GRAB Interaction** | 117D × ~120f (interaction phase only) | External | `dataset/GRAB_HANDS/diffh2o_representation_interaction/` | Same as above |
| **OakInk2 Primitive** (398D) | 398D × ~200f (body 135D + hand PCA 60D + MANO quat 134D + SDF 42D + obj 27D) | `prepare/preprocess_oakink2.py --mode primitive` | `dataset/OAKINK2/oakink2_primitive/` | `visualize/visualize_oakink2.py --motion_path` (SMPL-X body mesh + object) |
| **OakInk2 Complex** (398D) | 398D × variable (full sequences up to 10K frames) | `prepare/preprocess_oakink2.py --mode complex` | `dataset/OAKINK2/oakink2_complex/` | Same as above |
| **OakInk2 Transitions** (398D) | 398D × 60f (resampled gap segments) | `prepare/extract_transitions.py` | `dataset/OAKINK2/oakink2_transitions/` | `visualize/visualize_transitions.py` |
| **Flow Full** (360D) | 360D × 200f (body 135D + hand wrist 18D + hand 6D rot 180D + obj 27D) | `prepare/preprocess_oakink2_flow.py --feature_mode full` | `dataset/OAKINK2_FLOW/` | `visualize/visualize_oakink2.py --no_denormalize` |
| **Flow Body Only** (135D) | 135D × 200f (body_transl 3D + body_orient 6D + body_pose 126D) | `prepare/preprocess_oakink2_flow.py --feature_mode body_only` | `dataset/OAKINK2_FLOW_135D/` | `visualize/visualize_oakink2.py` (expand 135D→398D, body mesh) |
| **Flow Body+Hands** (315D) | 315D × 200f (body 135D + lhand 6D rot 90D + rhand 6D rot 90D) | `prepare/preprocess_oakink2_flow.py --feature_mode body_hands` | `dataset/OAKINK2_FLOW_315D/` | Same as 135D (expand to 398D) |
| **Flow Hand PCA** (60D) | 60D × 200f (hand pos 3D + orient 6D + PCA 21D per hand) | `prepare/preprocess_oakink2_flow.py --feature_mode hand_pca` | `dataset/OAKINK2_FLOW_HANDPCA/` | `visualize/visualize_sequences.py` (expand 60D→117D, MANO hand mesh) |
| **77 Markers** (258D) | 258D × 400f (77 SMPL-X surface markers 231D + 3 objects 27D) | `prepare/preprocess_oakink2_flow.py --feature_mode markers` | `dataset/OAKINK2_MARKERS/` | 77-marker spheres (aitviewer) or SMPL-X inverse fitting → body mesh + object |
| **OakInk2 as GRAB** (117D) | 117D × 200f (hand PCA mapped to GRAB layout, single sequence) | Inline script (see Way 3 experiment) | `dataset/OAKINK2_AS_GRAB/` | `visualize/visualize_sequences.py --is_pca` |

### Notes
- All Flow/Marker variants include per-frame **state arrays** `[T, 20]` with 7 entity states (Idle/Reach/Grasped/Grasped+Interacting/Release/Static/Interacting)
- All Flow/Marker variants include **text annotations** in `texts/*.txt` (format: `caption#tokens#start#end`)
- Normalization: each dataset has `Mean_flow.npy` and `Std_flow.npy` for per-channel normalization
- GRAB uses `Mean_diffh2o_*.npy` / `Std_diffh2o_*.npy` with optional random projection (`rand_proj_diffh2o.npy`)
- 77 markers are from InterAct's SSM67 body + 10 fingertips protocol (`/hhd4/lizhe/code/InterAct/process/markerset.py`)

---

## Table 2: Model Training Methods and Experiment Results

| Variant | Model Arch | Loss Type | Input Dim | Input Category | Condition Input | Training Script | Train Steps | Train Seqs | Loss @final | MSE | Model Storage | Visualization Results |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DiffH2O Grasp** (original) | UNet AdaGN | Diffusion (predict x₀) | 117D | Hand PCA + SDF + Object | Text (CLIP) + Object BPS | `train/train_grasp.py` | 200K | 2670 (GRAB) | — | — | `save/diffh2o_grasp/` | `visualize/visualize_sequences.py` |
| **DiffH2O Full** (original) | UNet AdaGN | Diffusion (predict x₀) | 117D | Hand PCA + SDF + Object | Text (CLIP) + Object BPS | `train/train_diffh2o.py` | 200K | 2673 (GRAB) | — | — | `save/diffh2o_full/` | `visualize/visualize_sequences.py` |
| **DiffH2O Full Detailed** (original) | UNet AdaGN | Diffusion (predict x₀) | 117D | Hand PCA + SDF + Object | Detailed Text (CLIP) + Object BPS | `train/train_diffh2o_detailed.py` | 200K | 2673 (GRAB) | — | — | `save/diffh2o_full_detailed/` | `visualize/visualize_sequences.py` |
| **OakInk2 Full** (398D extension) | UNet AdaGN | Diffusion (predict x₀) | 398D | Body + Hand PCA + MANO quat + SDF + Object | Text (CLIP) + Object BPS | `train/train_oakink2.py` | 400K | 2840 (OakInk2 prim) | — | — | `save/oakink2_full/` | `visualize/visualize_oakink2.py` |
| **Way 1** (FlowDiT + diffusion, 30K) | FlowDiT (4.5M) | Diffusion (predict x₀) | 135D | Body only (6D rot) | Text (CLIP) + State labels | `train/train_flow.py --use_diffusion` | 30K | 1 (overfit) | 0.090 | 0.104 | `save/way1_diffusion_135d/` | `output/way1_vis/gen.mp4` |
| **Way 1 ext** (100K) | FlowDiT (4.5M) | Diffusion (predict x₀) | 135D | Body only (6D rot) | Text (CLIP) + State labels | `train/train_flow.py --use_diffusion --resume` | 100K | 1 (overfit) | 0.055 | 0.030 | `save/way1_diffusion_135d_100k/` | `output/way1_100k_vis/gen.mp4` |
| **Way 2** (FlowDiT + FM, 30K) | FlowDiT (4.5M) | Flow matching (velocity) | 60D | Hand PCA (pos+orient+pca ×2) | Text (CLIP) + State labels | `train/train_flow.py` | 30K | 1 (overfit) | 0.630 | 0.032 | `save/way2_handpca_60d/` | `output/way2_vis/gen_samples/ours_videos/0000_2.mp4` |
| **Way 2 ext** (200K) | FlowDiT (4.5M) | Flow matching (velocity) | 60D | Hand PCA (pos+orient+pca ×2) | Text (CLIP) + State labels | `train/train_flow.py --resume` | 200K | 1 (overfit) | 0.434 | 0.006 | `save/way2_handpca_200k/` | `output/way2_200k_vis/gen_samples/ours_videos/0000_0.mp4` |
| **Way 3** (original DiffH2O UNet) | UNet AdaGN (orig) | Diffusion (predict x₀) | 117D | Hand PCA + SDF + Object (GRAB format) | Text (CLIP) + Object BPS | `train/train_diffh2o.py` | 30K | 1 (overfit) | 0.00002 | ~0 | `save/way3_diffh2o_117d/` | `output/way3_vis/gt_samples/ours_videos/0000_2.mp4` |
| **Way 1 FM 200K** | FlowDiT (4.5M) | Flow matching (velocity) | 135D | Body only (6D rot) | Text (CLIP) + State labels | `train/train_flow.py` | 200K | 1 (overfit) | 0.298 | 0.005 | `save/way1_fm_135d_400k/` | `output/way1_fm_200k_vis/gen.mp4` |
| **Way 1 FM 400K** | FlowDiT (4.5M) | Flow matching (velocity) | 135D | Body only (6D rot) | Text (CLIP) + State labels | `train/train_flow.py` | 400K | 1 (overfit) | 0.264 | **0.003** | `save/way1_fm_135d_400k/` | `output/way1_fm_400k_vis/gen.mp4` |
| **Markers FM 100K** | FlowDiT (31.8M) | Flow matching (velocity) | 258D | 77 markers (231D) + 3 objects (27D) | Text (CLIP) + State labels | `train/train_flow.py --split train_20` | 100K+ | 20 | ~0.1 | 0.007 | `save/markers_fm_200k/` | `output/markers_100k_vis/gen_raw_mesh.mp4`, `gt_mesh.mp4` |
| Flow 398D (early debug) | FlowDiT (4.5M) | Flow matching (velocity) | 398D | Body+HandPCA+MANO+SDF+Obj | Text (CLIP) + State labels | `train/train_flow.py` | 2K | 1 (overfit) | 1.420 | 0.170 | `save/oakink2_flow_debug/` | `output/flow_gen_debug_vis/` |
| Flow 360D (debug) | FlowDiT (4.5M) | Flow matching (velocity) | 360D | Body+Hand6D+Obj | Text (CLIP) + State labels | `train/train_flow.py` | 10K | 1 (overfit) | 0.477 | 0.057 | `save/oakink2_flow_360d/` | `output/flow_360d_vis/` |
| Flow 315D (debug) | FlowDiT (4.5M) | Flow matching (velocity) | 315D | Body+Hand6D | Text (CLIP) + State labels | `train/train_flow.py` | 10K | 1 (overfit) | 1.023 | 0.050 | `save/flow_315d_debug/` | `output/flow_315d_vis/` |
| Flow 135D (debug) | FlowDiT (4.5M) | Flow matching (velocity) | 135D | Body only (6D rot) | Text (CLIP) + State labels | `train/train_flow.py` | 10K | 1 (overfit) | 0.689 | — | `save/flow_135d_debug/` | `output/flow_135d_vis/` |

### Key Findings

1. **Original DiffH2O UNet** (Way 3): loss=0.00002 at 30K — perfect memorization. The proven UNet + diffusion architecture is the gold standard.
2. **FlowDiT + diffusion** (Way 1): loss=0.055 at 100K — body grounded, pose range matches GT (0.22 vs 0.26).
3. **FlowDiT + flow matching** (Way 1 FM 400K): MSE=0.003 — **best reconstruction quality**. Converges slowly but ultimately surpasses diffusion (10× lower MSE).
4. **77-Marker representation** (Markers FM): MSE=0.007 at 100K with 20 training sequences — body shape and position well-captured. Uses SMPL-X inverse fitting for mesh visualization.
5. **Flow matching needs 10-20× more steps** than diffusion to converge, but achieves lower MSE once converged.

### Architecture Comparison

| Architecture | Trainable Params | Input Format | Modulation | Key Feature |
|---|---|---|---|---|
| **DiffH2O UNet** (original) | ~50M+ | `[B, 117, 1, T]` | AdaGN | Skip connections, multi-scale, proven on GRAB |
| **FlowDiT Small** | 4.5M | `[B, T, D]` | AdaLN-Zero | latent=256, layers=4, heads=4, ff=512 |
| **FlowDiT Large** | 31.8M | `[B, T, D]` | AdaLN-Zero | latent=512, layers=8, heads=8, ff=1024 |

### Conditioning Inputs

| Condition | Shape | Source | Used by |
|---|---|---|---|
| **Text** (CLIP) | `list[B]` → 512D | Frozen CLIP ViT-B/32 | All models |
| **State labels** | `[B, T, 20]` int64 | Per-frame entity states (7 classes) | FlowDiT models only |
| **Object BPS** | `[B, 1024, 3]` | Ball Pivot Surface encoding | Original DiffH2O UNet only |
| **Timestep** | `[B]` float | Discrete 0-999 (diffusion) or continuous 0-1 (flow) | All models |

### Visualization Methods

| Method | Script | Environment | Input | Output |
|---|---|---|---|---|
| **GRAB hand mesh** (MANO PCA) | `visualize/visualize_sequences.py --is_pca --save_video` | mdm2 | `results_*.npy` dict (117D, denormalized) | `.mp4` in `ours_videos/` |
| **SMPL-X body mesh** (raw annotation) | `visualize/visualize_oakink2.py --raw_anno` | mdm2 | Annotation pickle | `.mp4` + `_frames/` |
| **SMPL-X body mesh** (from .npy) | `visualize/visualize_oakink2.py --motion_path --no_denormalize` | mdm2 | `[T, 398]` .npy (denormalized) | `.mp4` + `_frames/` |
| **77-marker spheres** | Inline aitviewer script | mdm2 | `[T, 77, 3]` .npy | `.mp4` + `_frames/` |
| **Markers → SMPL-X inverse fit** | Inline LBFGS optimization script | rog_env (fit) + mdm2 (render) | `[T, 77, 3]` markers | SMPL-X mesh `.mp4` + `_frames/` |
| **State overlay** | `visualize/overlay_flow_states.py` | rog_env | state_arrays + motion | State timeline image |

### Post-Processing (Mandatory per CLAUDE.md)

Every generated motion must be visualized in **3 versions**:
1. **Raw** — no post-processing
2. **Medium** — Savitzky-Golay filter (window=11, polyorder=3, axis=0)
3. **Heavy** — Savitzky-Golay filter (window=21, polyorder=3, axis=0)

All frames must be saved as PNGs. Each version rendered in a **separate xvfb-run process** (aitviewer GL context reuse bug).

### WandB Dashboard

All experiments logged to local WandB at `http://172.18.36.108:8080`:
- Project `crr-flow`: FlowDiT experiments (Way 1, Way 2, Markers)
- Project `diffh2o`: Original DiffH2O experiments (Way 3, OakInk2)
