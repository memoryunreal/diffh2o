# CRR-Flow V2: Object-Centric State-Aware Architecture

## Overview

Building upon the successful convergence of the V1 Marker Model, CRR-Flow V2 scales the architecture from a 20M parameter proof-of-concept to an 86M parameter (DiT-Base class) production model. The fundamental upgrade lies in dismantling the monolithic 258D frame vector into an **Object-Centric Token Sequence** with **Entity-wise State Injection**. 

Instead of treating the human and objects as a single flattened vector, the model now processes a spatio-temporal grid of tokens, allowing the Self-Attention mechanism to dynamically route physical constraints based on entity-specific state labels.

**Model Class:** `CRRFlowDiT_V2`
**Target Parameter Count:** ~86M
**Core Paradigm Shift:** Global Conditioning $\rightarrow$ Entity-wise Residual State Injection.

---

## 1. Input Redesign: Object-Centric Tokenization

Instead of a single `[B, T, 258]` tensor, a single frame is now decomposed into $N=7$ distinct entity slots. 

Based on the 77 SMPL-X markers and a maximum of 4 objects, the entities are partitioned as follows:

| Entity Slot | Feature Composition | Raw Dim | Projection Layer |
|:---|:---|:---|:---|
| **E1: Body** | 50 Markers (Head, Torso, Legs, Feet) $\times$ 3D | 150D | `Linear(150, 768)` |
| **E2: Left Hand** | 14 Markers (9 Hand + 5 Fingertips) $\times$ 3D | 42D | `Linear(42, 768)` |
| **E3: Right Hand**| 13 Markers (8 Hand + 5 Fingertips) $\times$ 3D | 39D | `Linear(39, 768)` |
| **E4: Object 1** | 3D Pos + 6D Rot | 9D | `Linear(9, 768)` |
| **E5: Object 2** | 3D Pos + 6D Rot | 9D | `Linear(9, 768)` |
| **E6: Object 3** | 3D Pos + 6D Rot | 9D | `Linear(9, 768)` |
| **E7: Object 4** | 3D Pos + 6D Rot | 9D | `Linear(9, 768)` |

*Note: Unused object slots (e.g., Objects 3 & 4 in a 2-object scene) are padded with zeros and masked out during attention.*

---

## 2. Entity-wise State Injection (The Core Novelty)

**The Problem in V1:** Padding state labels into a 20-dim vector and using it as a global condition confused the model. It didn't know *which* hand was grasping *which* object.
**The V2 Solution:** State labels are strictly injected into their corresponding entity tokens *before* the Transformer blocks.

### State Label Vocabulary Map
| ID | State | Applied To |
|:---|:---|:---|
| 0 | `[Static]` | Objects at rest |
| 1 | `[Reach]` | Hands approaching |
| 2 | `[Grasped]` | Hands (holding) OR Objects (being held) |
| 3 | `[Interacting]`| Objects being acted upon (e.g., sliced) |
| 4 | `[Release]` | Hands retreating |
| 5 | `[Padding]` | Unused object slots |

### Injection Mechanism (PyTorch Pseudo-code)
```python
# 1. Spatially project all entities to D=768
feat_body = self.proj_body(body_markers)  # [B, T, 768]
feat_lh   = self.proj_hand(lh_markers)    # [B, T, 768]
feat_rh   = self.proj_hand(rh_markers)    # [B, T, 768]
feat_obj1 = self.proj_obj(obj1_pose)      # [B, T, 768]
# ... (same for obj2, obj3, obj4)

# 2. Entity-wise Residual State Injection
feat_lh   = feat_lh   + self.state_embedder(states_lh)
feat_rh   = feat_rh   + self.state_embedder(states_rh)
feat_obj1 = feat_obj1 + self.state_embedder(states_obj1)
# ...

# 3. Stack into Spatio-Temporal Sequence
# Shape transitions to [B, T, 7, 768]
tokens = torch.stack([feat_body, feat_lh, feat_rh, feat_obj1, ...], dim=2)


3. Scaled Architecture Diagram (DiT-Base)
To scale to ~86M parameters and handle the Spatio-Temporal grid, the sequence is flattened, and dual positional encodings are applied.
Hyperparameters:
- latent_dim: 768
- num_layers: 12
- num_heads: 12
- dropout: 0.1

Sequence Reshaping:
Input Tokens: [B, T=400, N=7, D=768]
Flattened:    [B, L=2800, D=768]

  → + PositionalEncoding(Time)    [1, 400, 1, 768] -> broadcasted
  → + PositionalEncoding(Entity)  [1, 1, 7, 768]   -> broadcasted

  Global Condition (c):
      t_emb = SinusoidalTimestepEmbedder(768)
      text_emb = CLIP_ViT-B/32(text) → Linear(512, 768)
      c = MLP(cat(t_emb, text_emb))  [B, 768]

  → 12 × DiTBlock(d=768, heads=12, ff=3072)
      * Key_Padding_Mask is applied here to ignore [Padding] object tokens.
      * Self-Attention inherently learns Spatio-Temporal interactions (e.g., Token LH at t=50 attending to Token Obj1 at t=50).

  → FinalLayer(768)
  → Unflatten back to distinct entities
  → Linear projections back to raw dimensions (150, 42, 39, 9...)

Output: v_t [B, T=400, Raw_Dims] (predicted velocity field per entity)