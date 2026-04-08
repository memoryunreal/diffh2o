Here is the full prompt packaged cleanly into one Markdown file—ready to paste anywhere:

⸻

Prompt for Viber Coding: Long-Term Hand-Object Motion Generation Extension

We have an existing PyTorch codebase corresponding to Figure 2 (Hand-Object Diffusion).
Currently, the pipeline supports text → hand-object motion generation for a single sentence, using:
	•	CLIP text encoder
	•	Hand Diffusion + Hand-Object Diffusion
	•	Outputs a short motion segment (hand pose sequence)

I want to extend this baseline to support long-term motion generation from multi-sentence text, similar to the idea in Figure 1 (MDM-style long-term motion).

⸻

Goal

Given a multi-sentence paragraph, automatically:
	1.	Split text into sentences
	2.	For each sentence, reuse the existing Hand-Object Diffusion module to generate a motion segment
	3.	Introduce a new Switch Scheduler module that:
	•	Accepts two adjacent segments
	•	Uses a diffusion model to generate a transition motion between them
	•	Ensures smooth, natural continuity in pose and timing
	4.	Concatenate:
	•	segment₁
	•	transition₁→₂
	•	segment₂
	•	transition₂→₃
	•	…
into one long-term hand-object motion sequence

⸻

Detailed Requirements

1. Sentence Processing

Implement a helper function:

def split_into_sentences(prompt: str) -> List[str]:
    """Return a list of clean sentences from the input paragraph."""

Add an orchestrating function:

def generate_long_motion(prompt: str, object_repr, config) -> MotionSequence:
    """
    Main entry point: multi-sentence → long-term hand-object motion.
    """


⸻

2. Reusing the Existing Hand-Object Diffusion Module

Do NOT modify the core diffusion model.

Wrap it into:

def generate_segment_from_sentence(sentence: str, object_repr, seed=None) -> MotionSegment:
    """
    Generate one short hand-object motion segment from a single sentence.
    """

Requirements:
	•	Accepts one sentence at a time
	•	Controls segment length via config
	•	Returns a consistent pose tensor (e.g., [T, J, D])

⸻

3. Switch Scheduler Module (New Component)

Implement a new module:

class SwitchScheduler(nn.Module):
    def forward(self, segment_a: MotionSegment, segment_b: MotionSegment) -> MotionSegment:
        """
        Generate a transition motion that smoothly connects segment_a → segment_b.
        """

Design
	•	A small diffusion model that learns transitions
	•	Forward process:
	•	Start from a noisy linear interpolation between:
	•	last frames of segment A
	•	first frames of segment B
	•	Reverse process:
	•	Denoise into a natural transition respecting hand kinematics & object contact
	•	Conditioning options:
	•	Two adjacent sentences
	•	Frame indices
	•	Object representation

Output
	•	A short transition segment (e.g., 10–30 frames)
	•	Must blend seamlessly:
	•	First frames align with last frames of segment A
	•	Last frames align with first frames of segment B

⸻

4. Data Structures

class MotionSegment:
    poses: torch.Tensor  # [T, J, D]
    metadata: dict

class MotionSequence:
    segments: List[MotionSegment]
    transitions: List[MotionSegment]
    
    def to_tensor(self): ...


⸻

5. Training & Inference

Training (Switch Scheduler)
	•	Input: pairs of real motion clips with natural transitions
	•	Train diffusion model to reconstruct transitions
	•	Store weights separately from baseline diffusion model

Inference
	•	Load baseline hand-object diffusion weights
	•	Load Switch Scheduler weights
	•	For multi-sentence prompts:
	•	Produce segments from each sentence
	•	Generate transitions
	•	Concatenate into long-term output

⸻

6. Configurations & CLI Interface

Expose parameters such as:
	•	Segment length
	•	Transition length
	•	Diffusion steps (for both models)
	•	Seed, device, batch size
	•	Output formats

Provide a CLI entry point:

python generate_long_motion.py \
  --prompt "The person picks up the apple, examines it, then places it down and waves." \
  --output output/long_motion.npy


⸻

7. Code Quality Requirements
	•	Modular
	•	Well-documented
	•	Compatible with existing single-sentence pipeline
	•	New functionality layered on top (no breaking of legacy API)

