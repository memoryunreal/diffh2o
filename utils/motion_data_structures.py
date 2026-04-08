"""
Data structures for long-term motion generation.
"""

import os
import torch
from typing import List, Dict, Optional, Union
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MotionSegment:
    """
    Represents a single motion segment generated from one sentence.
    
    Attributes:
        poses: Motion tensor [T, J, D] where T=frames, J=joints, D=dimensions
        text: Original text that generated this segment
        object_repr: Object representation used for this segment
        start_frame: Starting frame index in the full sequence
        end_frame: Ending frame index in the full sequence
        metadata: Additional information about the segment
    """
    poses: torch.Tensor
    text: str
    object_repr: Optional[torch.Tensor] = None
    start_frame: int = 0
    end_frame: int = -1
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_frame == -1:
            self.end_frame = self.start_frame + self.poses.shape[0]
    
    @property
    def num_frames(self) -> int:
        return self.poses.shape[0]
    
    @property
    def num_joints(self) -> int:
        return self.poses.shape[1] if len(self.poses.shape) >= 2 else 0
    
    @property
    def dim_features(self) -> int:
        return self.poses.shape[2] if len(self.poses.shape) >= 3 else 0
    
    def get_last_frames(self, n: int = 5) -> torch.Tensor:
        """Get the last n frames of the segment."""
        return self.poses[-n:] if n < self.num_frames else self.poses
    
    def get_first_frames(self, n: int = 5) -> torch.Tensor:
        """Get the first n frames of the segment."""
        return self.poses[:n] if n < self.num_frames else self.poses
    
    def to_device(self, device: Union[str, torch.device]) -> 'MotionSegment':
        """Move segment to specified device."""
        return MotionSegment(
            poses=self.poses.to(device),
            text=self.text,
            object_repr=self.object_repr.to(device) if self.object_repr is not None else None,
            start_frame=self.start_frame,
            end_frame=self.end_frame,
            metadata=self.metadata
        )


@dataclass
class TransitionSegment(MotionSegment):
    """
    Represents a transition between two motion segments.
    
    Additional Attributes:
        source_segment_idx: Index of the source segment
        target_segment_idx: Index of the target segment
        blend_weights: Optional blending weights for smooth transition
    """
    source_segment_idx: int = -1
    target_segment_idx: int = -1
    blend_weights: Optional[torch.Tensor] = None
    
    def apply_blending(self, alpha: float = 0.5) -> torch.Tensor:
        """Apply blending weights to the transition."""
        if self.blend_weights is not None:
            return self.poses * self.blend_weights
        return self.poses


class MotionSequence:
    """
    Represents a complete long-term motion sequence composed of multiple segments and transitions.
    
    Attributes:
        segments: List of motion segments
        transitions: List of transition segments between motion segments
        full_text: Complete multi-sentence text
        object_repr: Object representation for the entire sequence
    """
    
    def __init__(self, 
                 segments: Optional[List[MotionSegment]] = None,
                 transitions: Optional[List[TransitionSegment]] = None,
                 full_text: str = "",
                 object_repr: Optional[torch.Tensor] = None):
        self.segments = segments or []
        self.transitions = transitions or []
        self.full_text = full_text
        self.object_repr = object_repr
        self._cached_tensor = None
        
    def add_segment(self, segment: MotionSegment):
        """Add a new motion segment to the sequence."""
        self.segments.append(segment)
        self._cached_tensor = None  # Invalidate cache
        
    def add_transition(self, transition: TransitionSegment):
        """Add a transition between segments."""
        self.transitions.append(transition)
        self._cached_tensor = None  # Invalidate cache
        
    def to_tensor(self, include_transitions: bool = True) -> torch.Tensor:
        """
        Concatenate all segments and transitions into a single motion tensor.
        
        Args:
            include_transitions: Whether to include transition segments
            
        Returns:
            Concatenated motion tensor [total_frames, J, D]
        """
        if self._cached_tensor is not None and include_transitions:
            return self._cached_tensor
            
        if not self.segments:
            return torch.tensor([])
        
        motion_parts = []
        
        for i, segment in enumerate(self.segments):
            motion_parts.append(segment.poses)
            
            # Add transition after segment if it exists and not the last segment
            if include_transitions and i < len(self.transitions):
                motion_parts.append(self.transitions[i].poses)
        
        # Concatenate along time dimension
        full_motion = torch.cat(motion_parts, dim=0)
        
        if include_transitions:
            self._cached_tensor = full_motion
            
        return full_motion
    
    def to_numpy(self, include_transitions: bool = True) -> np.ndarray:
        """Convert the sequence to a numpy array."""
        tensor = self.to_tensor(include_transitions)
        return tensor.cpu().numpy() if tensor.is_cuda else tensor.numpy()
    
    @property
    def total_frames(self) -> int:
        """Get total number of frames in the sequence."""
        total = sum(seg.num_frames for seg in self.segments)
        total += sum(trans.num_frames for trans in self.transitions)
        return total
    
    @property
    def num_segments(self) -> int:
        """Get number of segments in the sequence."""
        return len(self.segments)
    
    @property
    def num_transitions(self) -> int:
        """Get number of transitions in the sequence."""
        return len(self.transitions)
    
    def get_segment_boundaries(self) -> List[tuple]:
        """
        Get frame boundaries for each segment and transition.
        
        Returns:
            List of (start_frame, end_frame, type) tuples
        """
        boundaries = []
        current_frame = 0
        
        for i, segment in enumerate(self.segments):
            # Segment boundary
            boundaries.append((current_frame, current_frame + segment.num_frames, 'segment'))
            current_frame += segment.num_frames
            
            # Transition boundary if exists
            if i < len(self.transitions):
                trans = self.transitions[i]
                boundaries.append((current_frame, current_frame + trans.num_frames, 'transition'))
                current_frame += trans.num_frames
                
        return boundaries
    
    def to_device(self, device: Union[str, torch.device]) -> 'MotionSequence':
        """Move entire sequence to specified device."""
        new_segments = [seg.to_device(device) for seg in self.segments]
        new_transitions = [trans.to_device(device) for trans in self.transitions]
        
        return MotionSequence(
            segments=new_segments,
            transitions=new_transitions,
            full_text=self.full_text,
            object_repr=self.object_repr.to(device) if self.object_repr is not None else None
        )
    
    def save(self, path: str):
        """Save the motion sequence to disk."""
        import pickle
        data = {
            'segments': self.segments,
            'transitions': self.transitions,
            'full_text': self.full_text,
            'object_repr': self.object_repr,
            'tensor': self.to_tensor(),
            'boundaries': self.get_segment_boundaries()
        }
        
        # Save as both pickle and numpy for compatibility
        with open(path.replace('.npy', '.pkl'), 'wb') as f:
            pickle.dump(data, f)
        
        # Also save just the tensor as numpy
        np.save(path, self.to_numpy())
        
    @classmethod
    def load(cls, path: str) -> 'MotionSequence':
        """Load a motion sequence from disk."""
        import pickle
        
        pkl_path = path.replace('.npy', '.pkl')
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            return cls(
                segments=data['segments'],
                transitions=data['transitions'],
                full_text=data['full_text'],
                object_repr=data['object_repr']
            )
        else:
            # Fallback to loading just numpy array
            motion_array = np.load(path)
            motion_tensor = torch.from_numpy(motion_array)
            
            # Create a single segment from the loaded tensor
            segment = MotionSegment(
                poses=motion_tensor,
                text="Loaded from numpy file",
                metadata={'source': path}
            )
            
            return cls(segments=[segment])