"""
Wrapper for generating single motion segments using existing DiffH2O models.
"""

import os
import torch
import numpy as np
from typing import Optional, Dict, Tuple, Any
from dataclasses import dataclass

from utils.motion_data_structures import MotionSegment
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils import dist_util
from utils.fixseed import fixseed
from data_loaders.get_data import DatasetConfig, get_dataset_loader
from model.cfg_sampler import ClassifierFreeSampleModel
from utils.output_util import recover_from_ric


@dataclass
class SegmentGeneratorConfig:
    """Configuration for segment generation."""
    model_path: str
    grasp_model_path: Optional[str] = None
    device: str = 'cuda'
    guidance_param: float = 2.5
    seed: Optional[int] = None
    use_grasp_guidance: bool = True
    max_frames: int = 200
    batch_size: int = 1
    use_pca: bool = True
    text_detailed: bool = False


class SegmentGenerator:
    """
    Generates individual motion segments from single sentences using existing DiffH2O models.
    """
    
    def __init__(self, config: SegmentGeneratorConfig):
        """
        Initialize the segment generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config
        self.model = None
        self.diffusion = None
        self.grasp_model = None
        self.grasp_diffusion = None
        self.data_loader = None
        self._initialized = False
        
        # Set device
        dist_util.setup_dist(config.device)
        
        # Set seed if provided
        if config.seed is not None:
            fixseed(config.seed)
    
    def initialize(self):
        """Initialize models and data loaders."""
        if self._initialized:
            return
            
        print(f"Loading model from {self.config.model_path}...")
        
        # Load dataset configuration
        self._load_data_config()
        
        # Load main model
        self._load_main_model()
        
        # Load grasp model if needed
        if self.config.use_grasp_guidance and self.config.grasp_model_path:
            self._load_grasp_model()
        
        self._initialized = True
        print("Segment generator initialized successfully.")
    
    def _load_data_config(self):
        """Load data configuration based on model settings."""
        # Create minimal data config for model loading
        from utils.parser_util import generate_args
        from utils.generation_template import get_template
        
        # Get args from model checkpoint
        args = generate_args(model_path=self.config.model_path)
        args = get_template(args, guidance=self.config.use_grasp_guidance)
        
        # Update with our config
        args.device = self.config.device
        args.batch_size = self.config.batch_size
        args.guidance_param = self.config.guidance_param
        args.text_detailed = self.config.text_detailed
        
        # Store args for later use
        self.args = args
        
        # Load dataset
        data_conf = DatasetConfig(
            name=args.dataset,
            batch_size=self.config.batch_size,
            num_frames=self.config.max_frames,
            use_pca=self.config.use_pca,
            text_detailed=self.config.text_detailed,
        )
        
        # Create data loader
        self.data_loader = get_dataset_loader(
            name=args.dataset,
            batch_size=self.config.batch_size,
            num_frames=self.config.max_frames,
            split='test',
            load_mode='text_only'
        )
    
    def _load_main_model(self):
        """Load the main hand-object diffusion model."""
        # Create model and diffusion
        self.model, self.diffusion = create_model_and_diffusion(self.args, self.data_loader)
        
        # Load checkpoint
        load_saved_model(self.model, self.config.model_path)
        
        # Move to device and set to eval mode
        self.model.to(dist_util.dev())
        self.model.eval()
        
        # Wrap with classifier-free guidance if needed
        if self.config.guidance_param != 1:
            self.model = ClassifierFreeSampleModel(
                self.model,
                guidance_param=self.config.guidance_param
            )
    
    def _load_grasp_model(self):
        """Load the grasp guidance model."""
        from utils.parser_util import generate_args
        
        # Get args for grasp model
        grasp_args = generate_args(model_path=self.config.grasp_model_path)
        grasp_args.device = self.config.device
        
        # Create grasp model
        self.grasp_model, self.grasp_diffusion = create_model_and_diffusion(
            grasp_args, self.data_loader
        )
        
        # Load checkpoint
        load_saved_model(self.grasp_model, self.config.grasp_model_path)
        
        # Move to device and set to eval mode
        self.grasp_model.to(dist_util.dev())
        self.grasp_model.eval()
    
    def generate_segment_from_sentence(self, 
                                      sentence: str,
                                      object_repr: Optional[torch.Tensor] = None,
                                      num_frames: Optional[int] = None,
                                      seed: Optional[int] = None) -> MotionSegment:
        """
        Generate a single motion segment from a sentence.
        
        Args:
            sentence: Input sentence describing the motion
            object_repr: Optional object representation tensor
            num_frames: Number of frames to generate (default: from config)
            seed: Random seed for this generation
            
        Returns:
            MotionSegment containing the generated motion
        """
        if not self._initialized:
            self.initialize()
        
        # Set seed if provided
        if seed is not None:
            fixseed(seed)
        
        # Set number of frames
        if num_frames is None:
            num_frames = self.config.max_frames
        
        # Prepare model kwargs
        model_kwargs = self._prepare_model_kwargs(sentence, object_repr)
        
        # Generate motion using diffusion
        motion_tensor = self._generate_motion(model_kwargs, num_frames)
        
        # Create and return MotionSegment
        segment = MotionSegment(
            poses=motion_tensor,
            text=sentence,
            object_repr=object_repr,
            metadata={
                'model_path': self.config.model_path,
                'guidance_param': self.config.guidance_param,
                'num_frames': num_frames
            }
        )
        
        return segment
    
    def _prepare_model_kwargs(self, 
                             sentence: str,
                             object_repr: Optional[torch.Tensor]) -> Dict:
        """
        Prepare model keyword arguments for generation.
        
        Args:
            sentence: Input sentence
            object_repr: Object representation
            
        Returns:
            Dictionary of model kwargs
        """
        # Create text condition
        texts = [sentence] * self.config.batch_size
        
        model_kwargs = {
            'y': {
                'text': texts,
                'scale': torch.ones(self.config.batch_size, device=dist_util.dev()) 
                        * self.config.guidance_param
            }
        }
        
        # Add object representation if provided
        if object_repr is not None:
            model_kwargs['y']['object_repr'] = object_repr
        
        # Add any additional conditions from args
        if hasattr(self.args, 'unconstrained') and self.args.unconstrained:
            model_kwargs['y']['unconstrained'] = True
        
        return model_kwargs
    
    def _generate_motion(self, 
                        model_kwargs: Dict,
                        num_frames: int) -> torch.Tensor:
        """
        Generate motion using the diffusion model.
        
        Args:
            model_kwargs: Model keyword arguments
            num_frames: Number of frames to generate
            
        Returns:
            Generated motion tensor [T, J, D]
        """
        # Set diffusion parameters
        self.diffusion.data_get_mean_fn = self.data_loader.dataset.t2m_dataset.get_std_mean
        self.diffusion.data_transform_fn = self.data_loader.dataset.t2m_dataset.transform_th
        self.diffusion.data_inv_transform_fn = self.data_loader.dataset.t2m_dataset.inv_transform_th
        
        # Generate samples
        sample = self.diffusion.p_sample_loop(
            self.model,
            (self.config.batch_size, self.model.njoints, self.model.nfeats, num_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        
        # Take first sample from batch
        sample = sample[0]  # [J, D, T]
        
        # Permute to [T, J, D]
        sample = sample.permute(2, 0, 1)
        
        # Inverse transform to get actual motion
        sample = self.data_loader.dataset.t2m_dataset.inv_transform_th(
            sample.unsqueeze(0)
        ).squeeze(0)
        
        return sample
    
    def generate_batch(self,
                      sentences: list,
                      object_reprs: Optional[list] = None,
                      num_frames: Optional[list] = None) -> list:
        """
        Generate multiple segments in batch.
        
        Args:
            sentences: List of sentences
            object_reprs: List of object representations (optional)
            num_frames: List of frame counts (optional)
            
        Returns:
            List of MotionSegments
        """
        if not self._initialized:
            self.initialize()
        
        segments = []
        
        # Process in batches
        batch_size = self.config.batch_size
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(sentences))
            
            batch_sentences = sentences[start_idx:end_idx]
            batch_objects = None if object_reprs is None else object_reprs[start_idx:end_idx]
            batch_frames = None if num_frames is None else num_frames[start_idx:end_idx]
            
            # Generate each segment in the batch
            for i, sentence in enumerate(batch_sentences):
                obj_repr = None if batch_objects is None else batch_objects[i]
                n_frames = None if batch_frames is None else batch_frames[i]
                
                segment = self.generate_segment_from_sentence(
                    sentence, obj_repr, n_frames
                )
                segments.append(segment)
        
        return segments
    
    def close(self):
        """Clean up resources."""
        # Clear models from memory
        if self.model is not None:
            del self.model
        if self.grasp_model is not None:
            del self.grasp_model
        if self.diffusion is not None:
            del self.diffusion
        if self.grasp_diffusion is not None:
            del self.grasp_diffusion
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._initialized = False


def generate_segment_from_sentence(sentence: str,
                                  model_path: str,
                                  object_repr: Optional[torch.Tensor] = None,
                                  config: Optional[SegmentGeneratorConfig] = None,
                                  **kwargs) -> MotionSegment:
    """
    Convenience function to generate a single segment without managing the generator.
    
    Args:
        sentence: Input sentence
        model_path: Path to the model checkpoint
        object_repr: Optional object representation
        config: Optional generator configuration
        **kwargs: Additional arguments for SegmentGeneratorConfig
        
    Returns:
        Generated MotionSegment
    """
    if config is None:
        config = SegmentGeneratorConfig(model_path=model_path, **kwargs)
    
    generator = SegmentGenerator(config)
    generator.initialize()
    
    segment = generator.generate_segment_from_sentence(sentence, object_repr)
    
    generator.close()
    
    return segment