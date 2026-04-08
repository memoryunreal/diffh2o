"""
Text processing and orchestration for long-term motion generation.
"""

import re
from typing import List, Tuple, Optional, Dict
import nltk
from nltk.tokenize import sent_tokenize

# Try to download punkt tokenizer if not available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt', quiet=True)
    except:
        print("Warning: NLTK punkt tokenizer not available. Using fallback method.")


def split_into_sentences(prompt: str, 
                        use_nltk: bool = True,
                        min_sentence_length: int = 5) -> List[str]:
    """
    Split a multi-sentence paragraph into individual sentences.
    
    Args:
        prompt: Multi-sentence text to split
        use_nltk: Whether to use NLTK for sentence splitting (more accurate)
        min_sentence_length: Minimum character length for a valid sentence
        
    Returns:
        List of clean sentences
    """
    if not prompt or len(prompt.strip()) == 0:
        return []
    
    # Clean the prompt
    prompt = prompt.strip()
    
    if use_nltk:
        try:
            # Use NLTK sentence tokenizer
            sentences = sent_tokenize(prompt)
        except:
            # Fallback to simple splitting
            sentences = _simple_sentence_split(prompt)
    else:
        sentences = _simple_sentence_split(prompt)
    
    # Clean and filter sentences
    clean_sentences = []
    for sent in sentences:
        sent = sent.strip()
        # Filter out very short sentences (likely errors)
        if len(sent) >= min_sentence_length:
            # Ensure sentence ends with proper punctuation
            if not sent[-1] in '.!?':
                sent += '.'
            clean_sentences.append(sent)
    
    return clean_sentences


def _simple_sentence_split(text: str) -> List[str]:
    """
    Simple fallback method for sentence splitting.
    
    Args:
        text: Text to split
        
    Returns:
        List of sentences
    """
    # Split on common sentence endings
    sentences = re.split(r'[.!?]+', text)
    # Remove empty strings and clean
    return [s.strip() for s in sentences if s.strip()]


def extract_objects_from_sentences(sentences: List[str], 
                                  object_list: Optional[List[str]] = None) -> List[str]:
    """
    Extract object names from sentences.
    
    Args:
        sentences: List of sentences
        object_list: List of known objects to look for
        
    Returns:
        List of object names (one per sentence)
    """
    if object_list is None:
        # Default GRAB objects from the codebase
        from data_loaders.humanml.utils.data_utils import OBJECT_LIST_NEW
        object_list = OBJECT_LIST_NEW
    
    extracted_objects = []
    
    for sentence in sentences:
        sentence_lower = sentence.lower()
        found_object = None
        
        # Look for objects in the sentence
        for obj in object_list:
            if obj.lower() in sentence_lower:
                found_object = obj
                break
        
        # If no object found, try to infer from common patterns
        if found_object is None:
            # Common patterns like "picks up the X" or "grabs the X"
            pattern = r'(?:pick.*?up|grab|grasp|hold|take|lift|move|place|put|drop|throw|catch)\s+(?:the\s+)?(\w+)'
            match = re.search(pattern, sentence_lower)
            if match:
                potential_obj = match.group(1)
                # Check if it's a known object
                for obj in object_list:
                    if obj.lower() == potential_obj or potential_obj in obj.lower():
                        found_object = obj
                        break
        
        extracted_objects.append(found_object if found_object else "")
    
    return extracted_objects


def segment_temporal_markers(sentences: List[str]) -> List[Dict[str, any]]:
    """
    Extract temporal markers and dependencies between sentences.
    
    Args:
        sentences: List of sentences
        
    Returns:
        List of dictionaries containing temporal information
    """
    temporal_info = []
    temporal_markers = {
        'sequential': ['then', 'next', 'after', 'afterwards', 'subsequently'],
        'simultaneous': ['while', 'as', 'during', 'simultaneously'],
        'beginning': ['first', 'initially', 'to start'],
        'ending': ['finally', 'lastly', 'to finish']
    }
    
    for i, sentence in enumerate(sentences):
        sent_lower = sentence.lower()
        info = {
            'sentence': sentence,
            'index': i,
            'temporal_type': 'sequential',  # default
            'markers': []
        }
        
        # Check for temporal markers
        for marker_type, markers in temporal_markers.items():
            for marker in markers:
                if marker in sent_lower:
                    info['temporal_type'] = marker_type
                    info['markers'].append(marker)
        
        temporal_info.append(info)
    
    return temporal_info


def plan_segment_durations(sentences: List[str], 
                          default_duration: int = 100,
                          min_duration: int = 50,
                          max_duration: int = 200) -> List[int]:
    """
    Plan the duration (in frames) for each sentence segment.
    
    Args:
        sentences: List of sentences
        default_duration: Default number of frames per segment
        min_duration: Minimum frames per segment
        max_duration: Maximum frames per segment
        
    Returns:
        List of frame counts for each segment
    """
    durations = []
    
    # Keywords that suggest different durations
    quick_actions = ['quickly', 'rapidly', 'fast', 'swift', 'brief']
    slow_actions = ['slowly', 'carefully', 'gently', 'gradually', 'cautiously']
    complex_actions = ['examines', 'inspects', 'studies', 'manipulates', 'adjusts']
    
    for sentence in sentences:
        sent_lower = sentence.lower()
        
        # Start with default
        duration = default_duration
        
        # Adjust based on keywords
        if any(word in sent_lower for word in quick_actions):
            duration = int(duration * 0.7)
        elif any(word in sent_lower for word in slow_actions):
            duration = int(duration * 1.3)
        elif any(word in sent_lower for word in complex_actions):
            duration = int(duration * 1.2)
        
        # Count verbs as a proxy for complexity
        # Simple heuristic: more verbs = more actions = longer duration
        verb_patterns = ['ing ', 'ed ', 's ', 'es ']
        verb_count = sum(1 for pattern in verb_patterns if pattern in sent_lower)
        if verb_count > 2:
            duration = int(duration * 1.1)
        
        # Clamp to valid range
        duration = max(min_duration, min(max_duration, duration))
        durations.append(duration)
    
    return durations


class TextOrchestrator:
    """
    Orchestrates the generation of long-term motion from multi-sentence text.
    """
    
    def __init__(self, 
                 default_segment_frames: int = 100,
                 transition_frames: int = 20,
                 object_list: Optional[List[str]] = None):
        """
        Initialize the orchestrator.
        
        Args:
            default_segment_frames: Default number of frames per segment
            transition_frames: Number of frames for transitions
            object_list: List of known objects
        """
        self.default_segment_frames = default_segment_frames
        self.transition_frames = transition_frames
        self.object_list = object_list
        
    def process_text(self, prompt: str) -> Dict:
        """
        Process multi-sentence text into structured data for generation.
        
        Args:
            prompt: Multi-sentence text prompt
            
        Returns:
            Dictionary containing:
                - sentences: List of individual sentences
                - objects: List of objects per sentence
                - durations: List of frame durations per sentence
                - temporal_info: Temporal relationship information
                - generation_plan: Structured plan for generation
        """
        # Split into sentences
        sentences = split_into_sentences(prompt)
        
        if not sentences:
            raise ValueError("No valid sentences found in the prompt.")
        
        # Extract objects
        objects = extract_objects_from_sentences(sentences, self.object_list)
        
        # Plan segment durations
        durations = plan_segment_durations(
            sentences, 
            default_duration=self.default_segment_frames
        )
        
        # Extract temporal information
        temporal_info = segment_temporal_markers(sentences)
        
        # Create generation plan
        generation_plan = self._create_generation_plan(
            sentences, objects, durations, temporal_info
        )
        
        return {
            'full_text': prompt,
            'sentences': sentences,
            'objects': objects,
            'durations': durations,
            'temporal_info': temporal_info,
            'generation_plan': generation_plan,
            'total_frames': self._estimate_total_frames(generation_plan)
        }
    
    def _create_generation_plan(self, 
                               sentences: List[str],
                               objects: List[str],
                               durations: List[int],
                               temporal_info: List[Dict]) -> List[Dict]:
        """
        Create a detailed generation plan.
        
        Returns:
            List of generation steps
        """
        plan = []
        
        for i in range(len(sentences)):
            # Plan for segment generation
            segment_step = {
                'type': 'segment',
                'index': i,
                'sentence': sentences[i],
                'object': objects[i] if i < len(objects) else None,
                'duration': durations[i],
                'temporal_type': temporal_info[i]['temporal_type']
            }
            plan.append(segment_step)
            
            # Plan for transition (if not last segment)
            if i < len(sentences) - 1:
                transition_step = {
                    'type': 'transition',
                    'index': i,
                    'from_sentence': sentences[i],
                    'to_sentence': sentences[i + 1],
                    'duration': self.transition_frames,
                    'blend_type': 'smooth'  # Can be extended to other types
                }
                plan.append(transition_step)
        
        return plan
    
    def _estimate_total_frames(self, generation_plan: List[Dict]) -> int:
        """
        Estimate total number of frames in the final sequence.
        
        Args:
            generation_plan: Generation plan
            
        Returns:
            Estimated total frames
        """
        total = 0
        for step in generation_plan:
            total += step['duration']
        return total
    
    def set_planner_type(self, planner_type: str):
        """Switch planner type (for compatibility — no-op for heuristic)."""
        pass

    def validate_plan(self, plan: Dict) -> Tuple[bool, List[str]]:
        """
        Validate a generation plan.
        
        Args:
            plan: Generation plan dictionary
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for required fields
        required_fields = ['sentences', 'generation_plan', 'total_frames']
        for field in required_fields:
            if field not in plan:
                issues.append(f"Missing required field: {field}")
        
        # Check sentence count
        if 'sentences' in plan:
            if len(plan['sentences']) == 0:
                issues.append("No sentences to generate")
            elif len(plan['sentences']) > 10:
                issues.append(f"Too many sentences ({len(plan['sentences'])}). Consider limiting to 10.")
        
        # Check total frames
        if 'total_frames' in plan:
            if plan['total_frames'] > 2000:
                issues.append(f"Sequence too long ({plan['total_frames']} frames). Consider shorter text.")
        
        # Check objects
        if 'objects' in plan:
            missing_objects = [i for i, obj in enumerate(plan['objects']) if not obj]
            if missing_objects:
                issues.append(f"No objects detected for sentences: {missing_objects}")
        
        is_valid = len(issues) == 0
        return is_valid, issues


class TemplateOrchestrator:
    """
    Template-based orchestrator that uses the TemplatePlanner for
    more accurate verb-to-primitive mapping and duration estimation.

    Drop-in replacement for TextOrchestrator with the same interface.
    """

    def __init__(self,
                 default_segment_frames: int = 100,
                 transition_frames: int = 60,
                 object_list: Optional[List[str]] = None,
                 dataset_type: str = 'oakink2'):
        self.default_segment_frames = default_segment_frames
        self.transition_frames = transition_frames
        self.dataset_type = dataset_type

        from sample.template_planner import TemplatePlanner
        self.planner = TemplatePlanner(
            dataset_type=dataset_type,
            default_transition_frames=transition_frames,
            min_segment_frames=30,
            max_segment_frames=300,
        )

    def process_text(self, prompt: str) -> Dict:
        """Process multi-sentence text into structured data for generation."""
        plan = self.planner.plan(prompt)
        return self.planner.to_orchestrator_format(plan)

    def validate_plan(self, plan: Dict) -> Tuple[bool, List[str]]:
        """Validate a generation plan."""
        issues = []
        required_fields = ['sentences', 'generation_plan', 'total_frames']
        for field in required_fields:
            if field not in plan:
                issues.append(f"Missing required field: {field}")

        if 'sentences' in plan:
            if len(plan['sentences']) == 0:
                issues.append("No sentences to generate")
            elif len(plan['sentences']) > 10:
                issues.append(f"Too many sentences ({len(plan['sentences'])})")

        if 'total_frames' in plan and plan['total_frames'] > 5000:
            issues.append(f"Sequence too long ({plan['total_frames']} frames)")

        return len(issues) == 0, issues