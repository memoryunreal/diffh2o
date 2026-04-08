"""
Template-based semantic planner for decomposing multi-sentence prompts into
structured generation plans.

Maps common verbs to known dataset primitives without requiring an LLM API.
This makes the planner fully reproducible and dependency-free.

Supports both OakInk2 and GRAB primitive vocabularies.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ----------------------------------------------------------------
# Verb-to-primitive mappings
# ----------------------------------------------------------------

VERB_TO_PRIMITIVE = {
    # OakInk2 primitives
    "pick up": "grip",
    "grab": "grip",
    "grasp": "grip",
    "grip": "grip",
    "take": "take_outside",
    "remove": "take_outside",
    "place": "place_onto",
    "put": "place_onto",
    "set down": "place_onto",
    "put down": "place_onto",
    "pour": "pour",
    "stir": "stir",
    "mix": "stir",
    "cut": "cut",
    "slice": "cut",
    "chop": "cut",
    "press": "press_button",
    "push": "press_button",
    "open": "open",
    "close": "close",
    "shut": "close",
    "rearrange": "rearrange",
    "move": "rearrange",
    "arrange": "rearrange",
    "hold": "hold",
    "squeeze": "squeeze",
    "twist": "twist",
    "turn": "twist",
    "rotate": "twist",
    "flip": "flip",
    "shake": "shake",
    "wipe": "wipe",
    "clean": "wipe",
    "screw": "screw",
    "unscrew": "unscrew",
    "peel": "peel",
    "tear": "tear",
    "fold": "fold",
    "unfold": "unfold",
    "stack": "stack",
    "insert": "insert",
    # GRAB primitives
    "offhand": "offhand",
    "use": "use",
    "pass": "pass",
    "hand over": "pass",
    "drink": "drink",
    "sip": "drink",
    "eat": "eat",
    "bite": "eat",
    "lift": "lift",
    "raise": "lift",
    "lower": "place_onto",
    "drop": "place_onto",
    "throw": "throw",
    "toss": "throw",
    "catch": "catch",
    "wave": "wave",
    "point": "point",
    "inspect": "inspect",
    "examine": "inspect",
    "look at": "inspect",
}

# Primitive -> typical duration in seconds at 30fps
PRIMITIVE_DURATIONS = {
    "grip": 2.0,
    "take_outside": 2.5,
    "place_onto": 2.5,
    "pour": 4.0,
    "stir": 5.0,
    "cut": 4.0,
    "press_button": 1.5,
    "open": 2.0,
    "close": 2.0,
    "rearrange": 3.0,
    "hold": 3.0,
    "squeeze": 2.0,
    "twist": 2.5,
    "flip": 1.5,
    "shake": 2.0,
    "wipe": 3.0,
    "screw": 3.0,
    "unscrew": 3.0,
    "peel": 4.0,
    "stack": 2.5,
    "insert": 2.0,
    "offhand": 2.0,
    "use": 3.0,
    "pass": 2.0,
    "drink": 3.0,
    "eat": 4.0,
    "lift": 2.0,
    "throw": 1.5,
    "catch": 1.5,
    "wave": 2.0,
    "point": 1.5,
    "inspect": 3.0,
}

# Extended object lists (OakInk2 + GRAB)
OAKINK2_OBJECTS = [
    "tongs", "plate", "knife", "spoon", "fork", "cup", "mug", "bowl",
    "bottle", "pan", "pot", "lid", "chopsticks", "spatula", "ladle",
    "whisk", "peeler", "grater", "can", "jar", "box", "bag",
    "bread", "apple", "banana", "orange", "egg", "butter",
    "scale", "button", "switch", "handle", "tool rack",
    "cleaver", "fruit knife", "scissors",
]

GRAB_OBJECTS = [
    "apple", "binoculars", "camera", "cup", "cylindermedium",
    "cylindersmall", "doorknob", "elephant", "flashlight", "fryingpan",
    "gamecontroller", "hammer", "hand", "headphones", "knife",
    "lightbulb", "mouse", "mug", "phone", "pyramidlarge",
    "pyramidmedium", "pyramidsmall", "scissors", "spherelarge",
    "spheremedium", "spheresmall", "stamp", "stanfordbunny",
    "stapler", "toothbrush", "toothpaste", "train", "watch",
    "waterbottle", "wineglass", "flute", "alarmclock", "banana",
    "duck", "piggybank", "toruslarge", "torusmedium", "torussmall",
    "cubesmall", "cubemedium", "cubelarge",
]

ALL_OBJECTS = list(set(OAKINK2_OBJECTS + GRAB_OBJECTS))

# Hand dominance keywords
BIMANUAL_KEYWORDS = ["both hands", "two hands", "bimanual", "together"]
LEFT_HAND_KEYWORDS = ["left hand", "left-hand"]
RIGHT_HAND_KEYWORDS = ["right hand", "right-hand"]


@dataclass
class PlanStep:
    """A single step in the generation plan."""
    step_type: str  # 'segment' or 'transition'
    index: int
    sentence: str = ""
    primitive: str = ""
    object_name: str = ""
    duration_frames: int = 100
    hand_dominance: str = "right"  # 'right', 'left', 'bimanual'
    temporal_type: str = "sequential"
    # Transition-specific fields
    from_sentence: str = ""
    to_sentence: str = ""
    transition_type: str = "object_switch"  # 'object_switch' or 'same_object'
    blend_type: str = "smooth"


@dataclass
class GenerationPlan:
    """Complete structured generation plan."""
    full_text: str
    sentences: List[str]
    steps: List[PlanStep]
    objects: List[str]
    primitives: List[str]
    durations: List[int]
    total_frames: int
    dataset_type: str = "oakink2"


class TemplatePlanner:
    """
    Template-based planner that decomposes multi-sentence prompts into
    structured generation plans by mapping verbs to known primitives.
    """

    def __init__(self,
                 dataset_type: str = "oakink2",
                 fps: int = 30,
                 default_transition_frames: int = 60,
                 min_segment_frames: int = 30,
                 max_segment_frames: int = 300):
        """
        Args:
            dataset_type: 'oakink2' or 'grab' — determines object list
            fps: Frames per second
            default_transition_frames: Default transition duration
            min_segment_frames: Minimum segment length
            max_segment_frames: Maximum segment length
        """
        self.dataset_type = dataset_type
        self.fps = fps
        self.default_transition_frames = default_transition_frames
        self.min_segment_frames = min_segment_frames
        self.max_segment_frames = max_segment_frames

        if dataset_type == "oakink2":
            self.object_list = OAKINK2_OBJECTS + GRAB_OBJECTS
        else:
            self.object_list = GRAB_OBJECTS

    def plan(self, prompt: str) -> GenerationPlan:
        """Decompose a multi-sentence prompt into a structured generation plan.

        Args:
            prompt: Natural language description of the desired motion

        Returns:
            GenerationPlan with segments and transitions
        """
        sentences = self._split_sentences(prompt)
        if not sentences:
            raise ValueError("No valid sentences found in the prompt.")

        steps = []
        objects = []
        primitives = []
        durations = []

        for i, sentence in enumerate(sentences):
            # Classify primitive
            primitive = self._classify_primitive(sentence)
            primitives.append(primitive)

            # Extract object
            obj = self._extract_object(sentence)
            objects.append(obj)

            # Estimate duration
            duration = self._estimate_duration(sentence, primitive)
            durations.append(duration)

            # Determine hand dominance
            hand = self._detect_hand_dominance(sentence)

            # Detect temporal markers
            temporal = self._detect_temporal_type(sentence, i, len(sentences))

            # Add segment step
            steps.append(PlanStep(
                step_type="segment",
                index=i,
                sentence=sentence,
                primitive=primitive,
                object_name=obj,
                duration_frames=duration,
                hand_dominance=hand,
                temporal_type=temporal,
            ))

            # Add transition step (except after last segment)
            if i < len(sentences) - 1:
                next_obj = self._extract_object(sentences[i + 1])
                is_same_obj = (obj == next_obj and obj != "")
                trans_type = "same_object" if is_same_obj else "object_switch"

                steps.append(PlanStep(
                    step_type="transition",
                    index=i,
                    from_sentence=sentence,
                    to_sentence=sentences[i + 1],
                    duration_frames=self.default_transition_frames,
                    transition_type=trans_type,
                    blend_type="smooth",
                ))

        total_frames = sum(s.duration_frames for s in steps)

        return GenerationPlan(
            full_text=prompt,
            sentences=sentences,
            steps=steps,
            objects=objects,
            primitives=primitives,
            durations=durations,
            total_frames=total_frames,
            dataset_type=self.dataset_type,
        )

    def to_orchestrator_format(self, plan: GenerationPlan) -> Dict:
        """Convert plan to TextOrchestrator-compatible dict format."""
        generation_plan = []
        for step in plan.steps:
            if step.step_type == "segment":
                generation_plan.append({
                    "type": "segment",
                    "index": step.index,
                    "sentence": step.sentence,
                    "object": step.object_name,
                    "duration": step.duration_frames,
                    "temporal_type": step.temporal_type,
                    "primitive": step.primitive,
                    "hand_dominance": step.hand_dominance,
                })
            else:
                generation_plan.append({
                    "type": "transition",
                    "index": step.index,
                    "from_sentence": step.from_sentence,
                    "to_sentence": step.to_sentence,
                    "duration": step.duration_frames,
                    "blend_type": step.blend_type,
                    "transition_type": step.transition_type,
                })

        return {
            "full_text": plan.full_text,
            "sentences": plan.sentences,
            "objects": plan.objects,
            "durations": plan.durations,
            "temporal_info": [{"sentence": s, "index": i, "temporal_type": "sequential", "markers": []}
                              for i, s in enumerate(plan.sentences)],
            "generation_plan": generation_plan,
            "total_frames": plan.total_frames,
        }

    # ----------------------------------------------------------------
    # Internal methods
    # ----------------------------------------------------------------

    def _split_sentences(self, prompt: str) -> List[str]:
        """Split prompt into clean sentences."""
        prompt = prompt.strip()
        if not prompt:
            return []

        # Split on sentence-ending punctuation or commas followed by 'then/and then'
        # First handle "then" and "and then" as sentence separators
        prompt = re.sub(r',\s*(?:and\s+)?then\s+', '. ', prompt)
        prompt = re.sub(r'\.\s*(?:And\s+)?[Tt]hen\s+', '. ', prompt)

        # Split on periods, exclamation marks, question marks
        parts = re.split(r'[.!?]+', prompt)
        sentences = []
        for part in parts:
            part = part.strip()
            if len(part) >= 5:
                if not part[-1] in '.!?':
                    part += '.'
                sentences.append(part)
        return sentences

    def _classify_primitive(self, sentence: str) -> str:
        """Map a sentence to the best matching primitive type."""
        sent_lower = sentence.lower()

        # Try multi-word matches first (longer patterns take priority)
        sorted_verbs = sorted(VERB_TO_PRIMITIVE.keys(), key=len, reverse=True)
        for verb in sorted_verbs:
            if verb in sent_lower:
                return VERB_TO_PRIMITIVE[verb]

        # Fallback: check for common verb stems
        verb_stems = {
            "pick": "grip", "pour": "pour", "stir": "stir",
            "cut": "cut", "press": "press_button", "open": "open",
            "close": "close", "hold": "hold", "mov": "rearrange",
            "plac": "place_onto", "grab": "grip", "tak": "take_outside",
            "drink": "drink", "eat": "eat", "lift": "lift",
        }
        for stem, prim in verb_stems.items():
            if stem in sent_lower:
                return prim

        return "use"  # Generic fallback

    def _extract_object(self, sentence: str) -> str:
        """Extract the most likely object from a sentence."""
        sent_lower = sentence.lower()

        # Direct match against known objects (longest first)
        sorted_objects = sorted(self.object_list, key=len, reverse=True)
        for obj in sorted_objects:
            if obj.lower() in sent_lower:
                return obj

        # Pattern matching: "the X", "a X"
        pattern = (
            r'(?:the|a|an)\s+'
            r'(\w+(?:\s+\w+)?)'
            r'(?:\s+(?:from|to|on|onto|into|off|out|up|down))?'
        )
        match = re.search(pattern, sent_lower)
        if match:
            candidate = match.group(1).strip()
            # Check if candidate matches any known object
            for obj in sorted_objects:
                if candidate in obj.lower() or obj.lower() in candidate:
                    return obj

        return ""

    def _estimate_duration(self, sentence: str, primitive: str) -> int:
        """Estimate frame duration from primitive type and sentence keywords."""
        base_seconds = PRIMITIVE_DURATIONS.get(primitive, 3.0)
        base_frames = int(base_seconds * self.fps)

        sent_lower = sentence.lower()

        # Adjust based on keywords
        if any(w in sent_lower for w in ["quickly", "rapidly", "fast", "swift", "brief"]):
            base_frames = int(base_frames * 0.7)
        elif any(w in sent_lower for w in ["slowly", "carefully", "gently", "gradually"]):
            base_frames = int(base_frames * 1.3)
        elif any(w in sent_lower for w in ["thoroughly", "completely"]):
            base_frames = int(base_frames * 1.2)

        return max(self.min_segment_frames, min(self.max_segment_frames, base_frames))

    def _detect_hand_dominance(self, sentence: str) -> str:
        """Detect hand dominance from sentence."""
        sent_lower = sentence.lower()
        if any(kw in sent_lower for kw in BIMANUAL_KEYWORDS):
            return "bimanual"
        if any(kw in sent_lower for kw in LEFT_HAND_KEYWORDS):
            return "left"
        return "right"

    def _detect_temporal_type(self, sentence: str, idx: int, total: int) -> str:
        """Detect temporal relationship."""
        sent_lower = sentence.lower()
        if any(w in sent_lower for w in ["first", "initially", "to start", "begin"]):
            return "beginning"
        if any(w in sent_lower for w in ["finally", "lastly", "to finish", "at last"]):
            return "ending"
        if any(w in sent_lower for w in ["while", "as", "during", "simultaneously"]):
            return "simultaneous"
        if idx == 0:
            return "beginning"
        if idx == total - 1:
            return "ending"
        return "sequential"
