from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Set, Optional


class Step(str, Enum):
    INIT = "INIT"
    EXTRACTING = "EXTRACTING"
    QUESTIONING = "QUESTIONING"
    MATCHING = "MATCHING"
    CONFIRMING = "CONFIRMING"
    REFINING = "REFINING"
    DONE = "DONE"


@dataclass
class ConversationState:
    messages: List[Dict[str, str]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    questions_asked: int = 0
    rejected_candidates: Set[int] = field(default_factory=set)
    step: Step = Step.INIT
    api_calls_used: int = 0
    refinement_cycles: int = 0
    rag_available: bool = True

    def add_message(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self._trim_to_last_n_turns(10)

    def _trim_to_last_n_turns(self, n: int) -> None:
        """Keep only the last n complete turn pairs (user + assistant) to guard token budget."""
        # A turn is one user message plus the following assistant message.
        # We always keep all messages up to 2*n items from the end.
        max_messages = n * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def update_features(self, new_features: Dict[str, Any]) -> None:
        """Merge new features, keeping higher-confidence values."""
        conf = new_features.get("confidence", {})
        existing_conf = self.features.get("confidence", {})

        for key in ("genre_hint", "mood_hint", "energy_level", "acoustic_preference"):
            new_val = new_features.get(key)
            new_c = conf.get(key, 0.0) if isinstance(conf, dict) else getattr(conf, key, 0.0)
            old_c = existing_conf.get(key, 0.0) if isinstance(existing_conf, dict) else getattr(existing_conf, key, 0.0)
            if new_val is not None and new_c >= old_c:
                self.features[key] = new_val
                if "confidence" not in self.features:
                    self.features["confidence"] = {}
                self.features["confidence"][key] = new_c

    def lowest_confidence_feature(self) -> Optional[str]:
        """Return the name of the feature with the lowest confidence that hasn't been filled."""
        conf = self.features.get("confidence", {})
        candidates = {
            "genre_hint": conf.get("genre_hint", 0.0) if isinstance(conf, dict) else getattr(conf, "genre_hint", 0.0),
            "mood_hint": conf.get("mood_hint", 0.0) if isinstance(conf, dict) else getattr(conf, "mood_hint", 0.0),
            "energy_level": conf.get("energy_level", 0.0) if isinstance(conf, dict) else getattr(conf, "energy_level", 0.0),
            "acoustic_preference": conf.get("acoustic_preference", 0.0) if isinstance(conf, dict) else getattr(conf, "acoustic_preference", 0.0),
        }
        # Exclude features already highly confident
        low = {k: v for k, v in candidates.items() if v < 0.8}
        if not low:
            return None
        return min(low, key=lambda k: low[k])

    def all_features_confident(self, threshold: float = 0.75) -> bool:
        conf = self.features.get("confidence", {})
        for key in ("genre_hint", "mood_hint", "energy_level", "acoustic_preference"):
            c = conf.get(key, 0.0) if isinstance(conf, dict) else getattr(conf, key, 0.0)
            if c < threshold:
                return False
        return True

    def to_prefs_dict(self) -> Dict[str, Any]:
        """Convert current features to the dict shape expected by recommender.score_song."""
        return {
            "genre": self.features.get("genre_hint", ""),
            "mood": self.features.get("mood_hint", ""),
            "energy": self.features.get("energy_level", 0.5),
            "likes_acoustic": self.features.get("acoustic_preference", False),
        }
