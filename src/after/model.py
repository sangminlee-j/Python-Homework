import json
import time

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    learning_rate: float = 0.001
    epochs: int = 3
    batch_size: int = 1000
    top_k_words: int = 20

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs < 0:
            raise ValueError("epochs must be positive")
        if self.batch_size < 0:
            raise ValueError("batch_size must be positive")
        if self.top_k_words < 0:
            raise ValueError("top_k_words must be positive")

class NLPModel:
    def __init__(self, config=None):
        self.config = config or ModelConfig()
        self._weights = {}

    @property
    def weights(self):
        return dict(self._weights)

    def initialize_weights(self, words, initial_weight=0.5):
        for word in words:
            self._weights[word] = initial_weight

    def update_all_weights(self, delta):
        for word in self._weights:
            self._weights[word] += delta

    def get_weight(self, word, default=0.0):
        return self._weights.get(word, default)

    def save_weights(self, path: str):
        print(f"[{time.time()}] Saving weights to {path}...")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._weights, f)
            
    def load_weights(self, path: str):
        print(f"[{time.time()}] Loading weights from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            self._weights = json.load(f)
