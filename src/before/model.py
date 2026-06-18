import json
import time

class NLPModel:
    """
    WARNING: Bad OOP Design!
    This class mixes configurations (epochs, batch_size) with runtime state (weights).
    It also exposes its internal data structures to be mutated by other classes.
    """
    def __init__(self):
        # Configurations hardcoded
        self.learning_rate = 0.001
        self.epochs = 3
        self.batch_size = 1000
        self.top_k_words = 20
        
        # Runtime State
        self.weights = {}

    def save_weights(self, path: str):
        print(f"[{time.time()}] Saving weights to {path}...")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.weights, f)
            
    def load_weights(self, path: str):
        print(f"[{time.time()}] Loading weights from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            self.weights = json.load(f)
