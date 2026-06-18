import os
import time

class DataHandler:
    """
    WARNING: Tight Coupling and Inefficiency!
    This class takes the raw NLPModel and mutates its weights directly.
    It also eagerly loads data and uses O(N^2) list counting operations.
    """
    def __init__(self, data_path: str, model):
        self.data_path = data_path
        self.model = model  # Tightly coupled to the specific model implementation
        
        # Internal state that probably shouldn't be kept in memory all the time
        self.lines = []
        self.all_words = []
        self.word_counts = {}

    def load_data_and_build_vocab(self):
        start_time = time.time()
        print(f"[{start_time}] Loading data and building vocab...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        # BAD: Eager loading massive dataset into memory
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
            
        for line in self.lines:
            self.all_words.extend(line.strip().split())
            
        # BAD: O(N^2) list counting
        unique_words = list(set(self.all_words))
        for word in unique_words[:self.model.top_k_words]:
            self.word_counts[word] = self.all_words.count(word)
            
        # BAD: Directly mutating the model's internal state (Tight Coupling / Encapsulation violation)
        for word in self.word_counts:
            self.model.weights[word] = 0.5  # initial mock weight
            
        end_time = time.time()
        print(f"[{end_time}] Data loaded and vocab built in {end_time - start_time:.2f} seconds.")
