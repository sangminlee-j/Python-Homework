import os
import time

from collections import Counter

DATASET_FILES = {
    10: "dataset_10mb.txt",
    40: "dataset_40mb.txt",
    160: "dataset_160mb.txt",
}

def split_sentence(sentence: str):
    return sentence.strip().split()

def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[{end_time}] {func.__name__} took {end_time - start_time:.4f}s.")
        return result
    return wrapper

class DataHandler:
    
    def __init__(self, data_path: str, model, dataset_size: int = 40):
        self.data_path = self._resolve_data_path(data_path, dataset_size)
        self.model = model  # Tightly coupled to the specific model implementation
        
        # Internal state that probably shouldn't be kept in memory all the time
        self.lines = []
        self.all_words = []
        self.word_counts = {}

    def _resolve_data_path(self, data_path: str, dataset_size: int) -> str:
        dataset_size = int(dataset_size)
        if dataset_size not in DATASET_FILES:
            raise ValueError(f"dataset_size must be one of {sorted(DATASET_FILES)} MB")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        default_data_dir = os.path.join(project_root, "src", "data")

        if data_path and os.path.isdir(data_path):
            return os.path.join(data_path, DATASET_FILES[dataset_size])

        if data_path and os.path.exists(data_path):
            return data_path

        return os.path.join(default_data_dir, DATASET_FILES[dataset_size])

    def iter_words(self):

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                yield from split_sentence(line)

    def iter_batches(self, batch_size):
        batch = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                batch.append(line)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
        if batch:
            yield batch

    @timed
    def load_data_and_build_vocab(self):
        print("Loading data and building vocab...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")

        self.all_words.extend(self.iter_words())
        
        Count_words = Counter(self.all_words)
        unique_words = list(set(self.all_words))

        for word in unique_words[:self.model.config.top_k_words]:
            self.word_counts[word] = Count_words[word]
            
        self.model.initialize_weights(self.word_counts)
