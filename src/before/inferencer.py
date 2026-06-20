import time

class Inferencer:
    """
    WARNING: DRY Principle Violation!
    This class duplicates preprocessing logic (sentence stripping/splitting) 
    instead of sharing a common preprocessing pipeline.
    """
    def __init__(self, model):
        self.model = model

    def infer(self, sentence: str) -> float:
        """
        Mock inference method.
        """
        start_time = time.time()
        score = 0.0
        
        # BAD: Re-implementing the same preprocessing logic used in data_handler
        words = sentence.strip().split()
        
        for word in words:
            # Reaches directly into the model's dictionary
            if word in self.model.weights:
                score += self.model.weights[word]
                
        end_time = time.time()
        print(f"[{end_time}] Inference for '{sentence[:15]}...' took {end_time - start_time:.4f}s. Score: {score:.2f}")
        return score
