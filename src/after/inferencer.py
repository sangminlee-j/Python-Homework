import time

def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"[{end_time}] {func.__name__} took {end_time - start_time:.4f}s.")
        return result
    return wrapper

def split_sentence(sentence: str):
    return sentence.strip().split()

class Inferencer:

    def __init__(self, model):
        self.model = model

    @timed
    def infer(self, sentence: str) -> float:
        """
        Mock inference method.
        """
        score = 0.0
        
        words = split_sentence(sentence)
        
        for word in words:
            score += self.model.get_weight(word)
                
        print(f"Inference for '{sentence[:15]}...' Score: {score:.2f}")
        return score
