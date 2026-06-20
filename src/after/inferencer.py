from data_handler import timed

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
