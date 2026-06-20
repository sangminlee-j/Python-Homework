import os
import sys
import time
from model import NLPModel
from inferencer import Inferencer

def main():
    print("==================================================")
    print("         Starting Inference Pipeline              ")
    print("==================================================")
    
    start_total_time = time.time()

    dataset_size = sys.argv[1] if len(sys.argv) > 1 else 40
    
    # Define paths
    weights_path = os.path.join(os.path.dirname(__file__), f"../data/model_weights_{int(dataset_size)}mb.json")
    
    if not os.path.exists(weights_path):
        print("Model weights not found! Please run train.py first.")
        return
        
    model = NLPModel()
    model.load_weights(weights_path)
    
    inferencer = Inferencer(model)
    
    # Run inference on some dummy sentences
    test_sentences = [
        "deep learning is a subset of machine learning",
        "gradient descent optimizes the loss function",
        "random sentence with no keywords"
    ]
    
    print("\nRunning inference...")
    for sentence in test_sentences:
        inferencer.infer(sentence)
        
    end_total_time = time.time()
    print("\n==================================================")
    print(f"Total Inference Script Time: {end_total_time - start_total_time:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    main()
