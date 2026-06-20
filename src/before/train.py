import os
import sys
import time
from model import NLPModel
from data_handler import DataHandler
from trainer import Trainer

def main():
    print("==================================================")
    print("         Starting Training Pipeline               ")
    print("==================================================")
    
    start_total_time = time.time()
    
    dataset_size = sys.argv[1] if len(sys.argv) > 1 else 40

    # Define paths
    data_path = os.path.join(os.path.dirname(__file__), "../data")
    weights_path = os.path.join(os.path.dirname(__file__), f"../data/model_weights_{int(dataset_size)}mb.json")
    
    # Tightly coupled instantiation
    model = NLPModel()
    data_handler = DataHandler(data_path, model, dataset_size=dataset_size)
    trainer = Trainer(model, data_handler)
    
    # Execute the messy pipeline
    data_handler.load_data_and_build_vocab()
    trainer.run_training()
    model.save_weights(weights_path)
    
    end_total_time = time.time()
    print("==================================================")
    print(f"Total Training Script Time: {end_total_time - start_total_time:.2f} seconds.")
    print("==================================================")

if __name__ == "__main__":
    main()
