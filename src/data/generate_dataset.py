import os
import random

TARGET_DATASET_SIZES_MB = (10, 40, 160)

def generate_dummy_data(file_path: str, target_size_mb: int):
    """
    Generates a text file with synthetic words up to the requested size.
    """
    words = ["machine", "learning", "deep", "neural", "network", "transformer", 
             "attention", "gradient", "descent", "optimizer", "loss", "accuracy",
             "epoch", "batch", "tensor", "gpu", "cpu", "python", "pytorch", 
             "iteration", "gradient_clip", "activation", "relu", "sigmoid"]

    target_bytes = target_size_mb * 1024 * 1024
    bytes_written = 0

    print(f"Generating {target_size_mb} MB of synthetic data...")
    with open(file_path, "w", encoding="utf-8") as f:
        while bytes_written < target_bytes:
            # Generate a random sentence of 5 to 15 words
            sentence_length = random.randint(5, 15)
            sentence_words = random.choices(words, k=sentence_length)
            line = " ".join(sentence_words) + "\n"
            f.write(line)
            bytes_written += len(line.encode("utf-8"))
            
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"Data generation complete! Saved to {file_path} ({file_size_mb:.2f} MB)")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for size_mb in TARGET_DATASET_SIZES_MB:
        output_path = os.path.join(current_dir, f"dataset_{size_mb}mb.txt")
        generate_dummy_data(output_path, target_size_mb=size_mb)
