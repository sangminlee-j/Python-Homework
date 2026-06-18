import time

class Trainer:
    """
    WARNING: Poor abstraction and tightly coupled!
    This class directly reaches into the internal lists of the data_handler 
    and directly mutates the model's weight dictionary. 
    It also scatters time.time() calls everywhere instead of using a decorator.
    """
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler

    def run_training(self):
        print(f"[{time.time()}] Starting training loop...")
        start_train = time.time()
        
        # BAD: Accessing config from the model
        epochs = self.model.epochs
        batch_size = self.model.batch_size
        lr = self.model.learning_rate
        
        for epoch in range(epochs):
            epoch_start = time.time()
            total_loss = 0.0
            
            # BAD: Directly reaching into data_handler's internal eager-loaded list
            # Violates Encapsulation. Should be using an Iterator/Generator interface.
            lines_count = len(self.data_handler.lines)
            
            for i in range(0, lines_count, batch_size):
                batch = self.data_handler.lines[i:i+batch_size]
                
                # Mock forward pass and update
                loss_step = sum(len(line) for line in batch) * lr
                total_loss += loss_step
                
                # BAD: Directly mutating model state
                for word in self.model.weights:
                    self.model.weights[word] += 0.001
                    
            epoch_end = time.time()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Took: {epoch_end - epoch_start:.2f}s")
            
        end_train = time.time()
        print(f"[{end_train}] Training finished in {end_train - start_train:.2f} seconds.")
