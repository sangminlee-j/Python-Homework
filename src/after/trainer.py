from inferencer import timed

class Trainer:
    def __init__(self, model, data_handler):
        self.model = model
        self.data_handler = data_handler

    @timed
    def run_training(self):
        print("Starting training loop...")
        
        config = self.model.config
        epochs = config.epochs
        batch_size = config.batch_size
        lr = config.learning_rate
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in self.data_handler.iter_batches(batch_size):
                # Mock forward pass and update
                loss_step = sum(len(line) for line in batch) * lr
                total_loss += loss_step
                
                self.model.update_all_weights(0.001)
                    
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")
