import torch
import numpy as np

class AccidentDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = SimpleLSTM()  # Same as in train.py
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.threshold = threshold
    
    def predict(self, sequence):
        with torch.no_grad():
            output = self.model(sequence)
        return (output > self.threshold).float()
    
    def adjust_threshold(self, new_threshold):
        self.threshold = new_threshold

# Example usage:
# detector = AccidentDetector("models/accident_lstm.pth", threshold=0.3)
# prediction = detector.predict(test_sequence)