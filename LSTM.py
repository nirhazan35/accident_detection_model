import torch
import torch.nn as nn

# Configuration
INPUT_SIZE = 2 + 2  # num_vehicles + num_peds + (avg motion x, y)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last timestep output
        return torch.sigmoid(out)