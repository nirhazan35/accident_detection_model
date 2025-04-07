import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os

# Configuration
INPUT_SIZE = 2 + 2  # num_vehicles + num_peds + (avg motion x, y)
HIDDEN_SIZE = 64
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 20

class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Use last timestep output
        return torch.sigmoid(out)

def load_features():
    with open("features/metadata.json") as f:
        metadata = json.load(f)
    
    X, y = [], []
    for item in metadata:
        for seq_idx in range(item["total_sequences"]):
            seq = np.load(f"features/seq_{item['video_name']}_{seq_idx}.npy")
            # Convert to tensor format: [SEQ_LENGTH, INPUT_SIZE]
            processed_seq = []
            for frame in seq:
                # Flatten features: [vehicles, peds, avg_dx, avg_dy]
                avg_motion = np.mean(frame["motion"], axis=0) if len(frame["motion"]) > 0 else [0, 0]
                processed_frame = [
                    frame["num_vehicles"],
                    frame["num_peds"],
                    avg_motion[0],
                    avg_motion[1]
                ]
                processed_seq.append(processed_frame)
            X.append(processed_seq)
            y.append(item["label"])
    
    return torch.FloatTensor(X), torch.FloatTensor(y)

if __name__ == "__main__":
    # Load data
    X, y = load_features()
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train, y_train.unsqueeze(1))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    val_dataset = TensorDataset(X_val, y_val.unsqueeze(1))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    model = SimpleLSTM()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
            print(f"Epoch {epoch+1}, Val Loss: {val_loss/len(val_loader):.4f}")
    
    # Save model
    torch.save(model.state_dict(), "models/accident_lstm.pth")