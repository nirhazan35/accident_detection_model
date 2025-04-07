import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os
from pathlib import Path
import datetime

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
    # Check if metadata file exists
    if not os.path.exists("features/metadata.json"):
        raise FileNotFoundError("Metadata file not found. Please run feature_extractor.py first.")
    
    with open("features/metadata.json") as f:
        metadata = json.load(f)
    
    X, y = [], []
    
    # Check if there are any entries in metadata
    if not metadata:
        raise ValueError("Metadata is empty. Make sure feature_extractor.py ran successfully.")
    
    # Scan features directory for all available feature files
    feature_files = os.listdir("features/train")
    feature_files = [f for f in feature_files if f.startswith("seq_") and f.endswith(".npy")]
    
    print(f"Found {len(feature_files)} feature files.")
    
    for feature_file in feature_files:
        # Extract sequence information from filename
        parts = feature_file.split('_')
        
        # Find corresponding metadata
        video_name = parts[1]  # This assumes filenames are like "seq_VIDEONAME_INDEX.npy"
        # For filenames with multiple underscores, reconstruct the video name
        if len(parts) > 3:
            video_stem = "_".join(parts[1:-1])  # Join all parts except first and last
        else:
            video_stem = parts[1]
            
        # Try to find matching metadata
        metadata_item = None
        for item in metadata:
            if item["video_name"].startswith(video_stem):
                metadata_item = item
                break
        
        if metadata_item is None:
            # Try alternate approach - extract from filename directly
            label = None
            if "accident" in feature_file.lower():
                label = 1
            elif "non_accident" in feature_file.lower():
                label = 0
            else:
                continue  # Skip if can't determine label
        else:
            label = metadata_item["label"]
            
        # Load the feature file
        try:
            seq = np.load(f"features/train/{feature_file}", allow_pickle=True)
            
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
            y.append(label)
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
    
    if not X:
        raise ValueError("No valid feature files were loaded. Check your data directory structure.")
    
    print(f"Loaded {len(X)} sequences for training.")
    return torch.FloatTensor(X), torch.FloatTensor(y)

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
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
        train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
                
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Save model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/accident_lstm_{timestamp}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Training complete. Model saved to {model_path}")