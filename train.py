import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from LSTM import LSTM
import os
from pathlib import Path
import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
print(f"Using device: {DEVICE.upper()}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss(reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

def custom_collate(batch):
    """Custom collate function to handle our feature format"""
    features, labels = zip(*batch)
    return list(features), torch.stack(labels)

class AccidentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_features():
    # Check if metadata file exists
    if not os.path.exists("features/metadata.json"):
        raise FileNotFoundError("Metadata file not found. Please run feature_extractor.py first.")
    
    with open("features/metadata.json") as f:
        metadata = json.load(f)
    
    features, labels = [], []
    
    # Check if there are any entries in metadata
    if not metadata:
        raise ValueError("Metadata is empty. Make sure feature_extractor.py ran successfully.")
    
    # Scan features directory for all available feature files
    feature_files = os.listdir("features/train")
    feature_files = [f for f in feature_files if f.startswith("seq_") and f.endswith(".npy")]
    
    print(f"Found {len(feature_files)} feature files.")
    
    for feature_file in feature_files:
        try:
            seq = np.load(f"features/train/{feature_file}", allow_pickle=True)
            features.append(seq)
            
            # Find corresponding label from metadata
            video_name = feature_file.split('_')[1]
            label = None
            for item in metadata:
                if item["video_name"].startswith(video_name):
                    label = item["label"]
                    break
            
            if label is None:
                # Try to infer from filename
                if "accident" in feature_file.lower():
                    label = 1
                else:
                    label = 0
            
            labels.append(label)
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
    
    if not features:
        raise ValueError("No valid feature files were loaded. Check your data directory structure.")
    
    print(f"Loaded {len(features)} sequences for training.")
    return features, torch.FloatTensor(labels)

def plot_metrics(train_losses, val_losses, precisions, recalls, thresholds):
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot precision-recall curve
    plt.subplot(1, 2, 2)
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load data
    features, labels = load_features()
    
    # Train/val split
    train_features, val_features, train_labels, val_labels = train_test_split(
        features, labels, test_size=0.2, shuffle=True, stratify=labels
    )
    
    # Create datasets
    train_dataset = AccidentDataset(train_features, train_labels)
    val_dataset = AccidentDataset(val_features, val_labels)
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)
    
    # Initialize model
    model = LSTM().to(DEVICE)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)  # Focus more on hard examples
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_val_recall = 0.0
    
    # Training loop
    for epoch in range(EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs, _ = model(batch_features)
            loss = criterion(outputs, batch_labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        all_outputs = []
        all_labels = []
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs, _ = model(batch_features)
                loss = criterion(outputs, batch_labels.unsqueeze(1))
                val_loss += loss.item()
                all_outputs.extend(outputs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_outputs)
        ap_score = average_precision_score(all_labels, all_outputs)
        
        # Find threshold that maximizes recall while keeping precision above 0.5
        best_threshold = 0.5
        best_recall = 0.0
        for p, r, t in zip(precision, recall, thresholds):
            if p >= 0.5 and r > best_recall:
                best_recall = r
                best_threshold = t
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"AP Score: {ap_score:.4f}, Best Recall: {best_recall:.4f} at threshold {best_threshold:.4f}")
        
        # Save best model based on recall
        if best_recall > best_val_recall:
            best_val_recall = best_recall
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/accident_lstm_{timestamp}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_recall': best_recall,
                'threshold': best_threshold
            }, model_path)
            print(f"Saved new best model with recall {best_recall:.4f}")
        
        # Update learning rate
        scheduler.step(best_recall)
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, precision, recall, thresholds)
    print("Training complete. Metrics saved to training_metrics.png")