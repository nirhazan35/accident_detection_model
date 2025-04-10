import torch
import torch.nn as nn
import numpy as np
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, Dataset
import os
from pathlib import Path
import datetime
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import random

# Configuration
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 10
LR_PATIENCE = 5
CLASS_WEIGHTS = [1.0, 3.0]  # Non-accident, Accident (assuming accidents are rarer)

# Check if CUDA is available
assert torch.cuda.is_available(), "CUDA-enabled GPU is required!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Initialize TensorBoard logger
log_dir = f"logs/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
writer = SummaryWriter(log_dir)

class FeatureDataset(Dataset):
    """Custom dataset for handling feature sequences"""
    def __init__(self, feature_files, labels=None, transform=None):
        self.feature_files = feature_files
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.feature_files)
    
    def __getitem__(self, idx):
        # Load feature sequence
        features = np.load(self.feature_files[idx], allow_pickle=True)
        
        # Process features into a fixed-size tensor
        processed_features = self.process_features(features)
        
        # Apply transforms if any
        if self.transform:
            processed_features = self.transform(processed_features)
        
        # Return features and label
        if self.labels is not None:
            return processed_features, torch.tensor(self.labels[idx], dtype=torch.float32)
        else:
            return processed_features
    
    def process_features(self, frame_features):
        """Process raw frame features into a fixed-size tensor"""
        processed_seq = []
        
        for frame in frame_features:
            # Extract features
            vehicle_count = frame["num_vehicles"]
            ped_count = frame["num_peds"]
            
            # Process motion vectors
            avg_motion = np.mean(frame["motion"], axis=0) if len(frame["motion"]) > 0 else [0, 0]
            
            # Process acceleration vectors
            avg_accel = np.mean(frame["acceleration"], axis=0) if len(frame["acceleration"]) > 0 else [0, 0]
            
            # Process proximities (minimum distance between objects)
            min_proximity = 1.0  # Default to max distance
            if frame["proximities"]:
                min_proximity = min([p["distance"] for p in frame["proximities"]], default=1.0)
            
            # Process edge distances (minimum distance to edge)
            min_edge_dist = np.min(frame["edge_distances"]) if len(frame["edge_distances"]) > 0 else 1.0
            
            # Get time of day
            time_of_day = frame.get("time_of_day", 0.5)  # Default to mid-day if not available
            
            # Check for trajectory anomalies
            has_anomaly = 1.0 if "trajectory_anomalies" in frame and frame["trajectory_anomalies"] else 0.0
            
            # Combine all features into a single vector
            frame_vector = [
                vehicle_count, 
                ped_count,
                avg_motion[0], avg_motion[1],
                avg_accel[0], avg_accel[1],
                min_proximity,
                min_edge_dist,
                time_of_day,
                has_anomaly
            ]
            
            processed_seq.append(frame_vector)
        
        return torch.tensor(processed_seq, dtype=torch.float32)

class TemporalAugmentation:
    """Apply temporal augmentations to sequences"""
    def __init__(self, p=0.5, max_shift=2):
        self.p = p
        self.max_shift = max_shift
    
    def __call__(self, x):
        if random.random() < self.p:
            # Random temporal shift (drop some frames, duplicate others)
            seq_len = x.shape[0]
            shift = random.randint(-self.max_shift, self.max_shift)
            
            if shift > 0:
                # Duplicate some frames at the beginning
                x = torch.cat([x[:shift].repeat(2, 1), x[shift:]], dim=0)
                x = x[:seq_len]  # Keep original length
            elif shift < 0:
                # Duplicate some frames at the end
                shift = abs(shift)
                x = torch.cat([x[:-shift], x[-shift:].repeat(2, 1)], dim=0)
                x = x[:seq_len]  # Keep original length
        
        return x

class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for accident detection"""
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x shape: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden*2]
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out).squeeze(-1)  # [batch, seq_len]
        attn_weights = torch.softmax(attn_weights, dim=1).unsqueeze(2)  # [batch, seq_len, 1]
        
        # Apply attention to LSTM outputs
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden*2]
        
        # Final prediction
        out = self.fc(context)
        return torch.sigmoid(out)

def load_dataset():
    """Load and prepare the dataset"""
    # Check for feature directory
    if not os.path.exists("features/train"):
        raise FileNotFoundError("Features directory not found. Please run feature_extractor.py first.")
    
    # Get all feature files
    feature_files = [os.path.join("features/train", f) for f in os.listdir("features/train") 
                     if f.startswith("seq_") and f.endswith(".npy")]
    
    if not feature_files:
        raise ValueError("No feature files found. Run feature_extractor.py first.")
    
    # Load metadata for labels
    if not os.path.exists("features/metadata.json"):
        raise FileNotFoundError("Metadata file not found.")
    
    with open("features/metadata.json") as f:
        metadata = json.load(f)
    
    # Create a mapping from video name to label
    video_to_label = {item["video_name"]: item["label"] for item in metadata}
    
    # Extract labels for each feature file
    labels = []
    valid_files = []
    
    for file_path in feature_files:
        filename = os.path.basename(file_path)
        parts = filename.split('_')
        
        if len(parts) < 3:
            continue
            
        # Try to find video name from filename
        video_stem = '_'.join(parts[1:-1])
        
        # Find matching video in metadata
        label = None
        for video_name in video_to_label:
            if video_name.startswith(video_stem):
                label = video_to_label[video_name]
                break
        
        # If can't find in metadata, try alternate approach
        if label is None:
            if "accident" in filename.lower() and "non" not in filename.lower():
                label = 1
            elif "non_accident" in filename.lower():
                label = 0
            else:
                continue  # Skip if can't determine label
        
        labels.append(label)
        valid_files.append(file_path)
    
    print(f"Loaded {len(valid_files)} valid feature sequences")
    print(f"Class distribution: {sum(labels)} positive, {len(labels) - sum(labels)} negative")
    
    return valid_files, labels

def train_model():
    """Train the accident detection model"""
    # Load dataset
    feature_files, labels = load_dataset()
    
    # Train/val/test split
    train_files, temp_files, train_labels, temp_labels = train_test_split(
        feature_files, labels, test_size=0.3, stratify=labels, random_state=42
    )
    
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.5, stratify=temp_labels, random_state=42
    )
    
    print(f"Train: {len(train_files)}, Validation: {len(val_files)}, Test: {len(test_files)}")
    
    # Create datasets with augmentation
    train_transform = TemporalAugmentation(p=0.5)
    train_dataset = FeatureDataset(train_files, train_labels, transform=train_transform)
    val_dataset = FeatureDataset(val_files, val_labels)
    test_dataset = FeatureDataset(test_files, test_labels)
    
    # Sample a batch to determine input size
    sample_batch, _ = train_dataset[0]
    input_size = sample_batch.shape[1]  # Number of features per frame
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=4)
    
    # Initialize model
    model = AttentionLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT_RATE
    ).to('cuda')
    
    # Calculate class weights if needed
    if sum(labels) > 0:
        pos_weight = len(labels) / sum(labels)
        class_weight = torch.tensor([1.0, pos_weight]).to('cuda')
    else:
        class_weight = torch.tensor(CLASS_WEIGHTS).to('cuda')
    
    print(f"Using class weights: {class_weight.tolist()}")
    
    # Define loss and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=LR_PATIENCE, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    print(f"Starting training for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds, train_targets = [], []
        
        for batch_X, batch_y in train_loader:
            # Move to GPU
            batch_X, batch_y = batch_X.to('cuda'), batch_y.unsqueeze(1).to('cuda')
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_X)
            
            # Calculate weighted loss
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            train_preds.extend(outputs.cpu().detach().numpy())
            train_targets.extend(batch_y.cpu().numpy())
        
        # Calculate training metrics
        train_loss = train_loss / len(train_loader)
        train_preds = np.array(train_preds) > 0.5
        train_targets = np.array(train_targets)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_targets, train_preds, average='binary'
        )
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                # Move to GPU
                batch_X, batch_y = batch_X.to('cuda'), batch_y.unsqueeze(1).to('cuda')
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Track metrics
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(batch_y.cpu().numpy())
        
        # Calculate validation metrics
        val_loss = val_loss / len(val_loader)
        val_probs = np.array(val_preds)
        val_preds = val_probs > 0.5
        val_targets = np.array(val_targets)
        
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='binary'
        )
        
        try:
            val_roc_auc = roc_auc_score(val_targets, val_probs)
        except:
            val_roc_auc = 0.0
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Log metrics
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"  Train - Loss: {train_loss:.4f}, F1: {train_f1:.4f}, Precision: {train_precision:.4f}, Recall: {train_recall:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, F1: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, AUC: {val_roc_auc:.4f}")
        
        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/val', val_f1, epoch)
        writer.add_scalar('Precision/val', val_precision, epoch)
        writer.add_scalar('Recall/val', val_recall, epoch)
        writer.add_scalar('ROC-AUC/val', val_roc_auc, epoch)
        
        # Check if this is the best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_auc': val_roc_auc,
                'input_size': input_size,
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
            }, f"models/accident_lstm_best_f1.pth")
            
            print(f"  New best model saved! (F1: {val_f1:.4f})")
        else:
            patience_counter += 1
            
        # Check for early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs!")
            break
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    model.eval()
    test_preds, test_targets = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to('cuda')
            outputs = model(batch_X)
            test_preds.extend(outputs.cpu().numpy())
            test_targets.extend(batch_y.numpy())
    
    # Calculate test metrics
    test_probs = np.array(test_preds)
    test_preds = test_probs > 0.5
    test_targets = np.array(test_targets).reshape(-1, 1)
    
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_targets, test_preds, average='binary'
    )
    
    test_roc_auc = roc_auc_score(test_targets, test_probs)
    
    # Print final results
    print(f"Test Results:")
    print(f"  F1: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  AUC: {test_roc_auc:.4f}")
    
    # Plot confusion matrix
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = ['Non-Accident', 'Accident']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')
    
    # Save the final model
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
        'hidden_size': HIDDEN_SIZE,
        'num_layers': NUM_LAYERS,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_auc': test_roc_auc
    }, f"models/accident_lstm_final_{timestamp}.pth")
    
    print(f"Final model saved to models/accident_lstm_final_{timestamp}.pth")
    
    # Save the feature normalization statistics for inference
    train_data = torch.cat([batch[0] for batch in train_dataset])
    feature_mean = train_data.mean(dim=(0, 1))
    feature_std = train_data.std(dim=(0, 1))
    
    np.savez(
        "models/feature_normalization.npz",
        mean=feature_mean.numpy(),
        std=feature_std.numpy()
    )
    
    return model

if __name__ == "__main__":
    train_model()