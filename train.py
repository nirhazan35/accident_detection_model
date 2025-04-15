import torch
import torch.nn as nn
import numpy as np
import json
import os
from torch.utils.data import DataLoader, Dataset
from LSTM import LSTM
from pathlib import Path
import datetime
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt

# Configuration
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
# Target precision-recall balance (higher values prioritize precision)
PRECISION_WEIGHT = 0.6  # Balance between precision (1.0) and recall (0.0)
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
    return list(features), torch.stack(labels).to(DEVICE)

class AccidentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def load_features_from_split(split):
    """Load features from a specific split (train or val)"""
    split_dir = f"features/{split}"
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Features directory {split_dir} not found. Run feature_extractor.py first.")
    
    feature_files = os.listdir(split_dir)
    feature_files = [f for f in feature_files if f.startswith("seq_") and f.endswith(".npy")]
    
    print(f"Found {len(feature_files)} feature files in {split} split.")
    
    features, labels = [], []
    
    for feature_file in feature_files:
        try:
            # Load the sequence
            seq_data = np.load(f"{split_dir}/{feature_file}", allow_pickle=True).item()
            
            # Ensure proper format
            if "frames" in seq_data:
                frames_list = seq_data["frames"]
                if isinstance(frames_list, list) and all(isinstance(f, dict) and "features" in f for f in frames_list):
                    features.append(frames_list)
                    
                    # Get label 
                    if "label" in seq_data:
                        label = seq_data["label"]
                    else:
                        # Infer from filename
                        if "accident" in feature_file.lower():
                            label = 1
                        else:
                            label = 0
                    
                    labels.append(label)
                else:
                    print(f"Warning: File {feature_file} has invalid frame format. Skipping.")
            else:
                print(f"Warning: File {feature_file} has no 'frames' key. Skipping.")
        except Exception as e:
            print(f"Error loading {feature_file}: {e}")
    
    if not features:
        raise ValueError(f"No valid feature files were loaded from {split} split.")
    
    print(f"Loaded {len(features)} sequences for {split} split.")
    
    # Count class distribution
    total_positives = sum(labels)
    total_negatives = len(labels) - total_positives
    print(f"Class distribution in {split}: {total_positives} accident samples, {total_negatives} non-accident samples")
    
    return features, torch.FloatTensor(labels)

def calculate_f_beta_score(precision, recall, beta=1.0):
    """Calculate F-beta score from precision and recall"""
    beta_squared = beta ** 2
    if precision + recall == 0:
        return 0
    return (1 + beta_squared) * (precision * recall) / ((beta_squared * precision) + recall)

def find_optimal_threshold(precisions, recalls, thresholds, precision_weight=0.5):
    """Find threshold that optimizes weighted balance between precision and recall"""
    best_score = 0
    best_threshold = 0.5
    best_precision = 0
    best_recall = 0
    
    # Calculate F-beta score with beta that gives precision the desired weight
    # beta = sqrt((1-precision_weight)/precision_weight)
    beta = (1 - precision_weight) / precision_weight
    
    for p, r, t in zip(precisions, recalls, thresholds):
        # Skip if precision or recall is too low
        if p < 0.3 or r < 0.3:
            continue
        
        # F-beta score
        score = calculate_f_beta_score(p, r, beta)
        
        if score > best_score:
            best_score = score
            best_threshold = t
            best_precision = p
            best_recall = r
    
    return best_threshold, best_precision, best_recall, best_score

def plot_metrics(train_losses, val_losses, precisions, recalls, thresholds, confusion_mat):
    fig = plt.figure(figsize=(20, 8))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot precision-recall curve
    plt.subplot(1, 3, 2)
    plt.plot(recalls, precisions)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    
    # Plot confusion matrix
    plt.subplot(1, 3, 3)
    plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Non-Accident', 'Accident'])
    plt.yticks(tick_marks, ['Non-Accident', 'Accident'])
    
    # Add text annotations
    thresh = confusion_mat.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(confusion_mat[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if confusion_mat[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Load data directly from train and val splits
    train_features, train_labels = load_features_from_split("train")
    val_features, val_labels = load_features_from_split("val")
    
    # Create datasets
    train_dataset = AccidentDataset(train_features, train_labels)
    val_dataset = AccidentDataset(val_features, val_labels)
    
    # Check for severe class imbalance
    train_positive = train_labels.sum().item()
    train_negative = len(train_labels) - train_positive
    class_ratio = max(train_positive, train_negative) / min(train_positive, train_negative)
    
    # Apply class weights if severe imbalance exists
    if class_ratio > 1.5:
        print(f"Detected class imbalance ({class_ratio:.2f}:1), applying class weights")
        pos_weight = train_negative / max(train_positive, 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]).to(DEVICE))
    else:
        # Use focal loss for balanced datasets
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Create DataLoaders with custom collate function
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, collate_fn=custom_collate)
    
    # Initialize model and move to device
    model = LSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training metrics
    train_losses = []
    val_losses = []
    best_f_score = 0.0
    no_improve_epochs = 0
    best_confusion_matrix = np.zeros((2, 2))
    
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
        
        # Find threshold that balances precision and recall according to desired weight
        best_threshold, best_precision, best_recall, f_score = find_optimal_threshold(
            precision, recall, thresholds, PRECISION_WEIGHT
        )
        
        # Calculate confusion matrix at the best threshold
        predicted_labels = (np.array(all_outputs) >= best_threshold).astype(int)
        conf_matrix = confusion_matrix(all_labels, predicted_labels, labels=[0, 1])
        
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"AP Score: {ap_score:.4f}, F-Score: {f_score:.4f}")
        print(f"Precision: {best_precision:.4f}, Recall: {best_recall:.4f} at threshold {best_threshold:.4f}")
        
        # Save model if F-score improves
        if f_score > best_f_score:
            best_f_score = f_score
            best_confusion_matrix = conf_matrix
            no_improve_epochs = 0
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/accident_lstm_{timestamp}.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'threshold': best_threshold,
                'precision': best_precision,
                'recall': best_recall,
                'f_score': f_score
            }, model_path)
            print(f"Saved new best model with F-score {f_score:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
        
        # Early stopping
        if no_improve_epochs >= 10:
            print(f"No improvement for {no_improve_epochs} epochs. Stopping training.")
            break
        
        # Update learning rate
        scheduler.step(f_score)
    
    # Plot metrics
    plot_metrics(train_losses, val_losses, precision, recall, thresholds, best_confusion_matrix)
    print("Training complete. Metrics saved to training_metrics.png")