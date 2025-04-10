import torch
import torch.nn as nn
import math
import numpy as np

# Configuration
SCENE_FEATURES_SIZE = 1024  # Size of scene features
DETECTION_FEATURES_SIZE = 6    # num_vehicles, num_peds, collision_risk, avg_pos_x, avg_pos_y, avg_size
HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.2

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        # attention_weights shape: (batch_size, seq_len, 1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        # context shape: (batch_size, hidden_size)
        return context, attention_weights

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Scene features processing
        self.scene_fc = nn.Linear(SCENE_FEATURES_SIZE, 128)
        
        # Detection features processing
        self.detection_fc = nn.Linear(DETECTION_FEATURES_SIZE, 64)
        
        # Combined features size
        combined_size = 128 + 64
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=combined_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0
        )
        
        # Attention mechanism
        self.attention = Attention(HIDDEN_SIZE * 2)  # *2 for bidirectional
        
        # Final classification
        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_SIZE * 2, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )
    
    def process_features(self, features):
        """Process raw features into model input format"""
        batch_size = len(features)
        seq_length = len(features[0])
        
        # Initialize tensors
        scene_tensor = torch.zeros(batch_size, seq_length, SCENE_FEATURES_SIZE)
        detection_tensor = torch.zeros(batch_size, seq_length, DETECTION_FEATURES_SIZE)
        
        for b in range(batch_size):
            for t in range(seq_length):
                frame = features[b][t]
                
                # Process scene features
                scene_tensor[b, t] = torch.tensor(frame["scene_features"])
                
                # Process detection features
                num_vehicles = frame["num_vehicles"]
                num_peds = frame["num_peds"]
                collision_risk = frame["collision_risk"]
                
                # Calculate average position and size
                positions = np.array(frame["positions"])
                sizes = np.array(frame["sizes"])
                avg_pos = np.mean(positions, axis=0) if len(positions) > 0 else [0, 0]
                avg_size = np.mean(sizes, axis=0) if len(sizes) > 0 else [0, 0]
                
                detection_tensor[b, t] = torch.tensor([
                    num_vehicles,
                    num_peds,
                    collision_risk,
                    avg_pos[0],
                    avg_pos[1],
                    avg_size[0]
                ])
        
        return scene_tensor, detection_tensor
    
    def forward(self, features):
        # Process features
        scene_features, detection_features = self.process_features(features)
        
        # Process scene features
        scene_out = self.scene_fc(scene_features)
        
        # Process detection features
        detection_out = self.detection_fc(detection_features)
        
        # Combine features
        combined = torch.cat([scene_out, detection_out], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(combined)
        
        # Attention mechanism
        context, attention_weights = self.attention(lstm_out)
        
        # Final classification
        output = self.fc(context)
        
        return torch.sigmoid(output), attention_weights