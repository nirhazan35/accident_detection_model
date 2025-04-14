import torch
import torch.nn as nn
import math
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FeatureAnalyzer:
    def __init__(self):
        self.backbone_importance = []
        self.neck_importance = []
        self.attention_weights = []
        
    def update(self, backbone_weights, neck_weights, attention_weights):
        self.backbone_importance.append(backbone_weights)
        self.neck_importance.append(neck_weights)
        self.attention_weights.append(attention_weights)
        
    def analyze(self):
        # Convert to numpy arrays
        backbone_importance = np.array(self.backbone_importance)
        neck_importance = np.array(self.neck_importance)
        attention_weights = np.array(self.attention_weights)
        
        # Calculate mean importance
        mean_backbone = np.mean(backbone_importance, axis=0)
        mean_neck = np.mean(neck_importance, axis=0)
        mean_attention = np.mean(attention_weights, axis=0)
        
        return {
            'backbone_importance': mean_backbone,
            'neck_importance': mean_neck,
            'attention_weights': mean_attention
        }
    
    def visualize(self, save_path=None):
        analysis = self.analyze()
        
        plt.figure(figsize=(15, 5))
        
        # Plot backbone feature importance
        plt.subplot(1, 3, 1)
        plt.bar(range(len(analysis['backbone_importance'])), analysis['backbone_importance'])
        plt.title('Backbone Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        # Plot neck feature importance
        plt.subplot(1, 3, 2)
        plt.bar(range(len(analysis['neck_importance'])), analysis['neck_importance'])
        plt.title('Neck Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        
        # Plot attention weights
        plt.subplot(1, 3, 3)
        plt.plot(analysis['attention_weights'])
        plt.title('Attention Weights Over Time')
        plt.xlabel('Time Step')
        plt.ylabel('Attention Weight')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.close()

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        if lstm_output is None:
            raise ValueError("lstm_output cannot be None")
        if not isinstance(lstm_output, torch.Tensor):
            raise TypeError("lstm_output must be a torch.Tensor")
        if len(lstm_output.shape) != 3:
            raise ValueError("lstm_output must be 3-dimensional (batch_size, seq_len, hidden_size)")

        try:
            # lstm_output shape: (batch_size, seq_len, hidden_size)
            attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
            # attention_weights shape: (batch_size, seq_len, 1)
            context = torch.sum(attention_weights * lstm_output, dim=1)
            # context shape: (batch_size, hidden_size)
            return context, attention_weights
        except Exception as e:
            logging.error(f"Error in attention forward pass: {str(e)}")
            raise

class LSTM(nn.Module):
    def __init__(self, backbone_features_size=None, neck_features_size=None, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        try:
            # Log input sizes
            logging.info(f"Initializing LSTM with backbone_features_size={backbone_features_size}, neck_features_size={neck_features_size}")
            
            # Backbone features processing with importance tracking
            self.backbone_fc = nn.Sequential(
                nn.Linear(backbone_features_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256)
            )
            
            # Neck features processing with importance tracking
            self.neck_fc = nn.Sequential(
                nn.Linear(neck_features_size, 512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256)
            )
            
            # Combined features size
            combined_size = 256 + 256  # 256 from backbone + 256 from neck
            
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=combined_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0
            )
            
            # Attention mechanism
            self.attention = Attention(hidden_size * 2)  # *2 for bidirectional
            
            # Final classification
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 1)
            )
            
            # Feature analyzer
            self.feature_analyzer = FeatureAnalyzer()
            
            # Log model structure
            logging.info(f"LSTM model initialized with combined_size={combined_size}, hidden_size={hidden_size}")
        except Exception as e:
            logging.error(f"Error initializing LSTM model: {str(e)}")
            raise
    
    def process_features(self, features):
        """Process raw features into model input format"""
        if features is None:
            raise ValueError("features cannot be None")
        if not isinstance(features, list):
            raise TypeError("features must be a list")
        if not features:
            raise ValueError("features list cannot be empty")
        if not isinstance(features[0], list):
            raise TypeError("features must be a list of sequences")
        if not features[0]:
            raise ValueError("sequences cannot be empty")

        try:
            batch_size = len(features)
            seq_length = len(features[0])
            
            # Get feature sizes from first frame
            first_frame = features[0][0]
            backbone_size = len(first_frame["backbone_features"])
            neck_size = len(first_frame["neck_features"])
            
            # Log feature sizes
            logging.info(f"Processing features with backbone_size={backbone_size}, neck_size={neck_size}")
            
            # Initialize tensors
            backbone_tensor = torch.zeros(batch_size, seq_length, backbone_size)
            neck_tensor = torch.zeros(batch_size, seq_length, neck_size)
            
            for b in range(batch_size):
                for t in range(seq_length):
                    frame = features[b][t]
                    
                    if not isinstance(frame, dict):
                        raise TypeError("Each frame must be a dictionary")
                    if "backbone_features" not in frame or "neck_features" not in frame:
                        raise ValueError("Frame must contain both backbone_features and neck_features")
                    
                    # Process backbone features
                    backbone_tensor[b, t] = torch.tensor(frame["backbone_features"])
                    
                    # Process neck features
                    neck_tensor[b, t] = torch.tensor(frame["neck_features"])
            
            return backbone_tensor, neck_tensor
        except Exception as e:
            logging.error(f"Error processing features: {str(e)}")
            raise
    
    def forward(self, features):
        try:
            # Process features
            backbone_features, neck_features = self.process_features(features)
            
            # Move tensors to device
            device = next(self.parameters()).device
            backbone_features = backbone_features.to(device)
            neck_features = neck_features.to(device)
            
            # Process backbone features and track importance
            backbone_out = self.backbone_fc(backbone_features)
            backbone_weights = torch.mean(torch.abs(self.backbone_fc[0].weight), dim=0)
            
            # Process neck features and track importance
            neck_out = self.neck_fc(neck_features)
            neck_weights = torch.mean(torch.abs(self.neck_fc[0].weight), dim=0)
            
            # Combine features
            combined = torch.cat([backbone_out, neck_out], dim=-1)
            
            # LSTM processing
            lstm_out, _ = self.lstm(combined)
            
            # Attention mechanism
            context, attention_weights = self.attention(lstm_out)
            
            # Update feature analyzer
            self.feature_analyzer.update(
                backbone_weights.cpu().detach().numpy(),
                neck_weights.cpu().detach().numpy(),
                attention_weights.squeeze().cpu().detach().numpy()
            )
            
            # Final classification
            output = self.fc(context)
            
            return torch.sigmoid(output), attention_weights
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            raise
    
    def analyze_features(self, save_path=None):
        """Analyze and visualize feature importance"""
        return self.feature_analyzer.analyze()
    
    def visualize_features(self, save_path=None):
        """Visualize feature importance"""
        self.feature_analyzer.visualize(save_path)