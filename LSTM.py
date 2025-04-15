import torch
import torch.nn as nn
import numpy as np
import logging
from config import DEVICE, FEATURE_SPEC, MODEL_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

class FeatureProcessor:
    """Handles feature processing and validation"""
    def __init__(self):
        self.feature_spec = FEATURE_SPEC
    
    def validate_features(self, features):
        """Validate features against the feature specification"""
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
    
    def process_features(self, features):
        """Convert features to tensor format"""
        self.validate_features(features)
        
        batch_size = len(features)
        seq_length = len(features[0])
        
        # Initialize tensor
        feature_tensor = torch.zeros(batch_size, seq_length, self.feature_spec["dimensions"])
        
        for b in range(batch_size):
            for t in range(seq_length):
                frame = features[b][t]
                
                if not isinstance(frame, dict):
                    raise TypeError("Each frame must be a dictionary")
                if "features" not in frame:
                    raise ValueError("Frame must contain features")
                
                # Process features
                feature_tensor[b, t] = torch.tensor(frame["features"])
        
        return feature_tensor

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            # Feature processing
            self.feature_processor = FeatureProcessor()
            
            # Feature processing layers
            self.feature_fc = nn.Sequential(
                nn.Linear(MODEL_CONFIG["input_size"], 256),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG["dropout"])
            )
            
            # Bidirectional LSTM
            self.lstm = nn.LSTM(
                input_size=256,
                hidden_size=MODEL_CONFIG["hidden_size"],
                num_layers=MODEL_CONFIG["num_layers"],
                batch_first=True,
                bidirectional=True,
                dropout=MODEL_CONFIG["dropout"] if MODEL_CONFIG["num_layers"] > 1 else 0
            )
            
            # Attention mechanism
            self.attention = Attention(MODEL_CONFIG["hidden_size"] * 2)  # *2 for bidirectional
            
            # Final classification
            self.fc = nn.Sequential(
                nn.Linear(MODEL_CONFIG["hidden_size"] * 2, 64),
                nn.ReLU(),
                nn.Dropout(MODEL_CONFIG["dropout"]),
                nn.Linear(64, 1)
            )
        except Exception as e:
            logging.error(f"Error initializing LSTM model: {str(e)}")
            raise
    
    def forward(self, features):
        try:
            # Process features
            feature_tensor = self.feature_processor.process_features(features)
            
            # Move tensor to device
            device = next(self.parameters()).device
            feature_tensor = feature_tensor.to(device)
            
            # Process features
            processed_features = self.feature_fc(feature_tensor)
            
            # LSTM processing
            lstm_out, _ = self.lstm(processed_features)
            
            # Attention mechanism
            context, attention_weights = self.attention(lstm_out)
            
            # Final classification
            output = self.fc(context)
            
            return torch.sigmoid(output), attention_weights
        except Exception as e:
            logging.error(f"Error in forward pass: {str(e)}")
            raise