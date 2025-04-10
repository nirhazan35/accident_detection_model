import torch
import torch.nn as nn
import math

class LSTM(nn.Module):
    """
    Enhanced LSTM model for accident detection that processes rich feature sets
    including object detection data, motion patterns, and collision risks.
    """
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Feature embedding layer
        self.feature_embedding = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM for temporal pattern recognition
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism for focusing on important frames
        self.attention = nn.Sequential(
            nn.Linear(hidden_size*2, 64),  # *2 because bidirectional
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        # Output classification layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input features [batch, seq_length, features]
            
        Returns:
            Accident probability for each sequence
        """
        batch_size, seq_len, _ = x.shape
        
        # Embed features
        x = self.feature_embedding(x)  # [batch, seq_len, hidden_size]
        
        # Process through LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_size*2]
        
        # Apply attention mechanism
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch, seq_len, 1]
        
        # Weighted sum of LSTM outputs based on attention
        context = torch.sum(lstm_out * attn_weights, dim=1)  # [batch, hidden_size*2]
        
        # Final prediction
        output = self.classifier(context)
        return torch.sigmoid(output)

class MultiStreamLSTM(nn.Module):
    """
    Multi-stream LSTM that processes different feature types separately
    before fusing them for final prediction.
    """
    def __init__(self, 
                 motion_size=4,      # Motion/velocity features
                 proximity_size=2,   # Object proximity features
                 context_size=4,     # Context features (time, scene complexity)
                 hidden_size=128, 
                 num_layers=2, 
                 dropout=0.3):
        super().__init__()
        
        # Separate processing streams for different feature types
        self.motion_lstm = nn.LSTM(
            input_size=motion_size,
            hidden_size=hidden_size//2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.proximity_lstm = nn.LSTM(
            input_size=proximity_size,
            hidden_size=hidden_size//2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.context_embedding = nn.Linear(context_size, hidden_size)
        
        # Attention mechanisms for each stream
        self.motion_attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.proximity_attention = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size*2 + hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output classification layer
        self.classifier = nn.Linear(hidden_size, 1)
    
    def apply_attention(self, lstm_out, attention_layer):
        """Apply attention mechanism to LSTM outputs"""
        attn_weights = attention_layer(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return context
    
    def forward(self, x_motion, x_proximity, x_context):
        """
        Forward pass with multiple input streams
        
        Args:
            x_motion: Motion features [batch, seq_len, motion_features]
            x_proximity: Proximity features [batch, seq_len, proximity_features]
            x_context: Context features [batch, seq_len, context_features]
            
        Returns:
            Accident probability
        """
        # Process motion stream
        motion_out, _ = self.motion_lstm(x_motion)
        motion_context = self.apply_attention(motion_out, self.motion_attention)
        
        # Process proximity stream
        proximity_out, _ = self.proximity_lstm(x_proximity)
        proximity_context = self.apply_attention(proximity_out, self.proximity_attention)
        
        # Process context features (take the last frame's context)
        context_emb = self.context_embedding(x_context[:, -1, :])
        
        # Concatenate all streams
        combined = torch.cat([motion_context, proximity_context, context_emb], dim=1)
        
        # Fuse features
        fused = self.fusion(combined)
        
        # Final prediction
        output = self.classifier(fused)
        return torch.sigmoid(output)

# Include the TransformerAccidentDetector from paste-2.txt for completeness
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """Add positional encoding to input tensor"""
        x = x + self.pe[:x.size(1), :].unsqueeze(0)
        return self.dropout(x)

class TransformerAccidentDetector(nn.Module):
    """
    Transformer-based architecture for accident detection with both
    detection features and backbone features support
    """
    def __init__(self, 
                 detection_size=10, 
                 backbone_size=512,
                 d_model=128, 
                 nhead=4, 
                 num_layers=2, 
                 dropout=0.3,
                 use_backbone=True):
        super().__init__()
        
        self.use_backbone = use_backbone
        
        # Feature embedding layers
        self.detection_embedding = nn.Linear(detection_size, d_model // 2)
        
        if use_backbone:
            self.backbone_embedding = nn.Sequential(
                nn.Linear(backbone_size, d_model // 2),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            input_size = d_model  # Combined size
        else:
            input_size = d_model // 2  # Only detection features
            
        # Positional encoding for transformer
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # Output classification layers
        self.classifier = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        # Risk-level predictor
        self.risk_predictor = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 5)  # Predict risk level from 0-4
        )
        
    def forward(self, x_detection, x_backbone=None):
        """Forward pass through the transformer model"""
        # Process detection features
        detection_emb = self.detection_embedding(x_detection)
        
        # Process and fuse backbone features if available
        if self.use_backbone and x_backbone is not None:
            backbone_emb = self.backbone_embedding(x_backbone)
            combined_features = torch.cat([detection_emb, backbone_emb], dim=2)
        else:
            combined_features = detection_emb
            
        # Add positional encoding
        src = self.pos_encoder(combined_features)
        
        # Pass through transformer
        transformer_out = self.transformer_encoder(src)  # [batch, seq_len, d_model]
        
        # Global pooling (use mean of sequence outputs)
        pooled = transformer_out.mean(dim=1)
        
        # Generate outputs
        accident_prob = torch.sigmoid(self.classifier(pooled))
        risk_level = self.risk_predictor(pooled)
        
        return accident_prob, risk_level