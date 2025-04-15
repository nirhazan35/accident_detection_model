import torch

# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Feature extraction configuration
SEQ_LENGTH = 8          # Number of frames per sequence for faster processing
OVERLAP = 4             # Overlap between consecutive sequences
FEATURES_DIR = "features"  # Base features directory (train/val will be subdirectories)
DATA_DIR = "data"       # Base data directory (train/val will be subdirectories)

# Real-time configuration
FRAME_PROCESSING_FPS = 15  # Target FPS for processing
DETECTION_THRESHOLD = 0.5  # Balanced threshold based on precision-recall trade-off
ALERT_COOLDOWN = 60        # Seconds between alerts to avoid spam

# Backbone layer configuration
BACKBONE_LAYERS = {
    "use_layers": [2, 6, 10],  # Key layers for feature extraction
    "layer_output_sizes": [256, 512, 1024],  # Output channels from selected layers
    "spatial_sizes": [20, 10, 5],  # Spatial dimensions for feature maps
    "total_features": 166400  # Total flattened feature size
}

# Feature specifications
FEATURE_SPEC = {
    "name": "yolo11m_backbone",  # Name of the feature extractor
    "version": "1.0",         # Version of the feature format
    "dimensions": 768,        # Reduced dimensions while maintaining information
    "normalized": True,       # Whether features are normalized
    "format": {              # Structure of the feature data
        "frame": {
            "features": "list[float]"  # The feature vector
        },
        "sequence": {
            "frames": "list[frame]",
            "label": "int",
            "source_video": "str"
        }
    }
}

# Model configuration
MODEL_CONFIG = {
    "input_size": FEATURE_SPEC["dimensions"],
    "hidden_size": 256,
    "num_layers": 2,
    "dropout": 0.2
}

# Training configuration
TRAINING_CONFIG = {
    "precision_weight": 0.6,  # Higher values prioritize precision over recall (0.5 is balanced)
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "early_stopping_patience": 10
}

# Alert configuration
ALERT_API_ENDPOINT = "http://your-website.com/api/alerts"  # Replace with your actual API endpoint 