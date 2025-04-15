import torch
import cv2
import argparse
import numpy as np
from feature_extractor import YOLOFeatureExtractor
from config import DEVICE, BACKBONE_LAYERS, FEATURE_SPEC

def check_feature_dimensions(video_path):
    """Check the dimensions of features extracted from a sample frame"""
    print(f"Checking feature dimensions using video: {video_path}")
    
    # Initialize feature extractor
    extractor = YOLOFeatureExtractor()
    
    # Open video and get first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video")
        return
    
    cap.release()
    
    # Process frame
    print("Processing frame...")
    processed_frame = extractor.preprocess_frame(frame)
    print(f"Preprocessed frame shape: {processed_frame.shape}")
    
    # Extract features without using feature reducer
    with torch.no_grad():
        # Get features from each selected layer
        layer_features = []
        x = processed_frame
        
        print("\nLayer-by-layer analysis:")
        print("-" * 50)
        
        # Process through backbone layers sequentially
        for i, layer in enumerate(extractor.backbone):
            # Apply layer
            x = layer(x)
            print(f"Layer {i} output shape: {x.shape}")
            
            if i in BACKBONE_LAYERS["use_layers"]:
                # Calculate total elements
                total_elements = x.shape[1] * x.shape[2] * x.shape[3]
                
                # Suggested spatial dimensions
                suggested_size = int(np.sqrt(100 * x.shape[1] / FEATURE_SPEC["dimensions"]))
                
                print(f"  [SELECTED] Layer {i}:")
                print(f"    - Output channels: {x.shape[1]}")
                print(f"    - Spatial dimensions: {x.shape[2]}x{x.shape[3]}")
                print(f"    - Total elements when flattened: {total_elements}")
                print(f"    - Suggested spatial size for config: {suggested_size}")
                
                # Resize to target spatial size if specified
                spatial_idx = BACKBONE_LAYERS["use_layers"].index(i)
                target_size = BACKBONE_LAYERS["spatial_sizes"][spatial_idx]
                if x.shape[-1] != target_size:
                    orig_shape = x.shape
                    x_resized = torch.nn.functional.adaptive_avg_pool2d(x, (target_size, target_size))
                    print(f"    - Resized from {orig_shape[2]}x{orig_shape[3]} to {target_size}x{target_size}")
                    print(f"    - New total elements: {x_resized.shape[1] * target_size * target_size}")
                    # Flatten for feature extraction
                    flattened = x_resized.view(x_resized.size(0), -1)
                else:
                    # Flatten for feature extraction
                    flattened = x.view(x.size(0), -1)
                
                print(f"    - Flattened shape: {flattened.shape}")
                layer_features.append(flattened)
        
        # Concatenate features
        combined_features = torch.cat(layer_features, dim=1)
        
        print("\nSummary:")
        print("-" * 50)
        print(f"Combined feature shape: {combined_features.shape}")
        print(f"Expected feature size in config: {BACKBONE_LAYERS['total_features']}")
        
        if combined_features.shape[1] != BACKBONE_LAYERS["total_features"]:
            print(f"WARNING: Feature size mismatch!")
            print(f"  - Actual size: {combined_features.shape[1]}")
            print(f"  - Expected size: {BACKBONE_LAYERS['total_features']}")
            print("\nRecommended update for config.py:")
            print(f'  "total_features": {combined_features.shape[1]}')
        else:
            print("Feature dimensions match the configuration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check feature dimensions from YOLO model")
    parser.add_argument("--video", type=str, required=True, help="Path to a video file for testing")
    
    args = parser.parse_args()
    check_feature_dimensions(args.video)
