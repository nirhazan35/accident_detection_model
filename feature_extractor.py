import os
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
SEQ_LENGTH = 16          # Number of frames per sequence
OVERLAP = 8              # Overlap between sequences
CLASSES = [0, 1, 2, 3, 5, 7]  # Person, bicycle, car, motorcycle, bus, truck
FEATURES_DIR = "features/train"
DATA_DIR = "data"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE.upper()}")

def extract_features(video_path, label):
    # Convert string path to Path object
    video_path = Path(video_path)
    
    # Initialize YOLO model
    model = YOLO("yolo11m.pt").to(DEVICE)
    
    # Get the model's backbone and neck
    backbone = model.model.model[0]  # First module is the backbone
    neck = model.model.model[1]      # Second module is the neck
    
    # Log model structure
    logging.info(f"Backbone layers: {len(backbone)}")
    logging.info(f"Neck layers: {len(neck)}")
    
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return None
    
    features = []
    
    # Initialize progress bar for frames
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit='frame')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 640))  # YOLO default input size
        frame = frame.transpose(2, 0, 1)  # HWC to CHW
        frame = np.ascontiguousarray(frame)
        frame = torch.from_numpy(frame).to(DEVICE)
        frame = frame.float() / 255.0
        frame = frame.unsqueeze(0)  # Add batch dimension
        
        # Extract features using YOLO's forward pass
        with torch.no_grad():
            # Get backbone features
            backbone_outputs = backbone(frame)
            
            # Get neck features
            neck_outputs = neck(backbone_outputs)
            
            # Process backbone features
            backbone_features = []
            for i, feat in enumerate(backbone_outputs):
                if isinstance(feat, torch.Tensor):
                    # Flatten and normalize each backbone feature
                    flat_feat = feat.view(feat.size(0), -1)
                    norm_feat = (flat_feat - flat_feat.mean()) / (flat_feat.std() + 1e-6)
                    backbone_features.append(norm_feat)
                    if i == 0:  # Log dimensions of first feature map
                        logging.info(f"Backbone feature {i} shape: {feat.shape}, flattened: {flat_feat.shape}")
            
            # Process neck features
            neck_features = []
            for i, feat in enumerate(neck_outputs):
                if isinstance(feat, torch.Tensor):
                    # Flatten and normalize each neck feature
                    flat_feat = feat.view(feat.size(0), -1)
                    norm_feat = (flat_feat - flat_feat.mean()) / (flat_feat.std() + 1e-6)
                    neck_features.append(norm_feat)
                    if i == 0:  # Log dimensions of first feature map
                        logging.info(f"Neck feature {i} shape: {feat.shape}, flattened: {flat_feat.shape}")
            
            # Concatenate all features
            backbone_features = torch.cat(backbone_features, dim=1)
            neck_features = torch.cat(neck_features, dim=1)
            
            # Log final feature dimensions
            logging.info(f"Final backbone features shape: {backbone_features.shape}")
            logging.info(f"Final neck features shape: {neck_features.shape}")
            
            # Create frame feature
            frame_feature = {
                "backbone_features": backbone_features.cpu().numpy()[0].tolist(),
                "neck_features": neck_features.cpu().numpy()[0].tolist()
            }
            
            features.append(frame_feature)
        
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    if not features:
        logging.error(f"No frames extracted from: {video_path}")
        return None
    
    # Split into overlapping sequences
    sequences = []
    for i in range(0, len(features) - SEQ_LENGTH + 1, OVERLAP):
        seq = features[i:i+SEQ_LENGTH]
        sequences.append({
            "features": seq,
            "label": label,
            "source_video": video_path.name
        })
    
    # Save features with progress bar
    save_pbar = tqdm(total=len(sequences), desc=f"Saving sequences for {video_path.name}", unit='seq')
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for idx, seq in enumerate(sequences):
        np.save(f"{FEATURES_DIR}/seq_{video_path.stem}_{idx}.npy", seq["features"])
        save_pbar.update(1)
    save_pbar.close()
    
    metadata = {
        "video_name": video_path.name,
        "total_sequences": len(sequences),
        "label": label
    }
    return metadata

if __name__ == "__main__":
    all_metadata = []
    
    # Get list of video files with proper filtering
    def get_video_files(folder):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        return [
            f for f in os.listdir(folder)
            if not f.startswith('.') and 
            any(f.lower().endswith(ext) for ext in video_extensions)
        ]
    
    # Get all video files
    accident_videos = get_video_files("data/accidents")
    non_accident_videos = get_video_files("data/non_accidents")
    total_videos = len(accident_videos) + len(non_accident_videos)
    
    # Main progress bar for all videos
    main_pbar = tqdm(total=total_videos, desc="Processing all videos", unit='video')
    
    # Process accident videos (label=1)
    for video_file in accident_videos:
        video_path = os.path.join("data/accidents", video_file)
        metadata = extract_features(video_path, label=1)
        if metadata:
            all_metadata.append(metadata)
        main_pbar.update(1)
    
    # Process non-accident videos (label=0)
    for video_file in non_accident_videos:
        video_path = os.path.join("data/non_accidents", video_file)
        metadata = extract_features(video_path, label=0)
        if metadata:
            all_metadata.append(metadata)
        main_pbar.update(1)
    
    main_pbar.close()
    
    # Save metadata
    with open("features/metadata.json", "w") as f:
        json.dump(all_metadata, f)