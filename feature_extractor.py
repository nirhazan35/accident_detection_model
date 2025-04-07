import os
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm

# Configuration
SEQ_LENGTH = 16          # Number of frames per sequence
OVERLAP = 8              # Overlap between sequences
CLASSES = [0, 1, 2, 3, 5, 7]  # Person, bicycle, car, motorcycle, bus, truck
FEATURES_DIR = "features/train"
DATA_DIR = "data"

# Check if CUDA is available
assert torch.cuda.is_available(), "CUDA-enabled GPU is required!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def extract_features(video_path, label):
    # Convert string path to Path object
    video_path = Path(video_path)
    
    model = YOLO("yolo11m.pt").to('cuda')
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    
    features, metadata = [], []
    prev_positions = {}  # Track previous frame positions for speed calculation
    
    # Initialize progress bar for frames
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit='frame')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects
        results = model.track(frame, persist=True, classes=CLASSES, verbose=False, device='0')
        
        # Get object data
        boxes = results[0].boxes.xywhn.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Initialize frame feature vector
        frame_feature = {
            "num_vehicles": np.sum(np.isin(classes, [2, 3, 5, 7])),
            "num_peds": np.sum(classes == 0),
            "motion": []
        }
        
        # Calculate speeds using tracking IDs
        for idx, (track_id, box) in enumerate(zip(track_ids, boxes)):
            x, y = box[0], box[1]
            if track_id in prev_positions:
                dx = x - prev_positions[track_id][0]
                dy = y - prev_positions[track_id][1]
                frame_feature["motion"].append([dx, dy])
            prev_positions[track_id] = (x, y)
        
        features.append(frame_feature)
        frame_count += 1
        pbar.update(1)  # Update frame progress
    
    cap.release()
    pbar.close()  # Close frame progress bar
    
    if not features:
        print(f"No frames extracted from: {video_path}")
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