import os
import json
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path

# Configuration
SEQ_LENGTH = 16          # Number of frames per sequence
OVERLAP = 8              # Overlap between sequences
CLASSES = [0, 1, 2, 3, 5, 7]  # Person, bicycle, car, motorcycle, bus, truck
FEATURES_DIR = "features"
DATA_DIR = "data"

def extract_features(video_path, label):
    # Convert string path to Path object
    video_path = Path(video_path)
    
    model = YOLO("yolov8m.pt")
    cap = cv2.VideoCapture(str(video_path))  # Path object needs to be converted back to string for cv2
    features, metadata = [], []
    prev_positions = {}  # Track previous frame positions for speed calculation
    
    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects
        results = model.track(frame, persist=True, classes=CLASSES, verbose=False)
        
        # Get object data
        boxes = results[0].boxes.xywhn.cpu().numpy()  # Normalized (x_center, y_center, width, height)
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
    
    cap.release()
    
    # Check if we got any features
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
    
    # Save features
    os.makedirs(FEATURES_DIR, exist_ok=True)
    for idx, seq in enumerate(sequences):
        np.save(f"{FEATURES_DIR}/seq_{video_path.stem}_{idx}.npy", seq["features"])
    
    # Update metadata
    metadata = {
        "video_name": video_path.name,
        "total_sequences": len(sequences),
        "label": label
    }
    return metadata

if __name__ == "__main__":
    all_metadata = []
    
    # Process accident videos (label=1)
    for video_file in os.listdir("data/accidents"):
        # Skip .DS_Store and other hidden files
        if video_file.startswith('.'):
            continue
            
        # Only process video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if not any(video_file.lower().endswith(ext) for ext in video_extensions):
            print(f"Skipping non-video file: {video_file}")
            continue
            
        video_path = os.path.join("data/accidents", video_file)
        metadata = extract_features(video_path, label=1)
        if metadata:
            all_metadata.append(metadata)
    
    # Process non-accident videos (label=0)
    for video_file in os.listdir("data/non_accidents"):
        # Skip .DS_Store and other hidden files
        if video_file.startswith('.'):
            continue
            
        # Only process video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if not any(video_file.lower().endswith(ext) for ext in video_extensions):
            print(f"Skipping non-video file: {video_file}")
            continue
            
        video_path = os.path.join("data/non_accidents", video_file)
        metadata = extract_features(video_path, label=0)
        if metadata:
            all_metadata.append(metadata)
    
    # Save metadata
    with open("features/metadata.json", "w") as f:
        json.dump(all_metadata, f)