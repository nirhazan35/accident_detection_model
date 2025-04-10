import os
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import cdist

# Configuration
SEQ_LENGTH = 16          # Number of frames per sequence
OVERLAP = 8              # Overlap between sequences
CLASSES = [0, 1, 2, 3, 5, 7]  # Person, bicycle, car, motorcycle, bus, truck
FEATURES_DIR = "features/train"
DATA_DIR = "data"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE.upper()}")

def calculate_collision_risk(boxes, velocities, track_ids):
    """Calculate collision risk between objects"""
    if len(boxes) < 2:
        return 0.0
    
    try:
        # Convert boxes to center points
        centers = boxes[:, :2]
        
        # Calculate pairwise distances
        distances = cdist(centers, centers)
        
        # Calculate relative velocities
        velocities = np.array(velocities)
        if len(velocities) == 0:
            return 0.0
            
        relative_velocities = np.zeros((len(velocities), len(velocities), 2))
        for i in range(len(velocities)):
            for j in range(len(velocities)):
                relative_velocities[i,j] = velocities[i] - velocities[j]
        
        # Calculate collision risk
        risk = 0.0
        count = 0
        for i in range(len(boxes)):
            for j in range(i+1, len(boxes)):
                # Only consider vehicles
                if track_ids[i] in [2,3,5,7] and track_ids[j] in [2,3,5,7]:
                    # Calculate time to collision
                    rel_vel = np.linalg.norm(relative_velocities[i,j])
                    if rel_vel > 0:
                        ttc = distances[i,j] / rel_vel
                        if ttc < 2.0:  # Consider collision risk if TTC < 2 seconds
                            risk += 1.0 / (ttc + 1e-6)
                            count += 1
        
        return risk / (count + 1e-6)
    except Exception as e:
        print(f"Error in collision risk calculation: {e}")
        return 0.0

def extract_features(video_path, label):
    # Convert string path to Path object
    video_path = Path(video_path)
    
    model = YOLO("yolo11m.pt").to(DEVICE)
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    
    features, metadata = [], []
    prev_positions = {}  # Track previous frame positions for speed calculation
    prev_velocities = {}  # Track previous velocities
    
    # Initialize progress bar for frames
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit='frame')
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects and get features
        results = model.track(frame, persist=True, classes=CLASSES, verbose=False, device=DEVICE)
        
        # Get object data
        boxes = results[0].boxes.xywhn.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Calculate velocities and positions
        velocities = []
        for idx, (track_id, box) in enumerate(zip(track_ids, boxes)):
            x, y = box[0], box[1]
            if track_id in prev_positions:
                dx = x - prev_positions[track_id][0]
                dy = y - prev_positions[track_id][1]
                velocities.append([dx, dy])
                prev_velocities[track_id] = [dx, dy]
            else:
                velocities.append([0, 0])
                prev_velocities[track_id] = [0, 0]
            prev_positions[track_id] = (x, y)
        
        # Calculate collision risk
        collision_risk = calculate_collision_risk(boxes, velocities, classes)
        
        # Get scene features
        scene_features = np.zeros(1024)  # Placeholder for backbone features
        
        # Initialize frame feature vector
        frame_feature = {
            "num_vehicles": np.sum(np.isin(classes, [2, 3, 5, 7])),
            "num_peds": np.sum(classes == 0),
            "motion": velocities,
            "positions": boxes[:, :2].tolist(),  # Store normalized positions
            "sizes": boxes[:, 2:].tolist(),      # Store normalized sizes
            "collision_risk": collision_risk,
            "scene_features": scene_features.tolist()  # Placeholder for backbone features
        }
        
        features.append(frame_feature)
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
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