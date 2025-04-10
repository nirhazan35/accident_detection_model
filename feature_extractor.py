import os
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.spatial import distance

# Configuration
SEQ_LENGTH = 16          # Number of frames per sequence
OVERLAP = 8              # Overlap between sequences
CLASSES = [0, 1, 2, 3, 5, 7]  # Person, bicycle, car, motorcycle, bus, truck
FEATURES_DIR = "features/train"
DATA_DIR = "data"

def extract_features(video_path, label):
    # Convert string path to Path object
    video_path = Path(video_path)
    
    model = YOLO("yolo11m.pt").to('cuda')
    cap = cv2.VideoCapture(str(video_path))
    
    # Get total frames and video properties for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return None
    
    features, metadata = [], []
    prev_positions = {}  # Track previous frame positions for speed calculation
    prev_velocities = {} # Track previous velocities for acceleration calculation
    trajectory_history = {} # Track object trajectories
    
    # Initialize progress bar for frames
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit='frame')
    
    # Estimate time of day based on average brightness
    brightness_samples = []
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample brightness from first 10 frames
        if frame_count < 10:
            brightness = np.mean(frame)
            brightness_samples.append(brightness)
        
        # Track objects
        results = model.track(frame, persist=True, classes=CLASSES, verbose=False, device='0')
        
        # Get object data
        if results[0].boxes.xywhn.numel() > 0:
            boxes = results[0].boxes.xywhn.cpu().numpy()
            boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()  # Get absolute pixel coordinates
            track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
        else:
            boxes = np.array([])
            boxes_xyxy = np.array([])
            track_ids = np.array([])
            classes = np.array([])
            confidences = np.array([])
        
        # Initialize frame feature vector with enhanced features
        frame_feature = {
            "num_vehicles": np.sum(np.isin(classes, [2, 3, 5, 7])),
            "num_peds": np.sum(classes == 0),
            "motion": [],
            "acceleration": [],
            "proximities": [],
            "edge_distances": [],
            "object_sizes": [],
            "object_classes": []
        }
        
        # Calculate vehicle and pedestrian features
        for idx, (track_id, box, box_xyxy, cls, conf) in enumerate(zip(track_ids, boxes, boxes_xyxy, classes, confidences)):
            x, y, w, h = box
            
            # Store object class and size
            frame_feature["object_classes"].append(int(cls))
            frame_feature["object_sizes"].append([w, h])
            
            # Calculate distance from frame edges (normalized to [0,1])
            left_edge = x
            right_edge = 1.0 - (x + w)
            top_edge = y
            bottom_edge = 1.0 - (y + h)
            min_edge_dist = min(left_edge, right_edge, top_edge, bottom_edge)
            frame_feature["edge_distances"].append(min_edge_dist)
            
            # Initialize trajectory for this object if not exists
            if track_id not in trajectory_history:
                trajectory_history[track_id] = []
            
            # Add current position to trajectory (keep last 30 positions)
            trajectory_history[track_id].append((x, y))
            if len(trajectory_history[track_id]) > 30:
                trajectory_history[track_id].pop(0)
            
            # Calculate velocities and accelerations
            if track_id in prev_positions:
                prev_x, prev_y = prev_positions[track_id]
                dx = x - prev_x
                dy = y - prev_y
                
                # Store velocity
                frame_feature["motion"].append([dx, dy])
                
                # Calculate acceleration if we have previous velocity
                if track_id in prev_velocities:
                    prev_dx, prev_dy = prev_velocities[track_id]
                    acc_x = dx - prev_dx
                    acc_y = dy - prev_dy
                    frame_feature["acceleration"].append([acc_x, acc_y])
                
                # Update velocities
                prev_velocities[track_id] = (dx, dy)
            
            # Update position
            prev_positions[track_id] = (x, y)
        
        # Calculate proximity between objects (potential collisions)
        for i in range(len(track_ids)):
            for j in range(i+1, len(track_ids)):
                # Calculate center points
                x1, y1 = boxes[i][0], boxes[i][1]
                x2, y2 = boxes[j][0], boxes[j][1]
                
                # Euclidean distance between centers (normalized)
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                # Add class pair and their distance
                class_pair = sorted([int(classes[i]), int(classes[j])])
                frame_feature["proximities"].append({
                    "classes": class_pair,
                    "distance": dist,
                    "ids": [int(track_ids[i]), int(track_ids[j])]
                })
        
        # Perform trajectory analysis for anomaly detection
        for track_id, trajectory in trajectory_history.items():
            if len(trajectory) >= 5:  # Need enough points for analysis
                # Analyze if trajectory is smooth or has sudden changes
                if track_id in frame_feature.get("trajectory_anomalies", {}):
                    continue
                
                # Simple anomaly: check for sudden direction changes
                if len(trajectory) >= 3:
                    points = np.array(trajectory[-3:])
                    if len(points) == 3:
                        # Calculate angles between consecutive segments
                        v1 = points[1] - points[0]
                        v2 = points[2] - points[1]
                        
                        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                            
                            # If angle indicates sharp turn (> 45 degrees)
                            if angle > np.pi/4:
                                if "trajectory_anomalies" not in frame_feature:
                                    frame_feature["trajectory_anomalies"] = {}
                                frame_feature["trajectory_anomalies"][track_id] = angle
        
        features.append(frame_feature)
        frame_count += 1
        pbar.update(1)  # Update frame progress
    
    cap.release()
    pbar.close()  # Close frame progress bar
    
    if not features:
        print(f"No frames extracted from: {video_path}")
        return None
    
    # Calculate time of day feature (from 0 to 1, where 0 is dark, 1 is bright)
    avg_brightness = np.mean(brightness_samples) / 255.0 if brightness_samples else 0.5
    time_of_day = avg_brightness
    
    # Add video-level features to all frames
    for feature in features:
        feature["time_of_day"] = time_of_day
        feature["video_fps"] = fps
    
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
        "label": label,
        "time_of_day": time_of_day,
        "fps": fps,
        "resolution": [frame_width, frame_height]
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