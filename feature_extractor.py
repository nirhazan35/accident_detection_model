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
    """
    Extract rich features from video for accident detection, including YOLO backbone features
    
    Args:
        video_path: Path to the video file
        label: Video label (1 for accident, 0 for non-accident)
        
    Returns:
        Dictionary with metadata about the processed video
    """
    # Convert string path to Path object
    video_path = Path(video_path)
    
    # Load YOLO model
    model = YOLO("yolo11m.pt").to('cuda')
    
    # Access underlying PyTorch model
    yolo_model = model.model
    
    # Dictionary to store activation outputs
    activation = {}
    
    # Hook function to capture activations
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks for backbone and neck features
    # Note: These layer indices may need adjustment based on the exact YOLOv8 version
    yolo_model.model[2].register_forward_hook(get_activation('backbone_features'))
    yolo_model.model[9].register_forward_hook(get_activation('neck_features'))
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    
    # Get video properties
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
    collision_risk_history = {}  # Track collision risk over time
    
    # Initialize progress bar
    pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit='frame')
    
    # Estimate time of day based on average brightness
    brightness_samples = []
    
    # For scene complexity analysis
    scene_changes = []
    prev_frame_gray = None
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames for videos with very high frame rates to reduce processing
        if fps > 30 and frame_count % 2 != 0:
            frame_count += 1
            pbar.update(1)
            continue
            
        # Sample brightness from first 10 frames
        if frame_count < 10:
            brightness = np.mean(frame)
            brightness_samples.append(brightness)
        
        # Detect scene changes
        if prev_frame_gray is not None:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_diff = cv2.absdiff(frame_gray, prev_frame_gray)
            change_percent = np.count_nonzero(frame_diff > 25) / frame_diff.size
            scene_changes.append(change_percent)
            prev_frame_gray = frame_gray
        else:
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        # Track objects
        results = model.track(frame, persist=True, classes=CLASSES, verbose=False, device='0')
        
        # Get backbone and neck features from hooks
        if 'backbone_features' in activation and 'neck_features' in activation:
            # Global average pooling to reduce dimensions
            backbone_feats = torch.mean(activation['backbone_features'], dim=[2, 3]).cpu().numpy()
            neck_feats = torch.mean(activation['neck_features'], dim=[2, 3]).cpu().numpy()
        else:
            backbone_feats = np.zeros(512)  # Default size if features not available
            neck_feats = np.zeros(256)      # Default size if features not available
            
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
            "object_classes": [],
            "detection_confidences": [],
            "scene_change": np.mean(scene_changes[-5:]) if len(scene_changes) >= 5 else 0.0,
            "backbone_features": backbone_feats,
            "neck_features": neck_feats
        }
        
        # Calculate vehicle and pedestrian features
        high_risk_pairs = []
        
        for idx, (track_id, box, box_xyxy, cls, conf) in enumerate(zip(track_ids, boxes, boxes_xyxy, classes, confidences)):
            x, y, w, h = box
            
            # Store object class, size, and confidence
            frame_feature["object_classes"].append(int(cls))
            frame_feature["object_sizes"].append([w, h])
            frame_feature["detection_confidences"].append(float(conf))
            
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
            center_x = x + w/2
            center_y = y + h/2
            trajectory_history[track_id].append((center_x, center_y))
            if len(trajectory_history[track_id]) > 30:
                trajectory_history[track_id].pop(0)
            
            # Calculate velocities and accelerations
            if track_id in prev_positions:
                prev_x, prev_y = prev_positions[track_id]
                dx = center_x - prev_x
                dy = center_y - prev_y
                
                # Use normalized velocity (accounting for fps differences)
                vel_scale = 30.0 / fps if fps > 0 else 1.0
                dx *= vel_scale
                dy *= vel_scale
                
                # Store velocity
                frame_feature["motion"].append([dx, dy])
                
                # Calculate speed magnitude
                speed = np.sqrt(dx*dx + dy*dy)
                
                # Calculate acceleration if we have previous velocity
                if track_id in prev_velocities:
                    prev_dx, prev_dy = prev_velocities[track_id]
                    acc_x = dx - prev_dx
                    acc_y = dy - prev_dy
                    acc_magnitude = np.sqrt(acc_x*acc_x + acc_y*acc_y)
                    frame_feature["acceleration"].append([acc_x, acc_y, acc_magnitude])
                
                # Update velocities
                prev_velocities[track_id] = (dx, dy)
            
            # Update position
            prev_positions[track_id] = (center_x, center_y)
        
        # Calculate proximity between objects and collision risk
        for i in range(len(track_ids)):
            for j in range(i+1, len(track_ids)):
                # Calculate center points
                x1, y1 = boxes[i][0] + boxes[i][2]/2, boxes[i][1] + boxes[i][3]/2
                x2, y2 = boxes[j][0] + boxes[j][2]/2, boxes[j][1] + boxes[j][3]/2
                
                # Euclidean distance between centers (normalized)
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                
                # Add class pair and their distance
                class_pair = sorted([int(classes[i]), int(classes[j])])
                object_pair = (int(track_ids[i]), int(track_ids[j]))
                
                # Add basic proximity data
                frame_feature["proximities"].append({
                    "classes": class_pair,
                    "distance": dist,
                    "ids": [int(track_ids[i]), int(track_ids[j])]
                })
                
                # Calculate collision risk if we have velocity data
                if track_ids[i] in prev_velocities and track_ids[j] in prev_velocities:
                    v1_x, v1_y = prev_velocities[track_ids[i]]
                    v2_x, v2_y = prev_velocities[track_ids[j]]
                    
                    # Calculate relative velocity
                    rel_v_x = v1_x - v2_x
                    rel_v_y = v1_y - v2_y
                    rel_v_mag = np.sqrt(rel_v_x**2 + rel_v_y**2)
                    
                    # Calculate the dot product to see if objects are moving toward each other
                    direction_vector = [x2 - x1, y2 - y1]
                    dir_mag = np.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
                    
                    if dir_mag > 0:
                        direction_vector = [direction_vector[0]/dir_mag, direction_vector[1]/dir_mag]
                        dot_product = rel_v_x * direction_vector[0] + rel_v_y * direction_vector[1]
                        
                        # Collision risk is high when:
                        # - Objects are close
                        # - Moving toward each other (negative dot product)
                        # - With high relative velocity
                        collision_risk = 0
                        if dot_product < 0 and dist < 0.2:  # Moving toward each other and close
                            collision_risk = abs(dot_product) * rel_v_mag / max(0.01, dist)
                            
                            # Track high-risk pairs
                            if collision_risk > 5.0:
                                high_risk_pairs.append({
                                    "ids": object_pair,
                                    "risk": collision_risk,
                                    "distance": dist
                                })
                            
                            # Track collision risk history
                            if object_pair not in collision_risk_history:
                                collision_risk_history[object_pair] = []
                            collision_risk_history[object_pair].append(collision_risk)
                            
                            # Keep history limited
                            if len(collision_risk_history[object_pair]) > 10:
                                collision_risk_history[object_pair].pop(0)
        
        # Add collision risk data to frame feature
        frame_feature["high_risk_pairs"] = high_risk_pairs
        
        # Analyze sustained collision risks
        sustained_risks = []
        for pair, risks in collision_risk_history.items():
            if len(risks) >= 3:  # Need at least 3 frames of history
                avg_risk = np.mean(risks)
                if avg_risk > 3.0:  # Sustained high risk
                    sustained_risks.append({
                        "pair": pair,
                        "avg_risk": avg_risk,
                        "trend": np.mean(np.diff(risks))  # Positive means increasing risk
                    })
        
        frame_feature["sustained_collision_risks"] = sustained_risks
        
        # Perform trajectory analysis for anomaly detection
        trajectory_anomalies = {}
        for track_id, trajectory in trajectory_history.items():
            if len(trajectory) >= 5:  # Need enough points for analysis
                # Analyze if trajectory is smooth or has sudden changes
                points = np.array(trajectory[-5:])
                
                # Calculate angles between consecutive segments
                angles = []
                for i in range(len(points) - 2):
                    v1 = points[i+1] - points[i]
                    v2 = points[i+2] - points[i+1]
                    
                    if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
                        angles.append(angle)
                
                if angles:
                    # If any angle indicates sharp turn (> 45 degrees)
                    max_angle = max(angles)
                    if max_angle > np.pi/4:
                        trajectory_anomalies[track_id] = {
                            "max_angle": max_angle,
                            "idx": angles.index(max_angle) + 1  # Frame index where anomaly occurs
                        }
        
        frame_feature["trajectory_anomalies"] = trajectory_anomalies
        
        features.append(frame_feature)
        frame_count += 1
        pbar.update(1)
    
    cap.release()
    pbar.close()
    
    if not features:
        print(f"No frames extracted from: {video_path}")
        return None
    
    # Calculate time of day feature 
    avg_brightness = np.mean(brightness_samples) / 255.0 if brightness_samples else 0.5
    time_of_day = avg_brightness
    
    # Add video-level features to all frames
    for feature in features:
        feature["time_of_day"] = time_of_day
        feature["video_fps"] = fps
        feature["resolution"] = [frame_width, frame_height]
    
    # Split into overlapping sequences
    sequences = []
    for i in range(0, len(features) - SEQ_LENGTH + 1, OVERLAP):
        seq = features[i:i+SEQ_LENGTH]
        
        # Add sequence-level features
        seq_features = {
            "features": seq,
            "label": label,
            "source_video": video_path.name,
            "start_frame": i,
            "end_frame": i + SEQ_LENGTH - 1,
            "seq_collision_risks": []
        }
        
        # Extract sequence-level collision risk statistics
        all_risks = []
        for frame in seq:
            if "high_risk_pairs" in frame:
                for pair in frame["high_risk_pairs"]:
                    all_risks.append(pair["risk"])
        
        if all_risks:
            seq_features["seq_collision_risks"] = {
                "max_risk": np.max(all_risks),
                "avg_risk": np.mean(all_risks),
                "risk_count": len(all_risks)
            }
        
        sequences.append(seq_features)
    
    # Save features with progress bar
    save_pbar = tqdm(total=len(sequences), desc=f"Saving sequences for {video_path.name}", unit='seq')
    os.makedirs(FEATURES_DIR, exist_ok=True)
    
    for idx, seq in enumerate(sequences):
        # Save full feature data
        np.save(f"{FEATURES_DIR}/seq_{video_path.stem}_{idx}.npy", seq["features"])
        
        # Also save a compressed version without the backbone features for faster loading
        compressed_features = []
        for frame in seq["features"]:
            frame_copy = frame.copy()
            # Remove large backbone feature arrays to save space for quick loading
            if "backbone_features" in frame_copy:
                del frame_copy["backbone_features"]
            if "neck_features" in frame_copy:
                del frame_copy["neck_features"]
            compressed_features.append(frame_copy)
        
        np.save(f"{FEATURES_DIR}/seq_{video_path.stem}_{idx}_compressed.npy", compressed_features)
        save_pbar.update(1)
    
    save_pbar.close()
    
    metadata = {
        "video_name": video_path.name,
        "total_sequences": len(sequences),
        "label": label,
        "time_of_day": time_of_day,
        "fps": fps,
        "resolution": [frame_width, frame_height],
        "total_frames": frame_count,
        "processed_frames": len(features)
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
    
    print(f"Feature extraction complete for {len(all_metadata)} videos")
    print(f"Features saved to {FEATURES_DIR}")