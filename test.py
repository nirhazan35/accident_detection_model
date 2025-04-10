import torch
import numpy as np
import cv2
from LSTM import LSTM
from ultralytics import YOLO
from pathlib import Path
from feature_extractor import SEQ_LENGTH, CLASSES
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import time

# Configuration
MODEL_PATH = "models/accident_lstm_20250410_151010.pth"
VIDEO_PATH = "data/test/0IPAgFHyRVI.mp4"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE.upper()}")

def calculate_collision_risk(boxes, velocities, track_ids):
    """Calculate collision risk between objects"""
    if len(boxes) < 2:
        return 0.0
    
    # Convert boxes to center points
    centers = boxes[:, :2]
    
    # Calculate pairwise distances
    distances = cdist(centers, centers)
    
    # Calculate relative velocities
    velocities = np.array(velocities)
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

def visualize_attention(attention_weights, frame):
    """Visualize attention weights on the frame"""
    # Normalize attention weights
    weights = attention_weights.squeeze().cpu().numpy()
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-6)
    
    # Create attention heatmap
    heatmap = np.zeros_like(frame)
    for i, weight in enumerate(weights):
        # Map attention weight to color (red = high attention)
        color = np.array([0, 0, 255 * weight])  # BGR format
        # Draw attention indicator
        cv2.rectangle(heatmap, (10, 10 + i*20), (30, 30 + i*20), color, -1)
        # Add weight value text
        cv2.putText(heatmap, f"{weight:.2f}", (40, 25 + i*20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend heatmap with original frame
    alpha = 0.5
    output = cv2.addWeighted(frame, 1-alpha, heatmap, alpha, 0)
    return output

def test_video(video_path, model, threshold=0.5):
    """Test a single video with accident detection"""
    # Initialize YOLO model for tracking
    yolo_model = YOLO("yolo11m.pt").to(DEVICE)
    
    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize feature buffer
    features_buffer = []
    accident_detected = False
    accident_vehicles = set()  # Track vehicles involved in accidents
    
    # Process video
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Track objects and get features
        results = yolo_model.track(frame, persist=True, verbose=False, device=DEVICE)
        
        # Get object data
        boxes = results[0].boxes.xywhn.cpu().numpy()
        track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
        classes = results[0].boxes.cls.cpu().numpy()
        
        # Calculate velocities and positions
        velocities = []
        for idx, (track_id, box) in enumerate(zip(track_ids, boxes)):
            x, y = box[0], box[1]
            velocities.append([x, y])  # Simplified velocity calculation
        
        # Calculate collision risk
        collision_risk = calculate_collision_risk(boxes, velocities, classes)
        
        # Get scene features (placeholder)
        scene_features = np.zeros(1024)
        
        # Create frame feature
        frame_feature = {
            "num_vehicles": np.sum(np.isin(classes, [2, 3, 5, 7])),
            "num_peds": np.sum(classes == 0),
            "motion": velocities,
            "positions": boxes[:, :2].tolist(),
            "sizes": boxes[:, 2:].tolist(),
            "collision_risk": collision_risk,
            "scene_features": scene_features.tolist()
        }
        
        # Add to buffer
        features_buffer.append(frame_feature)
        
        # Process sequence if buffer is full
        if len(features_buffer) >= 16:  # Assuming sequence length of 16
            # Prepare input for model
            sequence = features_buffer[-16:]  # Get last 16 frames
            
            # Get model prediction
            with torch.no_grad():
                prediction, _ = model([sequence])
                prediction = prediction.item()
            
            # Check for accident
            if prediction > threshold and not accident_detected:
                # Get vehicles involved in the current frame
                current_vehicles = set(track_ids[np.isin(classes, [2, 3, 5, 7])])  # Vehicle classes
                
                # Check if these vehicles were already involved in an accident
                if not current_vehicles.issubset(accident_vehicles):
                    print(f"\n🚨 ACCIDENT DETECTED!")
                    print(f"Confidence: {prediction:.2%}")
                    print(f"Vehicles involved: {current_vehicles}")
                    accident_vehicles.update(current_vehicles)
                    accident_detected = True
            
            # Reset accident detection flag after some frames
            if accident_detected and frame_count % (fps * 2) == 0:  # Reset every 2 seconds
                accident_detected = False
            
            # Display original frame
            cv2.imshow('Accident Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load model
    model = LSTM().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])
    model.eval()
    
    # Test video
    video_path = Path(VIDEO_PATH)  # Replace with your test video path
    test_video(video_path, model)