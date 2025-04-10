import torch
import numpy as np
import cv2
from pathlib import Path
import os
from ultralytics import YOLO
from tqdm import tqdm

class AccidentDetector:
    def __init__(self, model_path, feature_norm_path=None, threshold=0.5, device='cuda'):
        """
        Initialize the accident detector with a trained model.
        
        Args:
            model_path: Path to the trained model checkpoint
            feature_norm_path: Path to feature normalization statistics
            threshold: Detection threshold (default: 0.5)
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device
        self.threshold = threshold
        self.seq_length = 16  # Number of frames to analyze
        
        # Load model architecture and weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Initialize model architecture
        from train import AttentionLSTM
        
        input_size = checkpoint.get('input_size', 10)  # Default to 10 if not specified
        hidden_size = checkpoint.get('hidden_size', 128)
        num_layers = checkpoint.get('num_layers', 2)
        
        self.model = AttentionLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(device)
        
        # Load model weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load feature normalization if available
        self.feature_mean = None
        self.feature_std = None
        if feature_norm_path and os.path.exists(feature_norm_path):
            norm_data = np.load(feature_norm_path)
            self.feature_mean = torch.tensor(norm_data['mean'], device=device)
            self.feature_std = torch.tensor(norm_data['std'], device=device)
            print(f"Loaded feature normalization statistics")
        
        # Initialize YOLO model for object detection
        self.detector = YOLO("yolo11m.pt").to(device)
        self.classes = [0, 1, 2, 3, 5, 7]  # Person, bicycle, car, motorcycle, bus, truck
        
        print(f"Accident detector initialized with threshold {threshold}")
        print(f"Model expects {input_size} features per frame")
    
    def adjust_threshold(self, new_threshold):
        """Adjust the detection threshold"""
        self.threshold = new_threshold
    
    def predict_from_video(self, video_path, visualize=False):
        """
        Predict accident probability from a video file
        
        Args:
            video_path: Path to video file
            visualize: Whether to create visualization of predictions
            
        Returns:
            Dictionary with prediction results
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize feature extraction variables
        features = []
        prev_positions = {}  # Track previous positions
        prev_velocities = {}  # Track previous velocities
        trajectory_history = {}  # Track trajectories
        brightness_samples = []  # Track brightness
        frame_count = 0
        
        # Initialize visualization
        if visualize:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_path = f"output_{video_path.stem}.mp4"
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process each frame
        pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample brightness for time-of-day estimation
            if frame_count < 10:
                brightness = np.mean(frame)
                brightness_samples.append(brightness)
            
            # Detect and track objects
            results = self.detector.track(
                frame, persist=True, classes=self.classes, verbose=False
            )
            
            # Process detections
            if results[0].boxes.xywhn.numel() > 0:
                boxes = results[0].boxes.xywhn.cpu().numpy()
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy() if results[0].boxes.id is not None else []
                classes = results[0].boxes.cls.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
            else:
                boxes = np.array([])
                boxes_xyxy = np.array([])
                track_ids = np.array([])
                classes = np.array([])
                confidences = np.array([])
            
            # Extract frame features
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
            
            # Process each detected object
            for idx, (track_id, box, box_xyxy, cls, conf) in enumerate(
                zip(track_ids, boxes, boxes_xyxy, classes, confidences)
            ):
                x, y, w, h = box
                
                # Store object class and size
                frame_feature["object_classes"].append(int(cls))
                frame_feature["object_sizes"].append([w, h])
                
                # Calculate distance from edges
                left_edge = x
                right_edge = 1.0 - (x + w)
                top_edge = y
                bottom_edge = 1.0 - (y + h)
                min_edge_dist = min(left_edge, right_edge, top_edge, bottom_edge)
                frame_feature["edge_distances"].append(min_edge_dist)
                
                # Track trajectory
                if track_id not in trajectory_history:
                    trajectory_history[track_id] = []
                
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
                    
                    # Calculate acceleration
                    if track_id in prev_velocities:
                        prev_dx, prev_dy = prev_velocities[track_id]
                        acc_x = dx - prev_dx
                        acc_y = dy - prev_dy
                        frame_feature["acceleration"].append([acc_x, acc_y])
                    
                    # Update velocities
                    prev_velocities[track_id] = (dx, dy)
                
                # Update position
                prev_positions[track_id] = (x, y)
            
            # Calculate proximity between objects
            for i in range(len(track_ids)):
                for j in range(i+1, len(track_ids)):
                    # Calculate centers
                    x1, y1 = boxes[i][0], boxes[i][1]
                    x2, y2 = boxes[j][0], boxes[j][1]
                    
                    # Euclidean distance
                    dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                    
                    # Add class pair and distance
                    class_pair = sorted([int(classes[i]), int(classes[j])])
                    frame_feature["proximities"].append({
                        "classes": class_pair,
                        "distance": dist,
                        "ids": [int(track_ids[i]), int(track_ids[j])]
                    })
            
            # Analyze trajectories for anomalies
            for track_id, trajectory in trajectory_history.items():
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
            
            # Add time of day feature
            if frame_count == 0:
                avg_brightness = np.mean(brightness_samples) / 255.0 if brightness_samples else 0.5
                time_of_day = avg_brightness
            
            frame_feature["time_of_day"] = time_of_day
            frame_feature["video_fps"] = fps
            
            features.append(frame_feature)
            
            # Create visualization if requested
            if visualize:
                # Make a copy of the frame for visualization
                vis_frame = frame.copy()
                
                # Draw detections
                for box_xyxy, track_id, cls, conf in zip(boxes_xyxy, track_ids, classes, confidences):
                    x1, y1, x2, y2 = box_xyxy.astype(int)
                    
                    # Set color based on class
                    color = (0, 255, 0)  # Green for vehicles
                    if cls == 0:  # Person
                        color = (0, 0, 255)  # Red for pedestrians
                    
                    # Draw bounding box
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    class_names = ["Person", "Bicycle", "Car", "Motorcycle", "Bus", "Truck"]
                    class_idx = int(cls)
                    class_name = class_names[self.classes.index(class_idx)] if class_idx in self.classes else f"Class {class_idx}"
                    label = f"{class_name} {conf:.2f} ID:{int(track_id)}"
                    cv2.putText(vis_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # If we have enough frames, run inference and show the risk level
                if len(features) >= self.seq_length:
                    # Use last seq_length frames
                    risk = self._predict_from_features(features[-self.seq_length:])
                    risk_text = f"Accident Risk: {risk:.1%}"
                    
                    # Set text color based on risk
                    text_color = (0, 255, 0)  # Green for low risk
                    if risk > 0.5:
                        text_color = (0, 165, 255)  # Orange for medium risk
                    if risk > 0.75:
                        text_color = (0, 0, 255)  # Red for high risk
                    
                    # Add risk text to frame
                    cv2.putText(vis_frame, risk_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
                
                # Write frame to output video
                out.write(vis_frame)
            
            frame_count += 1
            pbar.update(1)
            
            # Keep only the last seq_length frames to avoid using too much memory
            if len(features) > self.seq_length * 2:
                features = features[-self.seq_length * 2:]
        
        # Clean up
        cap.release()
        pbar.close()
        
        if visualize:
            out.release()
            print(f"Visualization saved to {output_path}")
        
        # Make prediction using all features
        if len(features) < self.seq_length:
            print(f"Warning: Not enough frames ({len(features)}) for reliable prediction")
            return {"probability": 0.0, "is_accident": False, "confidence": 0.0}
        
        # Split into overlapping sequences for more robust prediction
        all_probs = []
        for i in range(0, max(1, len(features) - self.seq_length + 1), 8):
            seq = features[i:i+self.seq_length]
            if len(seq) == self.seq_length:
                prob = self._predict_from_features(seq)
                all_probs.append(prob)
        
        # Aggregate predictions (use max as we're interested in detecting any accident)
        if all_probs:
            final_prob = max(all_probs)
            is_accident = final_prob > self.threshold
            
            # Calculate confidence (how far from threshold)
            confidence = abs(final_prob - self.threshold) * 2
            confidence = min(confidence, 1.0)
            
            return {
                "probability": float(final_prob),
                "is_accident": bool(is_accident),
                "confidence": float(confidence),
                "num_sequences": len(all_probs)
            }
        else:
            return {"probability": 0.0, "is_accident": False, "confidence": 0.0, "num_sequences": 0}
    
    def _predict_from_features(self, frame_features):
        """Process raw features into model input and make prediction"""
        # Convert features to tensor format
        processed_seq = []
        for frame in frame_features:
            # Extract features
            vehicle_count = frame["num_vehicles"]
            ped_count = frame["num_peds"]
            
            # Process motion vectors (average)
            avg_motion = np.mean(frame["motion"], axis=0) if len(frame["motion"]) > 0 else [0, 0]
            
            # Process acceleration vectors (average)
            avg_accel = np.mean(frame["acceleration"], axis=0) if len(frame["acceleration"]) > 0 else [0, 0]
            
            # Process proximities (minimum distance between objects)
            min_proximity = 1.0  # Default to max distance
            if frame["proximities"]:
                min_proximity = min([p["distance"] for p in frame["proximities"]], default=1.0)
            
            # Process edge distances (minimum distance to edge)
            min_edge_dist = np.min(frame["edge_distances"]) if len(frame["edge_distances"]) > 0 else 1.0
            
            # Get time of day
            time_of_day = frame.get("time_of_day", 0.5)
            
            # Check for trajectory anomalies
            has_anomaly = 1.0 if "trajectory_anomalies" in frame and frame["trajectory_anomalies"] else 0.0
            
            # Combine features into a single vector
            frame_vector = [
                vehicle_count, 
                ped_count,
                avg_motion[0], avg_motion[1],
                avg_accel[0], avg_accel[1],
                min_proximity,
                min_edge_dist,
                time_of_day,
                has_anomaly
            ]
            
            processed_seq.append(frame_vector)
        
        # Convert to tensor
        x = torch.tensor(processed_seq, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Normalize features if normalization statistics are available
        if self.feature_mean is not None and self.feature_std is not None:
            x = (x - self.feature_mean) / (self.feature_std + 1e-6)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(x)
        
        return output.item()

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Accident detection from video")
    parser.add_argument("video_path", help="Path to video file")
    parser.add_argument("--model", default="models/accident_lstm_best_f1.pth", help="Path to model file")
    parser.add_argument("--norm", default="models/feature_normalization.npz", help="Path to feature normalization file")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection threshold")
    parser.add_argument("--visualize", action="store_true", help="Create visualization video")
    args = parser.parse_args()
    
    # Initialize detector
    detector = AccidentDetector(
        model_path=args.model,
        feature_norm_path=args.norm,
        threshold=args.threshold
    )
    
    # Run prediction
    result = detector.predict_from_video(args.video_path, visualize=args.visualize)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Accident Probability: {result['probability']:.1%}")
    print(f"Is Accident: {'Yes' if result['is_accident'] else 'No'}")
    print(f"Confidence: {result['confidence']:.1%}")