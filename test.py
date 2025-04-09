import torch
import numpy as np
import cv2
from LSTM import LSTM
from ultralytics import YOLO
from pathlib import Path
from feature_extractor import SEQ_LENGTH, CLASSES

# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE.upper()}")

class AccidentDetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = LSTM()
        self.model = self.model.to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.threshold = threshold
        self.yolo_model = YOLO("yolo11m.pt").to(DEVICE)   # Initialize YOLO model for feature extraction
        # Configuration from feature_extractor.py
        self.seq_length = SEQ_LENGTH
        self.classes = CLASSES  # Person, bicycle, car, motorcycle, bus, truck
        print(f"SEQ_LENGTH: {self.seq_length}, CLASSES: {self.classes}") # remove @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    
    def extract_features(self, video_path):
        """Extract features from video using the same process as feature_extractor.py"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        features = []
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
            results = self.yolo_model.track(frame, persist=True, classes=self.classes, verbose=False, device='0')
            
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
        
        # Check if we got enough frames for a sequence
        if len(features) < self.seq_length:
            print(f"Not enough frames extracted from: {video_path}")
            return None
            
        return features
    
    def process_features(self, features):
        """Process raw features into the format expected by the model"""
        # Convert features to the same format used in training
        processed_seq = []
        for frame in features[:self.seq_length]:  # Take only the first seq_length frames
            # Calculate average motion
            avg_motion = np.mean(frame["motion"], axis=0) if len(frame["motion"]) > 0 else [0, 0]
            
            # Format each frame as [vehicles, peds, avg_dx, avg_dy]
            processed_frame = [
                frame["num_vehicles"],
                frame["num_peds"],
                avg_motion[0],
                avg_motion[1]
            ]
            processed_seq.append(processed_frame)
            
        # Convert to tensor with batch dimension
        return torch.FloatTensor([processed_seq]).to(DEVICE)
    
    def predict(self, video_path):
        """Predict if a video contains an accident"""
        # Extract features from video
        features = self.extract_features(video_path)
        if features is None:
            return None
        
        # Process features into the format expected by the model
        sequence = self.process_features(features)
        
        # Make prediction
        with torch.no_grad():
            output = self.model(sequence)
            
        prediction = (output > self.threshold).float().item()
        probability = output.item()

        torch.cuda.empty_cache()  # Clear GPU memory
        
        return {
            "prediction": int(prediction),  # 0 or 1
            "probability": probability,
            "label": "Accident" if prediction == 1 else "No Accident"
        }
    
    def adjust_threshold(self, new_threshold):
        self.threshold = new_threshold

# Example usage:
if __name__ == "__main__":
    detector = AccidentDetector("models/accident_lstm.pth", threshold=0.3)
    video_path = "data/test/Untitled6.mp4"  # Replace with your test video path
    result = detector.predict(video_path)
    
    if result:
        print(f"Prediction: {result['label']}")
        print(f"Probability: {result['probability']:.4f}")
    else:
        print("Failed to process video.")