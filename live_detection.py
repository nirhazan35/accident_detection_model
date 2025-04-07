import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from LSTM import LSTM

class LiveAccidentDetector:
    def __init__(self, model_path, threshold=0.3, window_size=16, overlap=15, processing_scale=0.5, frame_skip=2):
        # Model setup
        self.model = self.load_lstm(model_path)
        self.yolo = YOLO("yolo11m.pt")
        self.threshold = threshold
        
        # GPU configuration
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Performance optimization parameters
        self.processing_scale = processing_scale  # Scale factor for frame processing
        self.frame_skip = frame_skip  # Process every nth frame
        
        # Sequence configuration
        self.window_size = window_size
        self.overlap = overlap
        self.frame_buffer = deque(maxlen=window_size)
        self.prev_positions = {}
        self.accident_reported = False  # To prevent repeated alerts
        
        # Classes to track
        self.classes = [0, 1, 2, 3, 5, 7]  # Person, vehicles
        
    def load_lstm(self, model_path):
        model = LSTM()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def process_frame(self, frame):
        # YOLO object tracking with GPU
        results = self.yolo.track(
            frame, 
            persist=True, 
            classes=self.classes,
            device=self.device,
            verbose=False
        )
        
        # Feature extraction
        boxes = results[0].boxes.xywhn.numpy()
        track_ids = results[0].boxes.id.numpy() if results[0].boxes.id is not None else []
        classes = results[0].boxes.cls.numpy()
        
        frame_feature = {
            "num_vehicles": np.sum(np.isin(classes, [2, 3, 5, 7])),
            "num_peds": np.sum(classes == 0),
            "motion": []
        }
        
        # Motion calculation
        for idx, (track_id, box) in enumerate(zip(track_ids, boxes)):
            x, y = box[0], box[1]
            if track_id in self.prev_positions:
                dx = x - self.prev_positions[track_id][0]
                dy = y - self.prev_positions[track_id][1]
                frame_feature["motion"].append([dx, dy])
            self.prev_positions[track_id] = (x, y)
            
        return frame_feature

    def predict_accident(self):
        # Convert buffer to sequence
        processed_seq = []
        for frame in self.frame_buffer:
            avg_motion = np.mean(frame["motion"], axis=0) if frame["motion"] else [0, 0]
            processed_frame = [
                frame["num_vehicles"],
                frame["num_peds"],
                avg_motion[0],
                avg_motion[1]
            ]
            processed_seq.append(processed_frame)
            
        # Convert to tensor and predict using GPU
        sequence = torch.FloatTensor([processed_seq]).to(self.device)
        with torch.no_grad():
            prob = self.model(sequence).item()
            
        return prob > self.threshold, prob

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error opening video source")
            return

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Keep original frame for display
            display_frame = frame.copy()
            
            # Process only every nth frame based on frame_skip
            if frame_count % self.frame_skip == 0:
                # Resize frame for processing (lower resolution)
                h, w = frame.shape[:2]
                process_w = int(w * self.processing_scale)
                process_h = int(h * self.processing_scale)
                process_frame = cv2.resize(frame, (process_w, process_h))
                
                # Process downscaled frame
                frame_feature = self.process_frame(process_frame)
                self.frame_buffer.append(frame_feature)

                # Make predictions when buffer is full
                if len(self.frame_buffer) >= self.window_size:
                    accident_detected, confidence = self.predict_accident()
                    if accident_detected and not self.accident_reported:
                        print(f"! ACCIDENT DETECTED ! Confidence: {confidence:.2f}")
                        # Optional: Add visual indicator of detection
                        cv2.putText(display_frame, f"ACCIDENT: {confidence:.2f}", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        self.accident_reported = True
                    elif not accident_detected:
                        self.accident_reported = False

            # Increment frame counter
            frame_count += 1

            # Show the original video frame in a window
            cv2.imshow('Live Accident Detection', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LiveAccidentDetector(
        model_path="models/accident_lstm.pth",
        threshold=0.3,
        processing_scale=0.5,  # Process at 50% resolution
        frame_skip=2           # Process every 2nd frame
    )
    detector.run("data/temp/accidents/11.mp4")