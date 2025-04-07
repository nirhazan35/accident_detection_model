import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from LSTM import LSTM

class LiveAccidentDetector:
    def __init__(self, model_path, threshold=0.3, window_size=16, overlap=15):
        # Model setup
        self.model = self.load_lstm(model_path)
        self.yolo = YOLO("yolo11m.pt")
        self.threshold = threshold
        
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
        # YOLO object tracking (CPU only)
        results = self.yolo.track(
            frame, 
            persist=True, 
            classes=self.classes,
            device='cpu',
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
            
        # Convert to tensor and predict
        sequence = torch.FloatTensor([processed_seq])
        with torch.no_grad():
            prob = self.model(sequence).item()
            
        return prob > self.threshold, prob

    def run(self, video_source=0):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print("Error opening video source")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame (without bounding boxes)
            frame_feature = self.process_frame(frame)
            self.frame_buffer.append(frame_feature)

            # Make predictions when buffer is full
            if len(self.frame_buffer) >= self.window_size:
                accident_detected, confidence = self.predict_accident()
                if accident_detected and not self.accident_reported:
                    print(f"! ACCIDENT DETECTED ! Confidence: {confidence:.2f}")
                    self.accident_reported = True
                elif not accident_detected:
                    self.accident_reported = False

            # Show the video frame in a window (without bounding boxes)
            cv2.imshow('Live Accident Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = LiveAccidentDetector(
        model_path="models/accident_lstm.pth",
        threshold=0.3
    )
    detector.run("data/temp/accidents/11.mp4")
