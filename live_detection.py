import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
from pathlib import Path
from LSTM import LSTM
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LiveAccidentDetector:
    def __init__(self, model_path, threshold=0.3, window_size=16, overlap=15, processing_scale=0.5, frame_skip=2):
        # Input validation
        if not isinstance(model_path, (str, Path)):
            raise TypeError("model_path must be a string or Path object")
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if overlap < 0 or overlap >= window_size:
            raise ValueError("overlap must be non-negative and less than window_size")
        if not 0 < processing_scale <= 1:
            raise ValueError("processing_scale must be between 0 and 1")
        if frame_skip < 1:
            raise ValueError("frame_skip must be at least 1")

        try:
            # Model setup
            self.model = self.load_lstm(model_path)
            self.yolo = YOLO("yolo11m.pt")
            
            # Get the backbone and neck layers
            self.backbone = self.yolo.model.model[0:10]  # First 10 layers are backbone
            self.neck = self.yolo.model.model[10:15]     # Next 5 layers are neck
            
            self.threshold = threshold
            
            # GPU configuration
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logging.info(f"Using device: {self.device}")
            self.model.to(self.device)
            self.backbone.to(self.device)
            self.neck.to(self.device)
            
            # Performance optimization parameters
            self.processing_scale = processing_scale
            self.frame_skip = frame_skip
            
            # Sequence configuration
            self.window_size = window_size
            self.overlap = overlap
            self.frame_buffer = deque(maxlen=window_size)
            self.accident_reported = False
            
        except Exception as e:
            logging.error(f"Error during initialization: {str(e)}")
            raise

    def load_lstm(self, model_path):
        try:
            model = LSTM()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            logging.error(f"Error loading LSTM model: {str(e)}")
            raise

    def process_frame(self, frame):
        if frame is None:
            raise ValueError("Input frame cannot be None")
        if not isinstance(frame, np.ndarray):
            raise TypeError("Input frame must be a numpy array")
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            raise ValueError("Input frame must be a 3-channel image (BGR)")

        try:
            # Preprocess frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 640))  # YOLO default input size
            frame = frame.transpose(2, 0, 1)  # HWC to CHW
            frame = np.ascontiguousarray(frame)
            frame = torch.from_numpy(frame).to(self.device)
            frame = frame.float() / 255.0
            frame = frame.unsqueeze(0)  # Add batch dimension
            
            # Extract backbone and neck features
            with torch.no_grad():
                backbone_features = self.backbone(frame)
                neck_features = self.neck(backbone_features)
                
                # Flatten and concatenate features
                backbone_features = backbone_features.view(backbone_features.size(0), -1)
                neck_features = neck_features.view(neck_features.size(0), -1)
                
                # Convert to numpy and normalize
                backbone_np = backbone_features.cpu().numpy()
                neck_np = neck_features.cpu().numpy()
                
                backbone_np = (backbone_np - backbone_np.mean()) / (backbone_np.std() + 1e-6)
                neck_np = (neck_np - neck_np.mean()) / (neck_np.std() + 1e-6)
                
                # Create frame feature
                frame_feature = {
                    "backbone_features": backbone_np[0].tolist(),
                    "neck_features": neck_np[0].tolist()
                }
                
                return frame_feature
        except Exception as e:
            logging.error(f"Error processing frame: {str(e)}")
            raise

    def predict_accident(self):
        if len(self.frame_buffer) < self.window_size:
            logging.warning("Frame buffer not full, skipping prediction")
            return False, 0.0

        try:
            # Convert buffer to sequence
            sequence = list(self.frame_buffer)
            
            # Get model prediction
            with torch.no_grad():
                prediction, _ = self.model([sequence])  # Wrap in list to create batch dimension
                prediction = prediction.item()
                
            return prediction > self.threshold, prediction
        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            return False, 0.0

    def run(self, video_source=0):
        try:
            cap = cv2.VideoCapture(video_source)
            if not cap.isOpened():
                raise RuntimeError(f"Error opening video source: {video_source}")

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.info("End of video stream")
                    break

                # Keep original frame for display
                display_frame = frame.copy()
                
                # Process only every nth frame based on frame_skip
                if frame_count % self.frame_skip == 0:
                    try:
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
                                logging.info(f"Accident detected! Confidence: {confidence:.2f}")
                                cv2.putText(display_frame, f"ACCIDENT: {confidence:.2f}", (50, 50), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                                self.accident_reported = True
                            elif not accident_detected:
                                self.accident_reported = False
                    except Exception as e:
                        logging.error(f"Error processing frame {frame_count}: {str(e)}")
                        continue

                # Increment frame counter
                frame_count += 1

                # Show the original video frame in a window
                cv2.imshow('Live Accident Detection', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    logging.info("User requested exit")
                    break
            
        except Exception as e:
            logging.error(f"Error in video processing loop: {str(e)}")
        finally:
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