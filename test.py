import torch
import numpy as np
import cv2
from LSTM import LSTM
from feature_extractor import YOLOFeatureExtractor
from pathlib import Path
from config import SEQ_LENGTH, DEVICE
import os
import time

# Configuration
# Find most recent model file
models_dir = "models"
if os.path.exists(models_dir) and os.listdir(models_dir):
    model_files = [f for f in os.listdir(models_dir) if f.startswith("accident_lstm_") and f.endswith(".pth")]
    if model_files:
        MODEL_PATH = os.path.join(models_dir, sorted(model_files)[-1])  # Get most recent model
        print(f"Using model: {MODEL_PATH}")
    else:
        MODEL_PATH = None
        print("No model files found in models directory")
else:
    MODEL_PATH = None
    print("Models directory not found or empty")

VIDEO_PATH = "data/test/0IPAgFHyRVI.mp4"
DETECTION_THRESHOLD = 0.4  # Fixed threshold as requested

print(f"Using device: {DEVICE.upper()}")

def add_status_label(frame, is_accident, confidence=None):
    """Add a simple status label to the frame"""
    h, w = frame.shape[:2]
    label = "ACCIDENT" if is_accident else "NON-ACCIDENT"
    color = (0, 0, 255) if is_accident else (0, 255, 0)  # Red for accident, green otherwise
    
    # Create semi-transparent background for text
    overlay = frame.copy()
    # Make background wider if showing confidence
    bg_width = 350 if confidence is not None else 210
    cv2.rectangle(overlay, (10, 10), (bg_width, 50), (0, 0, 0), -1)
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    
    # Add text
    cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Add confidence if provided
    if confidence is not None:
        conf_text = f"Conf: {confidence:.2%}"
        cv2.putText(frame, conf_text, (220, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def test_video(video_path, model):
    """Test the accident detection model on a video"""
    # Initialize video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Processing video: {video_path}")
    print(f"Video info: {total_frames} frames, {fps} FPS, {duration:.2f} seconds")
    print(f"Detection threshold: {DETECTION_THRESHOLD}")
    print("-" * 50)
    
    # Initialize feature extractor
    feature_extractor = YOLOFeatureExtractor()
    
    # Buffer to hold frame features
    frame_buffer = []
    
    # Process the video
    frame_count = 0
    skip_frames = 2  # Process every nth frame for efficiency
    start_time = time.time()
    
    # To avoid repeating detections
    last_detection_frame = -100
    min_frames_between_detections = fps * 2  # About 2 seconds
    
    # Create window
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    
    # Status tracking
    accident_status = False
    current_confidence = 0.0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Add status label to frame
        display_frame = add_status_label(frame.copy(), accident_status, current_confidence)
        
        # Display the frame
        cv2.imshow("Video", display_frame)
        
        # Check for user exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        # Extract features using the same pipeline as training
        frame_tensor = feature_extractor.preprocess_frame(frame)
        frame_features = feature_extractor.extract_frame_features(frame_tensor)
        
        # Add to buffer
        frame_buffer.append(frame_features)
        
        # Ensure buffer doesn't exceed required length
        if len(frame_buffer) > SEQ_LENGTH:
            frame_buffer.pop(0)
        
        # Only make prediction when we have enough frames
        if len(frame_buffer) == SEQ_LENGTH:
            # Make prediction
            with torch.no_grad():
                prediction, _ = model([frame_buffer])
                prediction_value = prediction.item()
                current_confidence = prediction_value
            
            # Update accident status based on current prediction
            if prediction_value > DETECTION_THRESHOLD:
                # Print detection to terminal every time it's above threshold
                timestamp = frame_count / fps
                minutes = int(timestamp // 60)
                seconds = int(timestamp % 60)
                
                if not accident_status:
                    # First detection of an accident sequence
                    print(f"ACCIDENT DETECTED at {minutes:02d}:{seconds:02d} (Frame {frame_count}) - Confidence: {prediction_value:.2%}")
                else:
                    # Continuing accident 
                    print(f"ACCIDENT ONGOING at {minutes:02d}:{seconds:02d} (Frame {frame_count}) - Confidence: {prediction_value:.2%}")
                
                accident_status = True
                last_detection_frame = frame_count
            else:
                # Change to non-accident status when prediction is below threshold
                if accident_status:
                    timestamp = frame_count / fps
                    minutes = int(timestamp // 60)
                    seconds = int(timestamp % 60)
                    print(f"ACCIDENT ENDED at {minutes:02d}:{seconds:02d} (Frame {frame_count}) - Confidence: {prediction_value:.2%}")
                
                accident_status = False
        
        # Show progress periodically
        if frame_count % (fps * 10) == 0:  # Every ~10 seconds
            progress = (frame_count / total_frames) * 100 if total_frames > 0 else 0
            elapsed = time.time() - start_time
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames, {elapsed:.1f} seconds elapsed)")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    # Completion message
    elapsed = time.time() - start_time
    print("-" * 50)
    print(f"Video processing complete in {elapsed:.2f} seconds")
    print(f"Processed {frame_count} frames")

if __name__ == "__main__":
    if MODEL_PATH:
        # Load model
        model = LSTM().to(DEVICE)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Test video
        video_path = Path(VIDEO_PATH)
        test_video(video_path, model)
    else:
        print("No model available. Train a model first using train.py")