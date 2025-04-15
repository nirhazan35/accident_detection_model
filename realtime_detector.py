import cv2
import torch
import time
import numpy as np
import requests
import json
import os
from datetime import datetime
from pathlib import Path
from feature_extractor import YOLOFeatureExtractor
from LSTM import LSTM
from config import DEVICE, SEQ_LENGTH, FRAME_PROCESSING_FPS, DETECTION_THRESHOLD, ALERT_COOLDOWN, ALERT_API_ENDPOINT

class AccidentDetector:
    def __init__(self, model_path, camera_id="CCTV-1", save_clips=True):
        """Initialize the accident detector with a trained model"""
        # Initialize feature extractor
        try:
            self.feature_extractor = YOLOFeatureExtractor()
            print("Feature extractor initialized successfully")
        except Exception as e:
            print(f"Error initializing feature extractor: {str(e)}")
            raise
        
        # Initialize LSTM model
        try:
            self.model = LSTM().to(DEVICE)
            
            # Load trained model
            checkpoint = torch.load(model_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"LSTM model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading LSTM model: {str(e)}")
            raise
        
        # For FPS calculation
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
        # For detection
        self.detection_buffer = []
        self.detection_threshold = DETECTION_THRESHOLD
        self.accident_detected = False
        self.confidence = 0.0
        self.detection_time = None
        self.last_alert_time = 0
        
        # For alert system
        self.camera_id = camera_id
        self.save_clips = save_clips
        self.clips_dir = "accident_clips"
        if save_clips:
            os.makedirs(self.clips_dir, exist_ok=True)
        
        # For frame buffering (to save video before and after accident)
        self.frame_buffer = []
        self.max_buffer_size = 120  # 4 seconds at 30fps
    
    def process_frame(self, frame, timestamp=None):
        """Process a single frame for accident detection"""
        # Add timestamp if not provided
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Add to frame buffer
        self.frame_buffer.append((frame.copy(), timestamp))
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Update FPS calculation
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
        
        # Extract features
        sequence = self.feature_extractor.process_realtime_frame(frame)
        
        # Track previous detection state
        previous_detection = self.accident_detected
        
        # If we have a complete sequence, run inference
        if sequence is not None:
            with torch.no_grad():
                # Run inference
                outputs, attention = self.model(sequence)
                
                # Get confidence score
                confidence = outputs.item()
                
                # Add to detection buffer and keep last 10 detections (shorter window for recall)
                self.detection_buffer.append(confidence)
                if len(self.detection_buffer) > 10:
                    self.detection_buffer.pop(0)
                
                # Get average confidence over last detections
                avg_confidence = sum(self.detection_buffer) / len(self.detection_buffer)
                self.confidence = avg_confidence
                
                # Optimized for recall: use max confidence to detect accidents
                # This will detect accidents sooner, even if it means more false positives
                max_confidence = max(self.detection_buffer)
                
                # Detect using either average (more stable) or max (better recall) based on threshold
                self.accident_detected = max_confidence > self.detection_threshold
        
        # Draw information on frame
        self._draw_info(frame, timestamp)
        
        # Check if we need to send an alert
        if self.accident_detected and not previous_detection:
            self.detection_time = time.time()
            self._send_alert(frame, timestamp)
        
        return frame, self.accident_detected, self.confidence
    
    def _draw_info(self, frame, timestamp):
        """Draw detection information on the frame"""
        # Draw timestamp
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw camera ID
        cv2.putText(frame, f"Camera: {self.camera_id}", (frame.shape[1] - 200, frame.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw confidence
        color = (0, 255, 0)  # Green for normal
        if self.confidence > self.detection_threshold * 0.75:
            color = (0, 255, 255)  # Yellow for warning
        if self.accident_detected:
            color = (0, 0, 255)  # Red for accident
        
        cv2.putText(frame, f"Confidence: {self.confidence:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw threshold info
        cv2.putText(frame, f"Threshold: {self.detection_threshold:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw alert if accident detected
        if self.accident_detected:
            cv2.putText(frame, "ACCIDENT DETECTED!", (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            # Add red border
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
            
            # Show detection time if available
            if self.detection_time:
                elapsed = time.time() - self.detection_time
                cv2.putText(frame, f"Detected {elapsed:.1f}s ago", 
                            (frame.shape[1]//2 - 150, frame.shape[0]//2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Show "Alert Sent" message
                if elapsed < 5:  # Show for 5 seconds
                    cv2.putText(frame, "ALERT SENT!", 
                                (frame.shape[1]//2 - 100, frame.shape[0]//2 + 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    def _send_alert(self, frame, timestamp):
        """Send alert to the monitoring website"""
        current_time = time.time()
        
        # Check if we're in cooldown period
        if current_time - self.last_alert_time < ALERT_COOLDOWN:
            print(f"Alert cooldown period active. Skipping alert at {timestamp}")
            return
        
        self.last_alert_time = current_time
        
        # Save snapshot of accident
        snapshot_path = None
        if self.save_clips:
            try:
                # Create a unique filename
                filename = f"{self.camera_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                snapshot_path = os.path.join(self.clips_dir, f"{filename}_snapshot.jpg")
                clip_path = os.path.join(self.clips_dir, f"{filename}_clip.mp4")
                
                # Save snapshot
                cv2.imwrite(snapshot_path, frame)
                
                # Save video clip (2 seconds before and 4 seconds after detection)
                self._save_accident_clip(clip_path)
                
                print(f"Saved accident snapshot to {snapshot_path}")
                print(f"Saved accident clip to {clip_path}")
            except Exception as e:
                print(f"Error saving accident snapshot: {str(e)}")
        
        # Send alert to monitoring website
        try:
            # Prepare alert data
            alert_data = {
                "camera_id": self.camera_id,
                "timestamp": timestamp,
                "confidence": self.confidence,
                "snapshot_path": snapshot_path if snapshot_path else "",
                "location": "Test Location",  # Replace with actual location data
                "severity": "High" if self.confidence > 0.7 else "Medium"
            }
            
            # Send to API
            # Uncomment when you have an actual API endpoint
            # response = requests.post(ALERT_API_ENDPOINT, json=alert_data)
            # if response.status_code == 200:
            #     print(f"Alert sent successfully at {timestamp}")
            # else:
            #     print(f"Failed to send alert. Status code: {response.status_code}")
            
            # For testing, just print the alert
            print(f"ALERT! Accident detected at {timestamp} with confidence {self.confidence:.2f}")
            print(f"Alert details: {json.dumps(alert_data, indent=2)}")
            
        except Exception as e:
            print(f"Error sending alert: {str(e)}")
    
    def _save_accident_clip(self, output_path):
        """Save a video clip of the accident from the frame buffer"""
        if not self.frame_buffer:
            return
        
        # Get first frame to determine video properties
        first_frame, _ = self.frame_buffer[0]
        height, width = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
        
        # Write all frames in buffer
        for frame, _ in self.frame_buffer:
            out.write(frame)
        
        out.release()

def process_video(video_path, model_path, camera_id="CCTV-1", output_path=None, loop=False):
    """Process a video file for accident detection"""
    # Initialize detector
    detector = AccidentDetector(model_path, camera_id=camera_id)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video info: {width}x{height} at {fps}fps, {total_frames} frames")
    
    # Initialize video writer if output path is provided
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    detection_times = []
    running = True
    
    while running:
        # Reset to beginning of video if looping
        if frame_count >= total_frames and loop:
            print("End of video, looping back to start...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            detector.accident_detected = False
            detector.confidence = 0.0
            detector.detection_buffer = []
            detector.detection_time = None
        
        ret, frame = cap.read()
        if not ret:
            if loop:
                continue
            else:
                break
        
        # Skip frames to match target FPS
        frame_count += 1
        if fps > FRAME_PROCESSING_FPS and frame_count % int(fps / FRAME_PROCESSING_FPS) != 0:
            continue
        
        # Create a timestamp (simulating real-time)
        video_time = frame_count / fps
        hours = int(video_time / 3600)
        minutes = int((video_time % 3600) / 60)
        seconds = int(video_time % 60)
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        
        # Measure processing time
        start_time = time.time()
        
        # Process frame
        processed_frame, accident_detected, confidence = detector.process_frame(frame, timestamp)
        
        # Calculate processing time
        process_time = time.time() - start_time
        detection_times.append(process_time)
        
        # Write frame if writer is initialized
        if writer:
            writer.write(processed_frame)
        
        # Display frame
        cv2.imshow('Accident Detection (CCTV Simulation)', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            running = False
        elif key == ord(' '):  # Spacebar pauses
            cv2.waitKey(-1)  # Wait until any key is pressed
        elif key == ord('t'):  # Adjust threshold up
            detector.detection_threshold += 0.05
            detector.detection_threshold = min(detector.detection_threshold, 0.95)
            print(f"Threshold increased to {detector.detection_threshold:.2f}")
        elif key == ord('g'):  # Adjust threshold down
            detector.detection_threshold -= 0.05
            detector.detection_threshold = max(detector.detection_threshold, 0.05)
            print(f"Threshold decreased to {detector.detection_threshold:.2f}")
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Print processing statistics
    if detection_times:
        avg_time = sum(detection_times) / len(detection_times)
        print(f"Average processing time per frame: {avg_time*1000:.2f} ms")
        print(f"Effective FPS: {1/avg_time:.2f}")
        print(f"Total frames processed: {len(detection_times)}")

def start_webcam_detection(model_path, camera_id=0):
    """Start accident detection from webcam"""
    # Initialize detector
    detector = AccidentDetector(model_path, camera_id=f"WEBCAM-{camera_id}")
    
    # Open webcam
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        return
    
    # Process frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, accident_detected, confidence = detector.process_frame(frame)
        
        # Display frame
        cv2.imshow('Accident Detection', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('t'):  # Adjust threshold up
            detector.detection_threshold += 0.05
            detector.detection_threshold = min(detector.detection_threshold, 0.95)
            print(f"Threshold increased to {detector.detection_threshold:.2f}")
        elif key == ord('g'):  # Adjust threshold down
            detector.detection_threshold -= 0.05
            detector.detection_threshold = max(detector.detection_threshold, 0.05)
            print(f"Threshold decreased to {detector.detection_threshold:.2f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time accident detection")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model")
    parser.add_argument("--video", type=str, default=None, help="Path to video file (optional)")
    parser.add_argument("--camera_id", type=str, default="CCTV-1", help="Camera ID for logging")
    parser.add_argument("--camera", type=int, default=0, help="Camera device ID (default: 0)")
    parser.add_argument("--output", type=str, help="Path to output video (optional)")
    parser.add_argument("--loop", action="store_true", help="Loop the video indefinitely")
    
    args = parser.parse_args()
    
    if args.video:
        process_video(args.video, args.model, args.camera_id, args.output, args.loop)
    else:
        start_webcam_detection(args.model, args.camera)
