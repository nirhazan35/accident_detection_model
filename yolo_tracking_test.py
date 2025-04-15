import cv2
import torch
from ultralytics import YOLO
from pathlib import Path
import time

def visualize_tracking(video_path, confidence=0.3, show_labels=True, process_every_n_frames=2):
    """
    Opens a video and displays YOLO tracking results at close to normal speed.
    
    Args:
        video_path: Path to the video file
        confidence: Detection confidence threshold
        show_labels: Whether to show class labels and tracking IDs
        process_every_n_frames: Only process every Nth frame (higher = faster)
    """
    # Load YOLO model
    model = YOLO("yolo11m.pt")
    
    # Classes to track (person, bicycle, car, motorcycle, bus, truck)
    CLASSES = [0, 1, 2, 3, 5, 7]
    
    # Open video
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize for faster processing (adjust the scale as needed)
    processing_scale = 0.5  # Process at half resolution
    process_width = int(width * processing_scale)
    process_height = int(height * processing_scale)
    
    print(f"Video FPS: {fps}")
    
    # Create window
    window_name = "YOLO Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    print("Controls:")
    print("  Space - Pause/resume video")
    print("  Q/Esc - Quit")
    
    # Process video frames
    paused = False
    frame_count = 0
    last_processed_frame = None
    target_frame_time = 1.0 / fps  # Time per frame in seconds
    
    while cap.isOpened():
        if not paused:
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Only process every Nth frame to improve speed
            if frame_count % process_every_n_frames == 0:
                # Resize frame for faster processing
                process_frame = cv2.resize(frame, (process_width, process_height))
                
                # Run tracking on the resized frame
                results = model.track(process_frame, persist=True, classes=CLASSES, 
                                     conf=confidence, verbose=False)
                
                # Scale bounding boxes back to original size
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    # Create a copy of the original frame for drawing
                    annotated_frame = frame.copy()
                    
                    if results[0].boxes.id is not None:
                        # Extract detection results
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        track_ids = results[0].boxes.id.cpu().numpy()
                        classes = results[0].boxes.cls.cpu().numpy()
                        
                        # Scale boxes back to original size
                        boxes[:, [0, 2]] *= (width / process_width)
                        boxes[:, [1, 3]] *= (height / process_height)
                        
                        # Draw bounding boxes
                        for i, (box, track_id, cls) in enumerate(zip(boxes, track_ids, classes)):
                            x1, y1, x2, y2 = box.astype(int)
                            class_name = results[0].names[int(cls)]
                            
                            # Generate unique color for each track ID
                            color = (hash(int(track_id)) % 256, 
                                     hash(int(track_id) * 2) % 256, 
                                     hash(int(track_id) * 3) % 256)
                            
                            # Draw rectangle and label
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            if show_labels:
                                label = f"ID:{int(track_id)} {class_name}"
                                cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    last_processed_frame = annotated_frame
            
            # If we have a processed frame, display it
            if last_processed_frame is not None:
                cv2.imshow(window_name, last_processed_frame)
            else:
                cv2.imshow(window_name, frame)  # Show original if no processed frame yet
            
            # Calculate delay to maintain original video speed
            elapsed = time.time() - start_time
            delay = max(1, int((target_frame_time - elapsed) * 1000))
        
        # Handle keyboard input
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q') or key == 27:  # q or ESC
            break
        elif key == 32:  # Space
            paused = not paused
            print("Video", "paused" if paused else "resumed")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your video path
    video_file = "data/test/ahTF2fb1MQA.mp4"
    
    # Try different values for speed optimization
    # Higher numbers = faster but less smooth tracking
    process_every_n_frames = 2  # Process every 2nd frame
    
    visualize_tracking(
        video_file, 
        confidence=0.25, 
        show_labels=True,
        process_every_n_frames=process_every_n_frames
    )