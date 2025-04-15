import os
import json
import numpy as np
import torch
from ultralytics import YOLO
import cv2
from pathlib import Path
from tqdm import tqdm
from config import DEVICE, SEQ_LENGTH, OVERLAP, FEATURES_DIR, FEATURE_SPEC, BACKBONE_LAYERS

class YOLOFeatureExtractor:
    def __init__(self):
        # Initialize YOLO model
        self.model = YOLO("yolo11m.pt").to(DEVICE)
        
        # Get backbone layers (first 11 layers)
        self.backbone = self.model.model.model[0:11]
        
        # Get specified backbone layers
        self.selected_layers = []
        for layer_idx in BACKBONE_LAYERS["use_layers"]:
            if layer_idx < 11:  # We know there are 11 backbone layers
                self.selected_layers.append(self.backbone[layer_idx])
            else:
                print(f"Warning: Layer {layer_idx} not found in backbone")
        
        if not self.selected_layers:
            raise ValueError("No valid layers selected from backbone")
        
        # Initialize feature reducer with known size
        self.feature_reducer = torch.nn.Sequential(
            torch.nn.Linear(BACKBONE_LAYERS["total_features"], FEATURE_SPEC["dimensions"]),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1)
        ).to(DEVICE)
        
        # Enable memory efficient mode
        torch.backends.cudnn.benchmark = True
        
        # For real-time processing
        self.frame_buffer = []
        self.frame_skip = 2  # Process every n-th frame for efficiency
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame for feature extraction"""
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to lower resolution for faster processing
        frame = cv2.resize(frame, (320, 320))  # Smaller resolution (half of 640x640)
        
        # Convert to tensor and add batch dimension
        frame = torch.from_numpy(frame).to(DEVICE)
        frame = frame.permute(2, 0, 1)  # HWC to CHW
        frame = frame.unsqueeze(0)  # Add batch dimension
        
        # Normalize to [0, 1]
        frame = frame.float() / 255.0
        
        return frame
    
    def extract_frame_features(self, frame):
        """Extract features from a single frame using selected layers"""
        with torch.no_grad():
            # Get features from each selected layer
            layer_features = []
            x = frame
            
            # Process through backbone layers sequentially
            for i, layer in enumerate(self.backbone):
                if i in BACKBONE_LAYERS["use_layers"]:
                    x = layer(x)
                    # Resize to target spatial size
                    spatial_idx = BACKBONE_LAYERS["use_layers"].index(i)
                    target_size = BACKBONE_LAYERS["spatial_sizes"][spatial_idx]
                    if x.shape[-1] != target_size:
                        x = torch.nn.functional.adaptive_avg_pool2d(x, (target_size, target_size))
                    
                    # Flatten the features
                    flattened = x.view(x.size(0), -1)
                    layer_features.append(flattened)
                else:
                    x = layer(x)  # Still process through non-selected layers to maintain proper channel dimensions
            
            # Concatenate features from all selected layers
            combined_features = torch.cat(layer_features, dim=1)
            
            # Check if feature sizes match expected size
            if combined_features.size(1) != BACKBONE_LAYERS["total_features"]:
                print(f"WARNING: Feature size mismatch! Got {combined_features.size(1)}, expected {BACKBONE_LAYERS['total_features']}")
                print("Adjusting feature reducer to match actual dimensions...")
                # Recreate feature reducer with correct dimensions
                self.feature_reducer = torch.nn.Sequential(
                    torch.nn.Linear(combined_features.size(1), FEATURE_SPEC["dimensions"]),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(0.1)
                ).to(DEVICE)
            
            # Reduce features
            reduced_features = self.feature_reducer(combined_features)
            
            # Convert to numpy and normalize
            features_np = reduced_features.cpu().numpy()
            if FEATURE_SPEC["normalized"]:
                features_np = (features_np - features_np.mean()) / (features_np.std() + 1e-6)
            
            # Clear memory
            del layer_features, combined_features, reduced_features
            torch.cuda.empty_cache()
            
            return {
                "features": features_np[0].tolist()
            }
    
    def process_video(self, video_path, label=None, real_time=False):
        """Process a video file and extract features
        If real_time is True, processes in a streaming fashion"""
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return None
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        features = []
        
        # Process frames
        frame_count = 0
        pbar = tqdm(total=total_frames, desc=f"Processing {video_path.name}", unit='frame')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % self.frame_skip != 0:
                pbar.update(1)
                continue
            
            # Preprocess and extract features
            processed_frame = self.preprocess_frame(frame)
            frame_features = self.extract_frame_features(processed_frame)
            features.append(frame_features)
            
            pbar.update(1)
            
            # For real-time processing, maintain a buffer of recent features
            if real_time:
                self.frame_buffer.append(frame_features)
                if len(self.frame_buffer) > SEQ_LENGTH:
                    self.frame_buffer.pop(0)
                
                # If we have enough frames for a sequence, we could run inference here
                if len(self.frame_buffer) == SEQ_LENGTH:
                    # This is where real-time inference would happen
                    pass
        
        cap.release()
        pbar.close()
        
        if not features:
            print(f"No frames extracted from: {video_path}")
            return None
        
        # For offline processing, split into sequences with overlap
        if not real_time and label is not None:
            sequences = []
            for i in range(0, len(features) - SEQ_LENGTH + 1, OVERLAP):
                seq = features[i:i+SEQ_LENGTH]
                sequences.append({
                    "frames": seq,
                    "label": label,
                    "source_video": video_path.name
                })
            return sequences
        
        return features
    
    def process_realtime_frame(self, frame):
        """Process a single frame for real-time inference"""
        processed_frame = self.preprocess_frame(frame)
        frame_features = self.extract_frame_features(processed_frame)
        
        # Add to buffer and remove oldest if necessary
        self.frame_buffer.append(frame_features)
        if len(self.frame_buffer) > SEQ_LENGTH:
            self.frame_buffer.pop(0)
        
        # Return current sequence if we have enough frames
        if len(self.frame_buffer) == SEQ_LENGTH:
            return self.frame_buffer.copy()
        
        return None

def extract_features(video_path, label, split):
    """Main function to extract features from a video"""
    print(f"Processing video: {Path(video_path).name} (label: {label}, split: {split})")
    extractor = YOLOFeatureExtractor()
    sequences = extractor.process_video(video_path, label)
    
    if sequences:
        # Create output directory based on split (train or val)
        output_dir = f"features/{split}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save sequences without verbose logging
        for idx, seq in enumerate(sequences):
            seq_filename = f"{output_dir}/seq_{Path(video_path).stem}_{idx}.npy"
            try:
                np.save(seq_filename, seq)
            except Exception as e:
                print(f"Error saving sequence {idx}: {e}")
        
        print(f"Saved {len(sequences)} sequences for {Path(video_path).name}")
        
        return {
            "video_name": Path(video_path).name,
            "total_sequences": len(sequences),
            "label": label,
            "split": split
        }
    
    return None

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
    
    # Process videos from both train and val splits
    for split in ["train", "val"]:
        print(f"\n--- Processing {split.upper()} split ---\n")
        
        # Process accident videos (label=1)
        accident_folder = f"data/{split}/accidents"
        if os.path.exists(accident_folder):
            accident_videos = get_video_files(accident_folder)
            print(f"Found {len(accident_videos)} accident videos in {split} split")
            for video_file in accident_videos:
                video_path = os.path.join(accident_folder, video_file)
                metadata = extract_features(video_path, label=1, split=split)
                if metadata:
                    all_metadata.append(metadata)
        
        # Process non-accident videos (label=0)
        non_accident_folder = f"data/{split}/non_accidents"
        if os.path.exists(non_accident_folder):
            non_accident_videos = get_video_files(non_accident_folder)
            print(f"Found {len(non_accident_videos)} non-accident videos in {split} split")
            for video_file in non_accident_videos:
                video_path = os.path.join(non_accident_folder, video_file)
                metadata = extract_features(video_path, label=0, split=split)
                if metadata:
                    all_metadata.append(metadata)
    
    # Save metadata
    with open("features/metadata.json", "w") as f:
        json.dump(all_metadata, f)
    
    print("\nFeature extraction complete!")
    print(f"Processed {len(all_metadata)} videos total")
    print(f"Metadata saved to features/metadata.json")