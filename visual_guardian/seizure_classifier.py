"""
Seizure Classifier Module (V3 - Dual Ensemble)

Maintains a rolling buffer of frames and creates BOTH motion-only AND temporal map
encodings for seizure detection using up to 10 EfficientNet-B0 models (5+5 ensemble).
"""

import cv2
import torch
import numpy as np
import timm
from collections import deque
from pathlib import Path


class SeizureClassifier:
    """
    Seizure classifier V3 using dual ensemble (motion + temporal).
    
    **Motion-only encoding:** R=mean_diff, G=std_diff, B=max_diff
    **Temporal map encoding:** 2D spectrogram of motion over time
    
    Uses up to 10-model ensemble (5 motion + 5 temporal) for maximum performance.
    """
    
    def __init__(self, model_path, window_frames=60, target_size=224, device='auto', 
                 use_ensemble=True, use_temporal=True):
        """
        Args:
            model_path: Path to weights directory (e.g., seizure_detection/weights/)
            window_frames: Number of frames in motion summary window (default: 60 = 2 sec @ 30fps)
            target_size: Output image size (default: 224x224)
            device: 'auto', 'cuda', or 'cpu'
            use_ensemble: If True and fold models exist, use ensemble
            use_temporal: If True, also load temporal map models (10-model ensemble)
        """
        self.window_frames = window_frames
        self.target_size = target_size
        self.use_temporal = use_temporal
        
        # Rolling frame buffer
        self.frame_buffer = deque(maxlen=window_frames)
        
        # Determine device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load models
        model_path = Path(model_path)
        self.motion_models = []
        self.temporal_models = []
        
        # Load motion-only models (fold0.pt ... fold4.pt in main dir or seizure_v3_ensemble/)
        if use_ensemble and model_path.is_dir():
            fold_paths = sorted(model_path.glob('fold*.pt'))
            if len(fold_paths) < 3:
                motion_dir = model_path / 'seizure_v3_ensemble'
                if motion_dir.exists():
                    fold_paths = sorted(motion_dir.glob('fold*.pt'))
            if len(fold_paths) >= 3:
                print(f"Loading {len(fold_paths)} motion-only models...")
                for fold_path in fold_paths:
                    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2, 
                                             drop_rate=0.5, drop_path_rate=0.2)
                    model.load_state_dict(torch.load(fold_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.motion_models.append(model)
                print(f"✓ Loaded {len(self.motion_models)} motion-only models")
            else:
                # Fallback to single best.pt
                best_path = model_path / 'best.pt'
                if best_path.exists():
                    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2, 
                                             drop_rate=0.5, drop_path_rate=0.2)
                    model.load_state_dict(torch.load(best_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.motion_models.append(model)
                    print(f"✓ Loaded single motion-only model")
        else:
            # Single model file
            model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2, 
                                     drop_rate=0.5, drop_path_rate=0.2)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            self.motion_models.append(model)
            print(f"✓ Loaded single model")
        
        # Load temporal map models (temporal/ or seizure_temporal_ensemble/)
        if use_temporal and model_path.is_dir():
            temporal_dir = model_path / 'temporal'
            temporal_fold_paths = sorted(temporal_dir.glob('fold*.pt')) if temporal_dir.exists() else []
            if len(temporal_fold_paths) < 3:
                alt = model_path / 'seizure_temporal_ensemble'
                if alt.exists():
                    temporal_fold_paths = sorted(alt.glob('fold*.pt'))
            if len(temporal_fold_paths) >= 3:
                print(f"Loading {len(temporal_fold_paths)} temporal map models...")
                for fold_path in temporal_fold_paths:
                    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2,
                                             drop_rate=0.5, drop_path_rate=0.2)
                    model.load_state_dict(torch.load(fold_path, map_location=self.device))
                    model.to(self.device)
                    model.eval()
                    self.temporal_models.append(model)
                print(f"✓ Loaded {len(self.temporal_models)} temporal map models")
                print(f"✓ Total ensemble: {len(self.motion_models) + len(self.temporal_models)} models")
            elif use_temporal:
                print(f"⚠ Temporal models not found or insufficient, using motion-only")
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
        
        # Class names (assumes ImageFolder order: normal=0, seizure=1)
        self.class_names = ['normal', 'seizure']
    
    def update(self, frame):
        """
        Add a frame to the rolling buffer
        
        Args:
            frame: numpy array (H, W, 3) in BGR format
        """
        self.frame_buffer.append(frame.copy())
    
    def normalize_channel(self, channel):
        """Apply per-channel contrast stretching"""
        mn, mx = channel.min(), channel.max()
        if mx - mn < 1e-6:
            return np.zeros_like(channel, dtype=np.uint8)
        return ((channel - mn) / (mx - mn) * 255).astype(np.uint8)
    
    def create_temporal_map(self, frames, bbox):
        """Create temporal motion map (2D spectrogram)"""
        x1, y1, x2, y2 = bbox
        
        # Crop all frames using the SAME bbox
        cropped_frames = []
        for frame in frames:
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return None
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cropped_frames.append(gray)
        
        # Compute frame-to-frame differences
        diffs = []
        for i in range(1, len(cropped_frames)):
            diff = np.abs(cropped_frames[i] - cropped_frames[i-1])
            diffs.append(diff)
        
        diffs = np.array(diffs)  # Shape: (59, H, W)
        
        # Resize each diff to target size
        resized_diffs = []
        for t in range(len(diffs)):
            resized = cv2.resize(diffs[t], (self.target_size, self.target_size))
            resized_diffs.append(resized)
        
        resized_diffs = np.array(resized_diffs)  # Shape: (59, 224, 224)
        
        # Compute row-wise average: (59, 224)
        row_averages = np.mean(resized_diffs, axis=2)
        
        # Transpose and resize to square
        temporal_map_2d = row_averages.T  # Shape: (224, 59)
        temporal_map_2d = cv2.resize(temporal_map_2d, (self.target_size, self.target_size))
        
        # Normalize
        temporal_map_normalized = self.normalize_channel(temporal_map_2d)
        
        # Stack as RGB (grayscale repeated)
        temporal_map = np.stack([temporal_map_normalized] * 3, axis=-1)
        
        return temporal_map
    
    def create_motion_summary(self, frames, bbox):
        """
        Create MOTION-ONLY summary image from a window of frames (V3)
        
        Args:
            frames: list of 60 frames
            bbox: (x1, y1, x2, y2) detected from middle frame
            
        Returns:
            motion_summary: (224, 224, 3) RGB image where:
                - R = mean absolute diff (motion intensity)
                - G = std of diffs (motion rhythmicity)
                - B = max absolute diff (peak motion burst)
            ALL channels encode motion -- NO appearance leakage.
        """
        x1, y1, x2, y2 = bbox
        
        # Crop all frames using the SAME bbox
        cropped_frames = []
        for frame in frames:
            cropped = frame[y1:y2, x1:x2]
            if cropped.size == 0:
                return None
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cropped_frames.append(gray)
        
        # Compute frame-to-frame differences
        diffs = []
        for i in range(1, len(cropped_frames)):
            diff = np.abs(cropped_frames[i] - cropped_frames[i-1])
            diffs.append(diff)
        
        diffs = np.array(diffs)  # Shape: (59, H, W)
        
        # Compute motion statistics (ALL MOTION-BASED)
        mean_diff = np.mean(diffs, axis=0)   # Average motion intensity
        std_diff = np.std(diffs, axis=0)     # Motion rhythmicity
        max_diff = np.max(diffs, axis=0)     # Peak motion burst (V3 - replaces middle frame)
        
        # Per-channel contrast stretching (critical for visibility)
        mean_diff_norm = self.normalize_channel(mean_diff)
        std_diff_norm = self.normalize_channel(std_diff)
        max_diff_norm = self.normalize_channel(max_diff)
        
        # Stack as RGB: all motion-only
        motion_summary = np.stack([mean_diff_norm, std_diff_norm, max_diff_norm], axis=-1)
        
        # Resize
        motion_summary = cv2.resize(motion_summary, (self.target_size, self.target_size))
        
        return motion_summary
    
    def preprocess(self, motion_summary):
        """
        Preprocess motion summary for inference
        
        Args:
            motion_summary: (224, 224, 3) numpy array [0-255]
            
        Returns:
            tensor: (1, 3, 224, 224) normalized tensor
        """
        # Convert to float and normalize to [0, 1]
        img = motion_summary.astype(np.float32) / 255.0
        
        # Convert to tensor (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        # Normalize with ImageNet stats
        img = (img - self.mean) / self.std
        
        # Add batch dimension
        img = img.unsqueeze(0)
        
        return img
    
    def classify(self, person_detector, padding=0.2):
        """
        Classify current buffer as seizure or normal
        
        Args:
            person_detector: PersonDetector instance (shared utility)
            padding: Bbox padding factor (default: 0.2 = 20%)
            
        Returns:
            dict with keys:
                - 'class': 'seizure' or 'normal'
                - 'confidence': probability of predicted class (0-1)
                - 'seizure_prob': probability of seizure class (0-1)
                - 'normal_prob': probability of normal class (0-1)
            Returns None if:
                - Buffer not full
                - No person detected in middle frame
                - Motion summary creation failed
        """
        # Check if buffer is full
        if len(self.frame_buffer) < self.window_frames:
            return None
        
        # Get frames as list
        frames = list(self.frame_buffer)
        
        # Detect person in MIDDLE frame
        middle_idx = len(frames) // 2
        middle_frame = frames[middle_idx]
        
        detection = person_detector.detect(middle_frame, padding=padding)
        
        if detection is None:
            return None
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            return None
        
        try:
            # Create motion-only summary
            motion_summary = self.create_motion_summary(frames, bbox)
            if motion_summary is None:
                return None
            
            # Dual ensemble inference
            with torch.no_grad():
                all_probs = []
                
                # 1. Motion-only models
                if len(self.motion_models) > 0:
                    motion_tensor = self.preprocess(motion_summary).to(self.device)
                    for model in self.motion_models:
                        output = model(motion_tensor)
                        prob = torch.softmax(output, dim=1)
                        all_probs.append(prob)
                
                # 2. Temporal map models (if available)
                if len(self.temporal_models) > 0:
                    temporal_map = self.create_temporal_map(frames, bbox)
                    if temporal_map is not None:
                        temporal_tensor = self.preprocess(temporal_map).to(self.device)
                        for model in self.temporal_models:
                            output = model(temporal_tensor)
                            prob = torch.softmax(output, dim=1)
                            all_probs.append(prob)
                
                # Average all probabilities
                if len(all_probs) == 0:
                    return None
                
                probs = torch.stack(all_probs).mean(dim=0)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
            
            return {
                'class': self.class_names[pred_idx],
                'confidence': confidence,
                'normal_prob': probs[0, 0].item(),
                'seizure_prob': probs[0, 1].item()
            }
            
        except Exception as e:
            return None
    
    def reset(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()
    
    def is_ready(self):
        """Check if buffer is full and ready to classify"""
        return len(self.frame_buffer) == self.window_frames
