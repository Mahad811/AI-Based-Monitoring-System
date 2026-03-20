"""
Seizure Classification Dataset Preprocessing Script (V3)

Creates TWO motion-based classification datasets from seizure videos:

1. **Motion-Only Encoding** (seizure_classification/):
   - R channel = mean of absolute frame differences (motion intensity)
   - G channel = std of absolute frame differences (motion rhythmicity)
   - B channel = max of absolute frame differences (peak motion burst)
   - Per-channel contrast stretching for full 0-255 range
   - NO appearance information (prevents room/patient memorization)

2. **Temporal Motion Map** (seizure_temporal_map/):
   - 2D "spectrogram" of motion over time
   - X-axis = time (59 timesteps), Y-axis = spatial rows
   - Captures temporal structure of seizure rhythms
   - Grayscale repeated across RGB channels

Key features:
- Patient-level train/val/test splits (70/15/15) to prevent data leakage
- Sliding window extraction (2-second windows with 0.5-second stride)
- Person detection using YOLOv8n pretrained on COCO
- Consistent bbox cropping: detect person in middle frame, crop all 60 frames with same bbox
- Bbox padding (20%) to avoid cutting limbs
- Temporal augmentation (jitter + horizontal flip)
- Robust error handling for corrupted videos, missing files, no-person cases
"""

import os
import sys
import json
import random
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: ultralytics not installed. Run: pip install ultralytics")
    sys.exit(1)


class SeizureDatasetPreprocessor:
    """Preprocesses seizure videos into motion summary images for classification"""
    
    def __init__(self, video_root, output_root, window_frames=60, stride_frames=15,
                 padding=0.2, split_ratios=(0.7, 0.15, 0.15), target_size=224, seed=42,
                 create_temporal_map=True, detect_conf=0.25):
        """
        Args:
            video_root: Path to datasets/vision/processed/unusual_movement/videos/
            output_root: Base path (e.g., datasets/vision/) - will create two subdirs
            window_frames: Motion summary window size (default 60 = 2 sec @ 30fps)
            stride_frames: Stride for sliding window (default 15 = 0.5 sec)
            padding: Bbox expansion factor (0.2 = 20% padding)
            split_ratios: (train, val, test) ratios
            target_size: Output image size (224x224)
            seed: Random seed for reproducibility
            create_temporal_map: Boolean flag - if True, also creates temporal_map dataset
        """
        self.video_root = Path(video_root)
        self.output_root = Path(output_root)
        
        # V3: Create TWO output directories
        self.motion_output = self.output_root / 'seizure_classification'
        self.temporal_output = self.output_root / 'seizure_temporal_map'
        self.output_temporal_map = create_temporal_map  # Boolean flag (renamed to avoid collision with method)
        
        self.window_frames = window_frames
        self.stride_frames = stride_frames
        self.padding = padding
        self.detect_conf = detect_conf  # YOLO confidence (default 0.25)
        self.split_ratios = split_ratios
        self.target_size = target_size
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize YOLOv8n for person detection
        print("Loading YOLOv8n for person detection...")
        self.person_detector = YOLO('yolov8n.pt')
        print("✓ Model loaded\n")
        
        # Statistics tracking
        self.stats = {
            'videos_discovered': {'seizure': 0, 'normal': 0},
            'videos_processed': {'seizure': 0, 'normal': 0},
            'videos_failed': {'seizure': 0, 'normal': 0},
            'windows_extracted': {'train': {'seizure': 0, 'normal': 0},
                                 'val': {'seizure': 0, 'normal': 0},
                                 'test': {'seizure': 0, 'normal': 0}},
            'windows_skipped_no_person': {'train': {'seizure': 0, 'normal': 0},
                                         'val': {'seizure': 0, 'normal': 0},
                                         'test': {'seizure': 0, 'normal': 0}},
            'patients': {'seizure': set(), 'normal': set()},
            'processing_time': None,
            'config': {
                'window_frames': window_frames,
                'stride_frames': stride_frames,
                'padding': padding,
                'target_size': target_size,
                'split_ratios': split_ratios,
                'seed': seed
            }
        }
    
    def extract_patient_id(self, filename):
        """
        Extract patient ID from filename (e.g., 'S47_11_282.mp4' -> 'S47')
        
        Args:
            filename: Video filename
            
        Returns:
            patient_id: String like 'S47' or 'N23', or None if pattern not found
        """
        # Pattern: S<number> or N<number> at start of filename
        match = re.match(r'^([SN]\d+)', filename)
        if match:
            return match.group(1)
        return None
    
    def discover_videos(self):
        """
        Discover all video files and group by patient
        Returns: dict with 'seizure' and 'normal' lists of (video_path, patient_id) tuples
        """
        print("=" * 80)
        print("PHASE 1: VIDEO DISCOVERY & PATIENT GROUPING")
        print("=" * 80)
        
        videos = {'seizure': [], 'normal': []}
        
        # Seizure videos
        seizure_dir = self.video_root / 'seizure'
        if seizure_dir.exists():
            for video_path in seizure_dir.glob('*.mp4'):
                patient_id = self.extract_patient_id(video_path.name)
                if patient_id:
                    videos['seizure'].append((video_path, patient_id))
                    self.stats['patients']['seizure'].add(patient_id)
            print(f"✓ Found {len(videos['seizure'])} seizure videos from {len(self.stats['patients']['seizure'])} patients")
        else:
            print(f"⚠ Seizure video directory not found: {seizure_dir}")
        
        # Normal videos
        normal_dir = self.video_root / 'normal'
        if normal_dir.exists():
            for video_path in normal_dir.glob('*.mp4'):
                patient_id = self.extract_patient_id(video_path.name)
                if patient_id:
                    videos['normal'].append((video_path, patient_id))
                    self.stats['patients']['normal'].add(patient_id)
            print(f"✓ Found {len(videos['normal'])} normal videos from {len(self.stats['patients']['normal'])} patients")
        else:
            print(f"⚠ Normal video directory not found: {normal_dir}")
        
        self.stats['videos_discovered'] = {
            'seizure': len(videos['seizure']),
            'normal': len(videos['normal'])
        }
        
        print(f"\nTotal videos: {sum(self.stats['videos_discovered'].values())}")
        print(f"Total unique patients: {len(self.stats['patients']['seizure'] | self.stats['patients']['normal'])}")
        print()
        
        return videos
    
    def create_patient_level_splits(self, videos):
        """
        Create train/val/test splits at PATIENT level (prevents data leakage)
        All videos from same patient go to same split
        
        Args:
            videos: dict with 'seizure' and 'normal' lists of (video_path, patient_id) tuples
            
        Returns:
            splits: dict with 'train', 'val', 'test' keys, each containing 'seizure' and 'normal' lists
        """
        print("=" * 80)
        print("PHASE 2: PATIENT-LEVEL SPLITTING")
        print("=" * 80)
        
        splits = {'train': {'seizure': [], 'normal': []},
                 'val': {'seizure': [], 'normal': []},
                 'test': {'seizure': [], 'normal': []}}
        
        for class_name in ['seizure', 'normal']:
            # Group videos by patient
            patient_videos = defaultdict(list)
            for video_path, patient_id in videos[class_name]:
                patient_videos[patient_id].append(video_path)
            
            # Shuffle patients
            patients = list(patient_videos.keys())
            random.shuffle(patients)
            
            # Split patients
            n_patients = len(patients)
            n_train = int(n_patients * self.split_ratios[0])
            n_val = int(n_patients * self.split_ratios[1])
            
            train_patients = patients[:n_train]
            val_patients = patients[n_train:n_train + n_val]
            test_patients = patients[n_train + n_val:]
            
            # Assign videos to splits based on patient
            for patient in train_patients:
                splits['train'][class_name].extend(patient_videos[patient])
            for patient in val_patients:
                splits['val'][class_name].extend(patient_videos[patient])
            for patient in test_patients:
                splits['test'][class_name].extend(patient_videos[patient])
            
            print(f"{class_name.upper()}:")
            print(f"  Train: {len(train_patients)} patients, {len(splits['train'][class_name])} videos")
            print(f"  Val:   {len(val_patients)} patients, {len(splits['val'][class_name])} videos")
            print(f"  Test:  {len(test_patients)} patients, {len(splits['test'][class_name])} videos")
        
        print()
        return splits
    
    def detect_person_with_padding(self, frame):
        """
        Detect person in frame using YOLO and return padded bbox
        
        Args:
            frame: numpy array (H, W, 3) BGR
            
        Returns:
            bbox: (x1, y1, x2, y2) with padding, clamped to frame bounds, or None if no person
        """
        # Use explicit conf (self.detect_conf) so small/distant persons are still detected
        results = self.person_detector(frame, verbose=False, classes=[0], conf=self.detect_conf)
        
        if len(results) == 0 or results[0].boxes is None or len(results[0].boxes) == 0:
            return None
        
        # Get highest confidence detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        
        # Expand bbox by padding
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        pad_w = w * self.padding
        pad_h = h * self.padding
        
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(frame.shape[1], x2 + pad_w)
        y2_padded = min(frame.shape[0], y2 + pad_h)
        
        return (int(x1_padded), int(y1_padded), int(x2_padded), int(y2_padded))
    
    def normalize_channel(self, channel):
        """
        Apply per-channel contrast stretching to use full 0-255 range
        
        Args:
            channel: 2D numpy array (float32)
            
        Returns:
            normalized: 2D numpy array (uint8) stretched to [0, 255]
        """
        mn, mx = channel.min(), channel.max()
        if mx - mn < 1e-6:
            # No variation - return black
            return np.zeros_like(channel, dtype=np.uint8)
        # Stretch to full range
        return ((channel - mn) / (mx - mn) * 255).astype(np.uint8)
    
    def create_motion_summary(self, frames_window, bbox):
        """
        Create MOTION-ONLY summary image from a window of frames (V3)
        
        Args:
            frames_window: list of 60 frames
            bbox: (x1, y1, x2, y2) detected from middle frame
            
        Returns:
            motion_summary: (224, 224, 3) RGB image where:
                - R = mean of absolute frame differences (motion intensity)
                - G = std of absolute frame differences (motion rhythmicity)
                - B = max of absolute frame differences (peak motion burst)
            
            ALL channels encode motion -- NO appearance information.
            Per-channel contrast stretching ensures full 0-255 range utilization.
        """
        x1, y1, x2, y2 = bbox
        
        # Crop all frames using the SAME bbox (preserves motion within bbox)
        cropped_frames = []
        for frame in frames_window:
            cropped = frame[y1:y2, x1:x2]
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
        std_diff = np.std(diffs, axis=0)     # Motion variance (rhythmicity)
        max_diff = np.max(diffs, axis=0)     # Peak motion burst (NEW - replaces middle frame)
        
        # Per-channel contrast stretching (makes differences visible)
        mean_diff_norm = self.normalize_channel(mean_diff)
        std_diff_norm = self.normalize_channel(std_diff)
        max_diff_norm = self.normalize_channel(max_diff)
        
        # Stack as RGB: ALL channels are motion-only
        motion_summary = np.stack([mean_diff_norm, std_diff_norm, max_diff_norm], axis=-1)
        
        # Resize to target size
        motion_summary = cv2.resize(motion_summary, (self.target_size, self.target_size))
        
        return motion_summary
    
    def create_temporal_map(self, frames_window, bbox):
        """
        Create TEMPORAL MOTION MAP from a window of frames (V3 - Idea 2)
        
        This creates a 2D "spectrogram" of motion over time, capturing the
        TEMPORAL STRUCTURE of seizure rhythms that mean/std compression loses.
        
        Args:
            frames_window: list of 60 frames
            bbox: (x1, y1, x2, y2) detected from middle frame
            
        Returns:
            temporal_map: (224, 224, 3) RGB image where:
                - X-axis = time (59 timesteps = frame diffs)
                - Y-axis = spatial rows (224 rows after resize)
                - Pixel value = average absolute diff across that row at that timestep
                - Channels: R=G=B (grayscale repeated)
            
        For seizures: shows repeating horizontal bands (rhythmic motion).
        For normal: shows sparse/random motion.
        """
        x1, y1, x2, y2 = bbox
        
        # Crop all frames using the SAME bbox
        cropped_frames = []
        for frame in frames_window:
            cropped = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY).astype(np.float32)
            cropped_frames.append(gray)
        
        # Compute frame-to-frame differences
        diffs = []
        for i in range(1, len(cropped_frames)):
            diff = np.abs(cropped_frames[i] - cropped_frames[i-1])
            diffs.append(diff)
        
        diffs = np.array(diffs)  # Shape: (59, H, W)
        
        # Create temporal map: (rows, timesteps)
        # For each row and each timestep, compute average absolute diff
        num_timesteps = diffs.shape[0]  # 59
        
        # First resize to target height to get consistent row count
        resized_diffs = []
        for t in range(num_timesteps):
            # Resize each diff frame to target_size
            resized = cv2.resize(diffs[t], (self.target_size, self.target_size))
            resized_diffs.append(resized)
        
        resized_diffs = np.array(resized_diffs)  # Shape: (59, 224, 224)
        
        # Compute row-wise average: for each timestep, average across width (columns)
        # Result: (59, 224) - 59 timesteps, 224 rows
        row_averages = np.mean(resized_diffs, axis=2)  # Average across width
        
        # Transpose to get (224 rows, 59 timesteps) and repeat to 224 columns
        # This creates the "spectrogram" visualization
        temporal_map_2d = row_averages.T  # Shape: (224, 59)
        
        # Resize width from 59 to 224 to get square output
        temporal_map_2d = cv2.resize(temporal_map_2d, (self.target_size, self.target_size))
        
        # Normalize to full 0-255 range
        temporal_map_normalized = self.normalize_channel(temporal_map_2d)
        
        # Stack as RGB (grayscale repeated across all 3 channels)
        temporal_map = np.stack([temporal_map_normalized] * 3, axis=-1)
        
        return temporal_map
    
    def augment_window(self, frames, mode='none'):
        """
        Apply temporal augmentation to window
        
        Args:
            frames: list of frames
            mode: 'none', 'flip', 'jitter_forward', 'jitter_backward'
            
        Returns:
            augmented_frames: list of frames after augmentation
        """
        if mode == 'flip':
            # Horizontal flip
            return [cv2.flip(frame, 1) for frame in frames]
        elif mode == 'jitter_forward':
            # Shift window forward by 5 frames (if possible)
            return frames[5:] if len(frames) > 5 else frames
        elif mode == 'jitter_backward':
            # Shift window backward by 5 frames (if possible)
            return frames[:-5] if len(frames) > 5 else frames
        else:
            return frames
    
    def process_video(self, video_path, split_name, class_name, motion_dir, temporal_dir=None):
        """
        Process a single video: extract motion summary windows (and temporal maps)
        
        Args:
            video_path: Path to video file
            split_name: 'train', 'val', or 'test'
            class_name: 'seizure' or 'normal'
            motion_dir: Path to output directory for motion-only dataset
            temporal_dir: Path to output directory for temporal map dataset (optional)
            
        Returns:
            n_windows: number of windows successfully extracted
            n_skipped: number of windows skipped (no person detected)
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return 0, 0
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        
        if len(frames) < self.window_frames:
            # Video too short for window
            return 0, 0
        
        # Extract windows with stride
        n_windows = 0
        n_skipped = 0
        video_stem = video_path.stem
        
        # Augmentation modes (only for training split)
        aug_modes = ['none', 'flip'] if split_name == 'train' else ['none']
        
        for start_idx in range(0, len(frames) - self.window_frames + 1, self.stride_frames):
            window = frames[start_idx:start_idx + self.window_frames]
            
            # Process with different augmentations
            for aug_mode in aug_modes:
                aug_window = self.augment_window(window, aug_mode)
                
                if len(aug_window) < self.window_frames:
                    continue
                
                # Detect person in MIDDLE frame
                middle_idx = len(aug_window) // 2
                middle_frame = aug_window[middle_idx]
                bbox = self.detect_person_with_padding(middle_frame)
                
                if bbox is None:
                    n_skipped += 1
                    # Debug: print first skip
                    if n_skipped == 1 and n_windows == 0:
                        print(f"\n  [DEBUG] First skip: no person detected in window from {video_path.name} at frame {start_idx}")
                    continue
                
                # Create BOTH representations using consistent bbox
                try:
                    # 1. Motion-only summary (R=mean, G=std, B=max)
                    motion_summary = self.create_motion_summary(aug_window, bbox)
                    
                    aug_suffix = f"_{aug_mode}" if aug_mode != 'none' else ""
                    output_filename = f"{video_stem}_w{start_idx:04d}{aug_suffix}.jpg"
                    
                    # Save motion summary
                    motion_path = motion_dir / output_filename
                    cv2.imwrite(str(motion_path), motion_summary)
                    
                    # 2. Temporal motion map (spectrogram)
                    if temporal_dir is not None:
                        temporal_map = self.create_temporal_map(aug_window, bbox)
                        temporal_path = temporal_dir / output_filename
                        cv2.imwrite(str(temporal_path), temporal_map)
                    
                    n_windows += 1
                except Exception as e:
                    n_skipped += 1
                    # Debug: print first exception
                    if n_skipped == 1 and n_windows == 0:
                        print(f"\n  [DEBUG] First skip: exception in encoding from {video_path.name}: {type(e).__name__}: {e}")
                    continue
        
        return n_windows, n_skipped
    
    def process_all_videos(self, splits):
        """
        Process all videos in all splits and classes
        
        Args:
            splits: dict from create_patient_level_splits()
        """
        print("=" * 80)
        print("PHASE 3: MOTION SUMMARY EXTRACTION")
        print("=" * 80)
        print(f"Config: window={self.window_frames} frames, stride={self.stride_frames}, padding={self.padding}")
        print()
        
        # Create output directories for BOTH datasets
        for split_name in ['train', 'val', 'test']:
            for class_name in ['seizure', 'normal']:
                # Motion-only encoding dataset
                motion_dir = self.motion_output / split_name / class_name
                motion_dir.mkdir(parents=True, exist_ok=True)
                
                # Temporal map dataset
                if self.output_temporal_map:
                    temporal_dir = self.temporal_output / split_name / class_name
                    temporal_dir.mkdir(parents=True, exist_ok=True)
        
        # Sanity check: run person detection on first video's middle frame
        first_videos = splits['train']['seizure'][:1]
        if first_videos:
            cap = cv2.VideoCapture(str(first_videos[0]))
            if cap.isOpened():
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                mid = max(0, n_frames // 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid)
                ret, sample_frame = cap.read()
                cap.release()
                if ret and sample_frame is not None:
                    res = self.person_detector(sample_frame, verbose=False, classes=[0], conf=self.detect_conf)
                    n_boxes = len(res[0].boxes) if res and res[0].boxes is not None else 0
                    max_conf = float(res[0].boxes.conf.max()) if n_boxes > 0 else 0.0
                    print(f"Sanity check (first train video, middle frame): {n_boxes} person(s) detected, max_conf={max_conf:.2f} (conf threshold={self.detect_conf})")
                    if n_boxes == 0:
                        print("  WARNING: No person detected. Check: video has visible person? Run with --conf 0.1")
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            print(f"\n{'='*80}")
            print(f"Processing {split_name.upper()} split")
            print(f"{'='*80}")
            
            for class_name in ['seizure', 'normal']:
                video_list = splits[split_name][class_name]
                motion_dir = self.motion_output / split_name / class_name
                temporal_dir = self.temporal_output / split_name / class_name if self.create_temporal_map else None
                
                print(f"\n{class_name.upper()}: {len(video_list)} videos")
                
                pbar = tqdm(video_list, desc=f"  Extracting windows")
                
                for video_path in pbar:
                    try:
                        n_windows, n_skipped = self.process_video(
                            video_path, split_name, class_name, motion_dir, temporal_dir
                        )
                        
                        self.stats['windows_extracted'][split_name][class_name] += n_windows
                        self.stats['windows_skipped_no_person'][split_name][class_name] += n_skipped
                        self.stats['videos_processed'][class_name] += 1
                        
                        pbar.set_postfix({
                            'windows': self.stats['windows_extracted'][split_name][class_name],
                            'skipped': self.stats['windows_skipped_no_person'][split_name][class_name]
                        })
                        
                    except Exception as e:
                        self.stats['videos_failed'][class_name] += 1
                        pbar.write(f"    ✗ Failed: {video_path.name} ({str(e)})")
                        continue
                
                print(f"  ✓ Extracted {self.stats['windows_extracted'][split_name][class_name]} windows")
                print(f"    (Skipped {self.stats['windows_skipped_no_person'][split_name][class_name]} windows - no person detected)")
    
    def save_statistics(self):
        """Save processing statistics to JSON (both datasets)"""
        print("\n" + "=" * 80)
        print("PHASE 4: SAVING STATISTICS")
        print("=" * 80)
        
        # Convert sets to lists for JSON serialization
        self.stats['patients']['seizure'] = sorted(list(self.stats['patients']['seizure']))
        self.stats['patients']['normal'] = sorted(list(self.stats['patients']['normal']))
        
        # Add totals
        self.stats['totals'] = {
            'train': {
                'seizure': self.stats['windows_extracted']['train']['seizure'],
                'normal': self.stats['windows_extracted']['train']['normal'],
                'total': self.stats['windows_extracted']['train']['seizure'] + 
                        self.stats['windows_extracted']['train']['normal']
            },
            'val': {
                'seizure': self.stats['windows_extracted']['val']['seizure'],
                'normal': self.stats['windows_extracted']['val']['normal'],
                'total': self.stats['windows_extracted']['val']['seizure'] + 
                        self.stats['windows_extracted']['val']['normal']
            },
            'test': {
                'seizure': self.stats['windows_extracted']['test']['seizure'],
                'normal': self.stats['windows_extracted']['test']['normal'],
                'total': self.stats['windows_extracted']['test']['seizure'] + 
                        self.stats['windows_extracted']['test']['normal']
            }
        }
        
        # Save stats to both dataset directories
        motion_stats_path = self.motion_output / 'stats.json'
        with open(motion_stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"✓ Statistics saved to {motion_stats_path}")
        
        if self.output_temporal_map:
            temporal_stats_path = self.temporal_output / 'stats.json'
            with open(temporal_stats_path, 'w') as f:
                json.dump(self.stats, f, indent=2)
            print(f"✓ Statistics saved to {temporal_stats_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        
        print(f"\nVideos discovered: {sum(self.stats['videos_discovered'].values())}")
        print(f"  Seizure: {self.stats['videos_discovered']['seizure']}")
        print(f"  Normal: {self.stats['videos_discovered']['normal']}")
        
        print(f"\nPatients:")
        print(f"  Seizure: {len(self.stats['patients']['seizure'])}")
        print(f"  Normal: {len(self.stats['patients']['normal'])}")
        
        print(f"\nWindows extracted:")
        for split in ['train', 'val', 'test']:
            print(f"  {split.upper()}:")
            print(f"    Seizure: {self.stats['windows_extracted'][split]['seizure']}")
            print(f"    Normal: {self.stats['windows_extracted'][split]['normal']}")
            print(f"    Total: {self.stats['totals'][split]['total']}")
        
        print(f"\nWindows skipped (no person):")
        for split in ['train', 'val', 'test']:
            total_skipped = (self.stats['windows_skipped_no_person'][split]['seizure'] + 
                           self.stats['windows_skipped_no_person'][split]['normal'])
            total_attempted = (self.stats['windows_extracted'][split]['seizure'] + 
                             self.stats['windows_extracted'][split]['normal'] + 
                             total_skipped)
            skip_rate = 100.0 * total_skipped / total_attempted if total_attempted > 0 else 0
            print(f"  {split.upper()}: {total_skipped} ({skip_rate:.1f}%)")
        
        grand_total = sum(self.stats['totals'][split]['total'] for split in ['train', 'val', 'test'])
        print(f"\nOutput directories:")
        print(f"  1. Motion-only: {self.motion_output}")
        if self.output_temporal_map:
            print(f"  2. Temporal map: {self.temporal_output}")
        
        per_dataset_size = f"~{grand_total * 20 // 1024} - {grand_total * 30 // 1024} MB"
        num_datasets = 2 if self.output_temporal_map else 1
        print(f"Estimated size per dataset: {per_dataset_size}")
        print(f"Total size ({num_datasets} datasets): ~{grand_total * 20 * num_datasets // 1024} - {grand_total * 30 * num_datasets // 1024} MB")
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\nNext steps:")
        print(f"  1. Verify output in both dataset directories")
        print(f"  2. Zip both folders separately for Kaggle upload")
        print(f"  3. Upload to Kaggle as two datasets:")
        print(f"     - 'seizure-classification-v3' (motion-only)")
        print(f"     - 'seizure-temporal-map-v3' (spectrogram)")
        print(f"  4. Train 5 models on each dataset using k-fold notebooks")
        print(f"  5. Ensemble all 10 models for maximum performance")
    
    def run(self):
        """Execute full preprocessing pipeline"""
        start_time = datetime.now()
        
        print("\n" + "=" * 80)
        print("SEIZURE CLASSIFICATION DATASET PREPROCESSING (V3)")
        print("=" * 80)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Video data: {self.video_root}")
        print(f"Output:")
        print(f"  - Motion-only: {self.motion_output}")
        if self.output_temporal_map:
            print(f"  - Temporal map: {self.temporal_output}")
        print("=" * 80 + "\n")
        
        try:
            # Phase 1: Discover videos
            videos = self.discover_videos()
            
            if sum(self.stats['videos_discovered'].values()) == 0:
                print("ERROR: No videos found! Check video data paths.")
                return
            
            # Phase 2: Create patient-level splits
            splits = self.create_patient_level_splits(videos)
            
            # Phase 3: Process videos
            self.process_all_videos(splits)
            
            # Phase 4: Save statistics
            end_time = datetime.now()
            self.stats['processing_time'] = str(end_time - start_time)
            self.save_statistics()
            
            print(f"\nTotal time: {end_time - start_time}")
            
        except KeyboardInterrupt:
            print("\n\n⚠ Processing interrupted by user")
            print("Partial results saved to output directory")
            sys.exit(1)
        except Exception as e:
            print(f"\n\n✗ ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess seizure videos into motion summary classification dataset'
    )
    parser.add_argument(
        '--video_root',
        type=str,
        default='datasets/vision/processed/unusual_movement/videos',
        help='Path to seizure video directory'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='datasets/vision',
        help='Base output directory (will create seizure_classification/ and seizure_temporal_map/ subdirs)'
    )
    parser.add_argument(
        '--skip_temporal_map',
        action='store_true',
        help='Skip creating temporal map dataset (only create motion-only dataset)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='YOLO person detection confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--window_frames',
        type=int,
        default=60,
        help='Motion summary window size in frames (default: 60 = 2 sec @ 30fps)'
    )
    parser.add_argument(
        '--stride_frames',
        type=int,
        default=15,
        help='Stride for sliding window in frames (default: 15 = 0.5 sec)'
    )
    parser.add_argument(
        '--padding',
        type=float,
        default=0.2,
        help='Bbox expansion factor (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--target_size',
        type=int,
        default=224,
        help='Output image size (default: 224)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    preprocessor = SeizureDatasetPreprocessor(
        video_root=args.video_root,
        output_root=args.output_root,
        window_frames=args.window_frames,
        stride_frames=args.stride_frames,
        padding=args.padding,
        target_size=args.target_size,
        seed=args.seed,
        create_temporal_map=not args.skip_temporal_map,
        detect_conf=args.conf
    )
    
    preprocessor.run()


if __name__ == '__main__':
    main()
