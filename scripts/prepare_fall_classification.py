"""
Fall Classification Dataset Preprocessing Script (V4: Motion-Only Optimized)

Creates a motion-only classification dataset from raw fall detection videos.
Each sample encodes ONLY motion (no appearance) using 12 consecutive frames:
- Red channel = mean of absolute frame differences (motion intensity)
- Green channel = acceleration pattern (rate of change in motion - captures fall dynamics)
- Blue channel = max of absolute frame differences (peak motion burst)

Key features:
- Motion-only encoding prevents appearance shortcuts (can't learn "horizontal person = fall")
- 12-frame window (~0.4s at 30fps) captures full fall trajectory without post-fall stillness
- Acceleration pattern (G channel) optimized for falls (rapid acceleration, not rhythmic)
- Video-level train/val/test splits (70/15/15) to prevent data leakage
- Person detection using YOLOv8n pretrained on COCO
- Consistent bbox cropping: detect person in middle frame, crop all 12 frames with same bbox
- Bbox padding (20%) to avoid cutting limbs during falls
- Class balancing via undersampling majority class in train split
- Robust error handling for corrupted videos, missing files, no-person cases
"""

import os
import sys
import json
import random
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


class FallDatasetPreprocessor:
    """Preprocesses raw fall videos into motion-only temporal RGB for classification (V4: optimized for falls)"""
    
    def __init__(self, raw_root, output_root, stride=5, padding=0.2, 
                 split_ratios=(0.7, 0.15, 0.15), target_size=224, window_frames=12, 
                 slp_images_dir=None, use_slp=True, seed=42):
        """
        Args:
            raw_root: Path to datasets/vision/raw/fall/
            output_root: Path to datasets/vision/fall_classification/
            stride: Sample window every N frames (center frame)
            padding: Bbox expansion factor (0.2 = 20% padding)
            split_ratios: (train, val, test) ratios
            target_size: Output image size (224x224)
            window_frames: Number of frames in motion window (default: 12 = ~0.4s @ 30fps)
            slp_images_dir: Path to processed SLP images (default: datasets/vision/processed/slp/images)
            use_slp: Whether to include SLP images for in-bed examples (default: True)
            seed: Random seed for reproducibility
        """
        self.window_frames = window_frames
        self.raw_root = Path(raw_root)
        self.output_root = Path(output_root)
        self.stride = stride
        self.padding = padding
        self.split_ratios = split_ratios
        self.target_size = target_size
        self.use_slp = use_slp
        self.slp_images_dir = Path(slp_images_dir) if slp_images_dir else Path('datasets/vision/processed/slp/images')
        self.seed = seed
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize YOLOv8n for person detection (COCO pretrained, class 0 = person)
        print("Loading YOLOv8n for person detection...")
        self.person_detector = YOLO('yolov8n.pt')
        print("✓ Model loaded\n")
        
        # Statistics tracking
        self.stats = {
            'videos_discovered': {'fall': 0, 'normal': 0},
            'slp_images_discovered': 0,
            'videos_processed': {'fall': 0, 'normal': 0},
            'slp_images_processed': 0,
            'videos_failed': {'fall': 0, 'normal': 0},
            'windows_extracted': {'train': {'fall': 0, 'normal': 0},
                                  'val': {'fall': 0, 'normal': 0},
                                  'test': {'fall': 0, 'normal': 0}},
            'windows_skipped_no_person': {'train': {'fall': 0, 'normal': 0},
                                          'val': {'fall': 0, 'normal': 0},
                                          'test': {'fall': 0, 'normal': 0}},
            'windows_before_balancing': {'train': {'fall': 0, 'normal': 0},
                                        'val': {'fall': 0, 'normal': 0},
                                        'test': {'fall': 0, 'normal': 0}},
            'processing_time': None,
            'config': {
                'stride': stride,
                'padding': padding,
                'target_size': target_size,
                'window_frames': window_frames,
                'split_ratios': split_ratios,
                'use_slp': use_slp,
                'seed': seed
            }
        }
    
    def discover_videos(self):
        """
        Discover all video files in raw dataset and SLP images
        Returns: dict with 'fall' and 'normal' lists of video paths, and 'slp_images' list
        """
        print("=" * 80)
        print("PHASE 1: VIDEO & SLP DISCOVERY")
        print("=" * 80)
        
        videos = {'fall': [], 'normal': [], 'slp_images': []}
        
        # Fall videos: raw/fall/falls/Fall/Raw_Video/*.mp4
        fall_dir = self.raw_root / 'falls' / 'Fall' / 'Raw_Video'
        if fall_dir.exists():
            fall_videos = list(fall_dir.glob('*.mp4'))
            videos['fall'] = fall_videos
            print(f"✓ Found {len(fall_videos)} fall videos in {fall_dir}")
        else:
            print(f"⚠ Fall video directory not found: {fall_dir}")
        
        # Normal videos: raw/fall/normal/No_Fall/Raw_Video/*.mp4
        normal_dir = self.raw_root / 'normal' / 'No_Fall' / 'Raw_Video'
        if normal_dir.exists():
            normal_videos = list(normal_dir.glob('*.mp4'))
            videos['normal'] = normal_videos
            print(f"✓ Found {len(normal_videos)} normal videos in {normal_dir}")
        else:
            print(f"⚠ Normal video directory not found: {normal_dir}")
        
        # SLP images: datasets/vision/processed/slp/images/*.png
        if self.use_slp and self.slp_images_dir.exists():
            slp_images = list(self.slp_images_dir.glob('*.png'))
            videos['slp_images'] = slp_images
            self.stats['slp_images_discovered'] = len(slp_images)
            print(f"✓ Found {len(slp_images)} SLP images in {self.slp_images_dir}")
        elif self.use_slp:
            print(f"⚠ SLP images directory not found: {self.slp_images_dir}")
            print(f"  SLP integration disabled. Run: python scripts/preprocess_slp.py")
        else:
            print(f"ℹ SLP integration disabled (use_slp=False)")
        
        self.stats['videos_discovered'] = {
            'fall': len(videos['fall']),
            'normal': len(videos['normal'])
        }
        
        print(f"\nTotal videos discovered: {sum(self.stats['videos_discovered'].values())}")
        if self.use_slp:
            print(f"Total SLP images discovered: {self.stats['slp_images_discovered']}")
        print()
        
        return videos
    
    def create_splits(self, videos):
        """
        Create train/val/test splits at video level (prevents data leakage)
        
        Args:
            videos: dict with 'fall' and 'normal' video lists
            
        Returns:
            splits: dict with 'train', 'val', 'test' keys, each containing 'fall' and 'normal' lists
        """
        print("=" * 80)
        print("PHASE 2: VIDEO-LEVEL SPLITTING")
        print("=" * 80)
        
        splits = {'train': {'fall': [], 'normal': []},
                 'val': {'fall': [], 'normal': []},
                 'test': {'fall': [], 'normal': []}}
        
        for class_name in ['fall', 'normal']:
            video_list = videos[class_name].copy()
            random.shuffle(video_list)
            
            n_total = len(video_list)
            n_train = int(n_total * self.split_ratios[0])
            n_val = int(n_total * self.split_ratios[1])
            # n_test = remaining videos
            
            splits['train'][class_name] = video_list[:n_train]
            splits['val'][class_name] = video_list[n_train:n_train + n_val]
            splits['test'][class_name] = video_list[n_train + n_val:]
            
            print(f"{class_name.upper()}:")
            print(f"  Train: {len(splits['train'][class_name])} videos")
            print(f"  Val:   {len(splits['val'][class_name])} videos")
            print(f"  Test:  {len(splits['test'][class_name])} videos")
        
        print()
        return splits
    
    def balance_train_split(self, splits):
        """
        Balance train split by undersampling majority class (video-level, before extraction)
        Val and test splits kept natural (no balancing)
        NOTE: Final balancing happens at window-level after extraction
        
        Args:
            splits: dict from create_splits()
            
        Returns:
            balanced_splits: same structure with balanced train split
        """
        print("=" * 80)
        print("PHASE 3: VIDEO-LEVEL BALANCING (TRAIN SPLIT - PRELIMINARY)")
        print("=" * 80)
        print("Note: Final balancing will be done at window-level after extraction")
        
        n_train_fall = len(splits['train']['fall'])
        n_train_normal = len(splits['train']['normal'])
        
        print(f"Before balancing:")
        print(f"  Train fall: {n_train_fall}")
        print(f"  Train normal: {n_train_normal}")
        print(f"  Imbalance ratio: {max(n_train_fall, n_train_normal) / min(n_train_fall, n_train_normal):.2f}:1")
        
        # Undersample majority class
        min_count = min(n_train_fall, n_train_normal)
        
        balanced_splits = splits.copy()
        balanced_splits['train'] = {
            'fall': random.sample(splits['train']['fall'], min_count),
            'normal': random.sample(splits['train']['normal'], min_count)
        }
        
        print(f"\nAfter balancing:")
        print(f"  Train fall: {len(balanced_splits['train']['fall'])}")
        print(f"  Train normal: {len(balanced_splits['train']['normal'])}")
        print(f"  Imbalance ratio: 1.00:1 (perfectly balanced at video level)")
        
        print(f"\nVal/test splits kept natural (no balancing):")
        print(f"  Val fall: {len(splits['val']['fall'])}, Val normal: {len(splits['val']['normal'])}")
        print(f"  Test fall: {len(splits['test']['fall'])}, Test normal: {len(splits['test']['normal'])}")
        print()
        
        return balanced_splits
    
    def detect_person_with_padding(self, frame):
        """
        Detect person in frame using YOLO and return padded bbox
        
        Args:
            frame: numpy array (H, W, 3)
            
        Returns:
            bbox: (x1, y1, x2, y2) with padding, clamped to frame bounds, or None if no person
        """
        results = self.person_detector(frame, verbose=False, classes=[0])  # class 0 = person
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None
        
        # Get highest confidence detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        best_idx = np.argmax(confidences)
        bbox = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]
        
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
    
    def normalize_channel(self, ch):
        """Normalize channel to [0, 255] with contrast stretching"""
        mn, mx = ch.min(), ch.max()
        if mx - mn < 1e-6:
            return np.zeros_like(ch, dtype=np.uint8)
        return ((ch - mn) / (mx - mn) * 255).astype(np.uint8)
    
    def extract_slp_windows(self, slp_images, split_name, output_dir):
        """
        Extract zero-motion windows from SLP static images
        
        Args:
            slp_images: List of SLP image paths
            split_name: 'train', 'val', or 'test'
            output_dir: Path to output directory for this split/class
            
        Returns:
            n_windows: number of windows successfully extracted
            n_skipped: number of windows skipped (no person detected)
        """
        n_windows = 0
        n_skipped = 0
        
        for img_path in tqdm(slp_images, desc=f"  Extracting SLP windows"):
            try:
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    n_skipped += 1
                    continue
                
                # Detect person
                bbox = self.detect_person_with_padding(img)
                if bbox is None:
                    n_skipped += 1
                    continue
                
                # Create zero-motion window: repeat same image 12 times
                frames_window = [img.copy() for _ in range(self.window_frames)]
                
                # Create motion-only RGB (will be all zeros = zero motion)
                motion_rgb = self.create_motion_only_rgb(frames_window, bbox)
                
                if motion_rgb is None:
                    n_skipped += 1
                    continue
                
                # Save as JPEG
                img_stem = img_path.stem
                output_filename = f"slp_{img_stem}.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), motion_rgb)
                
                n_windows += 1
                
            except Exception as e:
                n_skipped += 1
                continue
        
        return n_windows, n_skipped
    
    def create_motion_only_rgb(self, frames_window, bbox):
        """
        Create motion-only RGB image from 12 consecutive frames (V4: optimized for falls)
        
        Args:
            frames_window: list of 12 frames [frame_t-5, ..., frame_t, ..., frame_t+6]
            bbox: (x1, y1, x2, y2) detected from middle frame (frame_t)
            
        Returns:
            motion_rgb: (224, 224, 3) RGB image where ALL channels encode motion:
                - R = mean of absolute frame differences (motion intensity)
                - G = acceleration pattern (rate of change in motion - captures fall dynamics)
                - B = max of absolute frame differences (peak motion burst)
            
            NO appearance information - prevents shortcuts like "horizontal person = fall"
        """
        x1, y1, x2, y2 = bbox
        
        # Crop all frames using the SAME bbox (preserves motion within bbox)
        cropped_frames = []
        for frame in frames_window:
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
        
        diffs = np.array(diffs)  # Shape: (11, H, W)
        
        # R channel: Mean motion intensity
        mean_diff = np.mean(diffs, axis=0)
        
        # G channel: Acceleration pattern (rate of change in motion)
        # Compute second-order differences: how motion is changing
        if len(diffs) >= 2:
            # First-order: diffs[i] = |frame[i] - frame[i-1]|
            # Second-order: accel[i] = |diffs[i+1] - diffs[i]| = rate of change in motion
            accel_diffs = []
            for i in range(len(diffs) - 1):
                accel = np.abs(diffs[i+1] - diffs[i])
                accel_diffs.append(accel)
            accel_pattern = np.mean(accel_diffs, axis=0)  # Average acceleration pattern
        else:
            accel_pattern = np.zeros_like(mean_diff)
        
        # B channel: Peak motion burst
        max_diff = np.max(diffs, axis=0)
        
        # Normalize each channel with contrast stretching
        R = self.normalize_channel(mean_diff)
        G = self.normalize_channel(accel_pattern)
        B = self.normalize_channel(max_diff)
        
        # Stack as RGB: ALL channels are motion-only
        motion_rgb = np.stack([R, G, B], axis=-1)
        
        # Resize to target size
        motion_rgb = cv2.resize(motion_rgb, (self.target_size, self.target_size))
        
        return motion_rgb
    
    def process_video(self, video_path, split_name, class_name, output_dir):
        """
        Process a single video: extract motion-only temporal RGB windows (V4: 12-frame optimized)
        
        Args:
            video_path: Path to video file
            split_name: 'train', 'val', or 'test'
            class_name: 'fall' or 'normal'
            output_dir: Path to output directory for this split/class
            
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
        
        half_window = self.window_frames // 2
        if len(frames) < self.window_frames:
            # Video too short for window
            return 0, 0
        
        # Extract windows with stride
        n_windows = 0
        n_skipped = 0
        video_stem = video_path.stem
        
        for center_idx in range(half_window, len(frames) - half_window, self.stride):
            # Get window: [center-half_window, ..., center, ..., center+half_window]
            start_idx = center_idx - half_window
            end_idx = center_idx + half_window + 1
            window_frames = frames[start_idx:end_idx]
            
            if len(window_frames) != self.window_frames:
                continue
            
            # Detect person in MIDDLE frame (frame_t)
            bbox = self.detect_person_with_padding(window_frames[half_window])
            
            if bbox is None:
                n_skipped += 1
                continue
            
            # Create motion-only RGB using consistent bbox across all frames
            try:
                motion_rgb = self.create_motion_only_rgb(window_frames, bbox)
                
                if motion_rgb is None:
                    n_skipped += 1
                    continue
                
                # Save as JPEG
                output_filename = f"{video_stem}_f{center_idx:05d}.jpg"
                output_path = output_dir / output_filename
                cv2.imwrite(str(output_path), motion_rgb)
                
                n_windows += 1
            except Exception as e:
                # Handle any cropping/encoding errors
                n_skipped += 1
                continue
        
        return n_windows, n_skipped
    
    def process_all_videos(self, splits, slp_images_list=None):
        """
        Process all videos in all splits and classes, then process SLP images
        
        Args:
            splits: dict from balance_train_split()
            slp_images_list: List of SLP image paths (optional)
        """
        print("=" * 80)
        print("PHASE 4: MOTION-ONLY RGB EXTRACTION (V4: Optimized for Falls)")
        print("=" * 80)
        print(f"Config: stride={self.stride}, padding={self.padding}, target_size={self.target_size}")
        print(f"Window: {self.window_frames} frames (~{self.window_frames/30:.2f}s @ 30fps)")
        print(f"Encoding: R=mean_motion, G=acceleration_pattern, B=max_motion (motion-only, no appearance)")
        print()
        
        # Create output directories
        for split_name in ['train', 'val', 'test']:
            for class_name in ['fall', 'normal']:
                output_dir = self.output_root / split_name / class_name
                output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track window file paths for balancing
        window_files = {
            'train': {'fall': [], 'normal': []},
            'val': {'fall': [], 'normal': []},
            'test': {'fall': [], 'normal': []}
        }
        
        # Process each split
        for split_name in ['train', 'val', 'test']:
            print(f"\n{'='*80}")
            print(f"Processing {split_name.upper()} split")
            print(f"{'='*80}")
            
            for class_name in ['fall', 'normal']:
                video_list = splits[split_name][class_name]
                output_dir = self.output_root / split_name / class_name
                
                print(f"\n{class_name.upper()}: {len(video_list)} videos")
                
                pbar = tqdm(video_list, desc=f"  Extracting windows")
                
                for video_path in pbar:
                    try:
                        n_windows, n_skipped = self.process_video(
                            video_path, split_name, class_name, output_dir
                        )
                        
                        self.stats['windows_extracted'][split_name][class_name] += n_windows
                        self.stats['windows_before_balancing'][split_name][class_name] += n_windows
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
        
        # Process SLP images if enabled
        if self.use_slp and slp_images_list:
            print(f"\n{'='*80}")
            print(f"Processing SLP Images (Zero-Motion Windows)")
            print(f"{'='*80}")
            
            # Split SLP images: 70% train, 15% val, 15% test
            random.shuffle(slp_images_list)
            n_total_slp = len(slp_images_list)
            n_train_slp = int(n_total_slp * self.split_ratios[0])
            n_val_slp = int(n_total_slp * self.split_ratios[1])
            
            slp_splits = {
                'train': slp_images_list[:n_train_slp],
                'val': slp_images_list[n_train_slp:n_train_slp + n_val_slp],
                'test': slp_images_list[n_train_slp + n_val_slp:]
            }
            
            for split_name in ['train', 'val', 'test']:
                slp_images = slp_splits[split_name]
                if len(slp_images) == 0:
                    continue
                
                output_dir = self.output_root / split_name / 'normal'
                print(f"\n{split_name.upper()}: {len(slp_images)} SLP images")
                
                n_windows, n_skipped = self.extract_slp_windows(slp_images, split_name, output_dir)
                
                self.stats['windows_extracted'][split_name]['normal'] += n_windows
                self.stats['windows_before_balancing'][split_name]['normal'] += n_windows
                self.stats['slp_images_processed'] += len(slp_images)
                
                print(f"  ✓ Extracted {n_windows} SLP windows")
                print(f"    (Skipped {n_skipped} windows - no person detected)")
    
    def balance_extracted_windows(self):
        """
        Balance windows AFTER extraction (window-level balancing)
        This ensures perfect 1:1 balance regardless of video-level differences
        
        Strategy:
        1. Collect all window file paths from train split
        2. Count windows per class
        3. Randomly sample to match minimum count
        4. Delete excess windows
        """
        print("\n" + "=" * 80)
        print("PHASE 5: WINDOW-LEVEL BALANCING (TRAIN SPLIT)")
        print("=" * 80)
        
        # Get actual window counts from extracted files
        train_fall_dir = self.output_root / 'train' / 'fall'
        train_normal_dir = self.output_root / 'train' / 'normal'
        
        # Collect all window files
        fall_windows = list(train_fall_dir.glob('*.jpg')) if train_fall_dir.exists() else []
        normal_windows = list(train_normal_dir.glob('*.jpg')) if train_normal_dir.exists() else []
        
        n_fall = len(fall_windows)
        n_normal = len(normal_windows)
        
        print(f"Before window-level balancing:")
        print(f"  Train fall windows: {n_fall}")
        print(f"  Train normal windows: {n_normal}")
        print(f"  Imbalance ratio: {max(n_fall, n_normal) / min(n_fall, n_normal) if min(n_fall, n_normal) > 0 else float('inf'):.2f}:1")
        
        if n_fall == 0 or n_normal == 0:
            print("⚠ Warning: One class has zero windows. Cannot balance.")
            return
        
        # Balance: keep minimum count
        min_count = min(n_fall, n_normal)
        
        # Randomly sample to match min_count
        random.shuffle(fall_windows)
        random.shuffle(normal_windows)
        
        fall_to_keep = fall_windows[:min_count]
        normal_to_keep = normal_windows[:min_count]
        
        # Delete excess windows
        fall_to_delete = fall_windows[min_count:]
        normal_to_delete = normal_windows[min_count:]
        
        deleted_fall = 0
        deleted_normal = 0
        
        for window_file in fall_to_delete:
            try:
                window_file.unlink()
                deleted_fall += 1
            except Exception:
                pass
        
        for window_file in normal_to_delete:
            try:
                window_file.unlink()
                deleted_normal += 1
            except Exception:
                pass
        
        # Update statistics
        self.stats['windows_extracted']['train']['fall'] = len(fall_to_keep)
        self.stats['windows_extracted']['train']['normal'] = len(normal_to_keep)
        
        print(f"\nAfter window-level balancing:")
        print(f"  Train fall windows: {len(fall_to_keep)} (deleted {deleted_fall})")
        print(f"  Train normal windows: {len(normal_to_keep)} (deleted {deleted_normal})")
        print(f"  Imbalance ratio: 1.00:1 (perfectly balanced)")
        print()
    
    def save_statistics(self):
        """Save processing statistics to JSON"""
        print("\n" + "=" * 80)
        print("PHASE 6: SAVING STATISTICS")
        print("=" * 80)
        
        stats_path = self.output_root / 'stats.json'
        
        # Add totals
        self.stats['totals'] = {
            'train': {
                'fall': self.stats['windows_extracted']['train']['fall'],
                'normal': self.stats['windows_extracted']['train']['normal'],
                'total': self.stats['windows_extracted']['train']['fall'] + 
                        self.stats['windows_extracted']['train']['normal']
            },
            'val': {
                'fall': self.stats['windows_extracted']['val']['fall'],
                'normal': self.stats['windows_extracted']['val']['normal'],
                'total': self.stats['windows_extracted']['val']['fall'] + 
                        self.stats['windows_extracted']['val']['normal']
            },
            'test': {
                'fall': self.stats['windows_extracted']['test']['fall'],
                'normal': self.stats['windows_extracted']['test']['normal'],
                'total': self.stats['windows_extracted']['test']['fall'] + 
                        self.stats['windows_extracted']['test']['normal']
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"✓ Statistics saved to {stats_path}")
        
        # Print summary
        print("\n" + "=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        
        print(f"\nVideos discovered: {sum(self.stats['videos_discovered'].values())}")
        print(f"  Fall: {self.stats['videos_discovered']['fall']}")
        print(f"  Normal: {self.stats['videos_discovered']['normal']}")
        if self.use_slp:
            print(f"  SLP images: {self.stats['slp_images_discovered']}")
        
        print(f"\nVideos processed: {sum(self.stats['videos_processed'].values())}")
        print(f"  Fall: {self.stats['videos_processed']['fall']}")
        print(f"  Normal: {self.stats['videos_processed']['normal']}")
        
        print(f"\nVideos failed: {sum(self.stats['videos_failed'].values())}")
        print(f"  Fall: {self.stats['videos_failed']['fall']}")
        print(f"  Normal: {self.stats['videos_failed']['normal']}")
        
        print(f"\nWindows extracted (AFTER balancing):")
        for split in ['train', 'val', 'test']:
            print(f"  {split.upper()}:")
            print(f"    Fall: {self.stats['windows_extracted'][split]['fall']}")
            print(f"    Normal: {self.stats['windows_extracted'][split]['normal']}")
            print(f"    Total: {self.stats['totals'][split]['total']}")
            if split == 'train':
                # Show balance ratio
                fall_count = self.stats['windows_extracted'][split]['fall']
                normal_count = self.stats['windows_extracted'][split]['normal']
                if min(fall_count, normal_count) > 0:
                    ratio = max(fall_count, normal_count) / min(fall_count, normal_count)
                    print(f"    Balance ratio: {ratio:.2f}:1")
        
        print(f"\nWindows skipped (no person):")
        for split in ['train', 'val', 'test']:
            total_skipped = (self.stats['windows_skipped_no_person'][split]['fall'] + 
                           self.stats['windows_skipped_no_person'][split]['normal'])
            total_attempted = (self.stats['windows_extracted'][split]['fall'] + 
                             self.stats['windows_extracted'][split]['normal'] + 
                             total_skipped)
            skip_rate = 100.0 * total_skipped / total_attempted if total_attempted > 0 else 0
            print(f"  {split.upper()}: {total_skipped} ({skip_rate:.1f}%)")
        
        grand_total = sum(self.stats['totals'][s]['total'] for s in ['train', 'val', 'test'])
        # ~20-30 KB per JPEG → MB = grand_total * 25 / 1024
        size_mb_low = grand_total * 20 // 1024
        size_mb_high = grand_total * 30 // 1024
        print(f"\nOutput directory: {self.output_root}")
        print(f"Estimated size: ~{size_mb_low} - {size_mb_high} MB (~{size_mb_low // 1024:.1f} - {size_mb_high // 1024:.1f} GB)")
        
        print("\n" + "=" * 80)
        print("PREPROCESSING COMPLETE!")
        print("=" * 80)
        print(f"\nNext steps:")
        print(f"  1. Verify output in {self.output_root}")
        print(f"  2. Zip the folder: {self.output_root.name}.zip")
        print(f"  3. Upload to Kaggle")
        print(f"  4. Run training notebook: notebooks/train-fall-classifier.ipynb")
    
    def run(self):
        """Execute full preprocessing pipeline"""
        start_time = datetime.now()
        
        print("\n" + "=" * 80)
        print("FALL CLASSIFICATION DATASET PREPROCESSING")
        print("=" * 80)
        print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Raw data: {self.raw_root}")
        print(f"Output: {self.output_root}")
        print("=" * 80 + "\n")
        
        try:
            # Phase 1: Discover videos
            videos = self.discover_videos()
            
            if sum(self.stats['videos_discovered'].values()) == 0:
                print("ERROR: No videos found! Check raw data paths.")
                return
            
            # Phase 2: Create splits
            splits = self.create_splits(videos)
            
            # Phase 3: Balance train split (video-level, preliminary)
            balanced_splits = self.balance_train_split(splits)
            
            # Phase 4: Process videos and SLP images
            slp_images_list = videos.get('slp_images', [])
            self.process_all_videos(balanced_splits, slp_images_list)
            
            # Phase 5: Window-level balancing (AFTER extraction)
            self.balance_extracted_windows()
            
            # Phase 6: Save statistics
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
        description='Preprocess fall detection videos into temporal RGB classification dataset'
    )
    parser.add_argument(
        '--raw_root',
        type=str,
        default='datasets/vision/raw/fall',
        help='Path to raw fall data directory'
    )
    parser.add_argument(
        '--output_root',
        type=str,
        default='datasets/vision/fall_classification',
        help='Path to output classification dataset'
    )
    parser.add_argument(
        '--stride',
        type=int,
        default=5,
        help='Sample motion window every N frames (default: 5)'
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
    parser.add_argument(
        '--slp_images_dir',
        type=str,
        default='datasets/vision/processed/slp/images',
        help='Path to processed SLP images directory (default: datasets/vision/processed/slp/images)'
    )
    parser.add_argument(
        '--no_slp',
        action='store_true',
        help='Disable SLP integration (skip in-bed examples)'
    )
    
    args = parser.parse_args()
    
    preprocessor = FallDatasetPreprocessor(
        raw_root=args.raw_root,
        output_root=args.output_root,
        stride=args.stride,
        padding=args.padding,
        target_size=args.target_size,
        slp_images_dir=args.slp_images_dir,
        use_slp=not args.no_slp,
        seed=args.seed
    )
    
    preprocessor.run()


if __name__ == '__main__':
    main()
