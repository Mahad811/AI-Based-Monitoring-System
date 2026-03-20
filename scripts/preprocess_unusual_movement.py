"""
Unusual Movement (Seizure Detection) Dataset Preprocessing Script

Processes epilepsy seizure detection video dataset from Guangdong 999 Brain Hospital.

Dataset:
    - 403 seizure segments (334 tonic-clonic, 69 absence)
    - 403 non-seizure segments (normal activities)
    - 5-second video clips at various resolutions

Usage:
    python scripts/preprocess_unusual_movement.py
    python scripts/preprocess_unusual_movement.py --extract-frames
    python scripts/preprocess_unusual_movement.py --quick

Output:
    - datasets/vision/processed/unusual_movement/videos/ (normalized videos)
    - datasets/vision/processed/unusual_movement/frames/ (extracted frames)
    - datasets/vision/processed/unusual_movement/metadata.json
"""

import os
import json
import shutil
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from tqdm import tqdm
except ImportError:
    print("[!] tqdm not installed. Install with: pip install tqdm")
    tqdm = lambda x, **kwargs: x


class UnusualMovementPreprocessor:
    """Preprocess unusual movement/seizure detection dataset"""
    
    def __init__(self,
                 raw_root='datasets/vision/raw/unusual_movement/data',
                 output_root='datasets/vision/processed/unusual_movement'):
        self.raw_root = raw_root
        self.output_root = output_root
        
        # Target video specifications
        self.target_size = (640, 480)  # Width x Height
        self.target_fps = 30
        
        # Create output directories
        self.dirs = {
            'videos_normal': os.path.join(output_root, 'videos', 'normal'),
            'videos_seizure': os.path.join(output_root, 'videos', 'seizure'),
            'frames_normal': os.path.join(output_root, 'frames', 'normal'),
            'frames_seizure': os.path.join(output_root, 'frames', 'seizure')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Statistics
        self.stats = {
            'normal': {'videos': 0, 'frames': 0, 'patients': set()},
            'seizure': {'videos': 0, 'frames': 0, 'patients': set()},
            'video_properties': defaultdict(lambda: {'count': 0, 'total_duration': 0})
        }
    
    def extract_patient_info(self, filename):
        """Extract patient ID and session from filename (e.g., S47_11_282.mp4)"""
        try:
            parts = filename.split('_')
            patient_id = parts[0]  # S47
            session_id = parts[1]  # 11
            clip_id = parts[2].replace('.mp4', '')  # 282
            return patient_id, session_id, clip_id
        except:
            return None, None, None
    
    def process_video(self, video_path, output_dir, extract_frames=False, frame_rate=5):
        """
        Process a single video: normalize resolution, fps, and optionally extract frames
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save processed video
            extract_frames: If True, extract frames
            frame_rate: Extract every Nth frame (default: 5 = 6 fps at 30fps video)
        
        Returns:
            dict: Processing results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'success': False, 'error': 'Cannot open video'}
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / original_fps if original_fps > 0 else 0
        
        # Prepare output
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}.mp4")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.target_size)
        
        frames_extracted = 0
        frame_idx = 0
        
        # Create frames directory if needed
        if extract_frames:
            frames_dir = output_dir.replace('videos', 'frames')
            os.makedirs(frames_dir, exist_ok=True)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            processed_frame = cv2.resize(frame, self.target_size)
            
            # Write to output video
            out.write(processed_frame)
            
            # Extract frames if requested
            if extract_frames and frame_idx % frame_rate == 0:
                frame_path = os.path.join(frames_dir, f"{video_name}_frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, processed_frame)
                frames_extracted += 1
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return {
            'success': True,
            'input_path': video_path,
            'output_path': output_path,
            'original_fps': original_fps,
            'original_resolution': f"{width}x{height}",
            'frame_count': frame_count,
            'duration': duration,
            'frames_extracted': frames_extracted
        }
    
    def process_class(self, class_name, extract_frames=False):
        """Process all videos in a class (Normal or Seizure)"""
        class_dir = os.path.join(self.raw_root, class_name)
        output_dir = self.dirs[f'videos_{class_name.lower()}']
        
        if not os.path.exists(class_dir):
            print(f"[!] Warning: {class_dir} not found")
            return
        
        # Get all video files
        video_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.mp4')])
        
        print(f"\n[*] Processing {class_name} class: {len(video_files)} videos")
        
        for video_file in tqdm(video_files, desc=f"  {class_name}"):
            video_path = os.path.join(class_dir, video_file)
            
            # Extract patient info
            patient_id, session_id, clip_id = self.extract_patient_info(video_file)
            
            # Process video
            result = self.process_video(video_path, output_dir, extract_frames)
            
            if result['success']:
                self.stats[class_name.lower()]['videos'] += 1
                self.stats[class_name.lower()]['frames'] += result['frames_extracted']
                
                if patient_id:
                    self.stats[class_name.lower()]['patients'].add(patient_id)
                
                # Track video properties
                res_key = result['original_resolution']
                self.stats['video_properties'][res_key]['count'] += 1
                self.stats['video_properties'][res_key]['total_duration'] += result['duration']
    
    def process_all(self, extract_frames=False, quick=False):
        """
        Process entire unusual movement dataset
        
        Args:
            extract_frames: If True, extract frames from videos
            quick: If True, process only first 10 videos per class
        """
        print("\n" + "="*70)
        print("UNUSUAL MOVEMENT (SEIZURE DETECTION) PREPROCESSING")
        print("="*70)
        print(f"\nSource: {self.raw_root}")
        print(f"Output: {self.output_root}")
        print(f"Extract frames: {extract_frames}")
        print(f"Target resolution: {self.target_size[0]}x{self.target_size[1]}")
        print(f"Target FPS: {self.target_fps}")
        
        if quick:
            print("\n[*] QUICK MODE: Processing first 10 videos per class only")
        
        # Process both classes
        for class_name in ['Normal', 'Seizure']:
            if quick:
                # Quick mode: process only first 10 videos
                self.process_class_quick(class_name, extract_frames, limit=10)
            else:
                self.process_class(class_name, extract_frames)
        
        # Save metadata
        self.save_metadata()
        
        # Print summary
        self.print_summary()
    
    def process_class_quick(self, class_name, extract_frames=False, limit=10):
        """Quick processing for testing"""
        class_dir = os.path.join(self.raw_root, class_name)
        output_dir = self.dirs[f'videos_{class_name.lower()}']
        
        if not os.path.exists(class_dir):
            print(f"[!] Warning: {class_dir} not found")
            return
        
        video_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.mp4')])[:limit]
        
        print(f"\n[*] Processing {class_name} class: {len(video_files)} videos (quick mode)")
        
        for video_file in tqdm(video_files, desc=f"  {class_name}"):
            video_path = os.path.join(class_dir, video_file)
            patient_id, _, _ = self.extract_patient_info(video_file)
            
            result = self.process_video(video_path, output_dir, extract_frames)
            
            if result['success']:
                self.stats[class_name.lower()]['videos'] += 1
                self.stats[class_name.lower()]['frames'] += result['frames_extracted']
                if patient_id:
                    self.stats[class_name.lower()]['patients'].add(patient_id)
    
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, seed=42):
        """
        Create train/val/test splits with patient-level separation
        
        Ensures same patient doesn't appear in multiple splits
        """
        print("\n[*] Creating train/val/test splits...")
        
        np.random.seed(seed)
        
        # Collect all processed videos with patient info
        videos_by_patient = defaultdict(lambda: {'normal': [], 'seizure': []})
        
        for class_name in ['normal', 'seizure']:
            video_dir = self.dirs[f'videos_{class_name}']
            
            for video_file in os.listdir(video_dir):
                if not video_file.endswith('.mp4'):
                    continue
                
                patient_id, _, _ = self.extract_patient_info(video_file)
                if patient_id:
                    videos_by_patient[patient_id][class_name].append(video_file)
        
        # Split patients by video count to balance splits
        patients = list(videos_by_patient.keys())
        
        # Calculate total videos per patient
        patient_video_counts = {}
        for patient in patients:
            total = len(videos_by_patient[patient]['normal']) + len(videos_by_patient[patient]['seizure'])
            patient_video_counts[patient] = total
        
        # Sort patients by video count (helps balance)
        patients_sorted = sorted(patients, key=lambda p: patient_video_counts[p], reverse=True)
        
        # Greedy assignment to balance video counts
        train_patients = []
        val_patients = []
        test_patients = []
        train_count = 0
        val_count = 0
        test_count = 0
        total_videos = sum(patient_video_counts.values())
        
        target_train = int(total_videos * train_ratio)
        target_val = int(total_videos * val_ratio)
        
        for patient in patients_sorted:
            count = patient_video_counts[patient]
            
            # Assign to split with lowest count relative to target
            if train_count < target_train:
                train_patients.append(patient)
                train_count += count
            elif val_count < target_val:
                val_patients.append(patient)
                val_count += count
            else:
                test_patients.append(patient)
                test_count += count
        
        # Assign videos to splits
        splits = {
            'train': {'normal': [], 'seizure': []},
            'val': {'normal': [], 'seizure': []},
            'test': {'normal': [], 'seizure': []}
        }
        
        for patient in train_patients:
            splits['train']['normal'].extend(videos_by_patient[patient]['normal'])
            splits['train']['seizure'].extend(videos_by_patient[patient]['seizure'])
        
        for patient in val_patients:
            splits['val']['normal'].extend(videos_by_patient[patient]['normal'])
            splits['val']['seizure'].extend(videos_by_patient[patient]['seizure'])
        
        for patient in test_patients:
            splits['test']['normal'].extend(videos_by_patient[patient]['normal'])
            splits['test']['seizure'].extend(videos_by_patient[patient]['seizure'])
        
        # Save splits
        splits_file = os.path.join(self.output_root, 'splits.json')
        with open(splits_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"[OK] Splits saved: {splits_file}")
        print(f"\n  Patients - Train: {len(train_patients)}, Val: {len(val_patients)}, Test: {len(test_patients)}")
        print(f"  Normal videos - Train: {len(splits['train']['normal'])}, Val: {len(splits['val']['normal'])}, Test: {len(splits['test']['normal'])}")
        print(f"  Seizure videos - Train: {len(splits['train']['seizure'])}, Val: {len(splits['val']['seizure'])}, Test: {len(splits['test']['seizure'])}")
        
        return splits
    
    def save_metadata(self):
        """Save preprocessing metadata"""
        # Convert sets to lists for JSON serialization
        stats_json = {
            'normal': {
                'videos': self.stats['normal']['videos'],
                'frames': self.stats['normal']['frames'],
                'patients': list(self.stats['normal']['patients'])
            },
            'seizure': {
                'videos': self.stats['seizure']['videos'],
                'frames': self.stats['seizure']['frames'],
                'patients': list(self.stats['seizure']['patients'])
            },
            'video_properties': dict(self.stats['video_properties'])
        }
        
        metadata = {
            'dataset': 'Unusual Movement - Epilepsy Seizure Detection',
            'source': 'Guangdong 999 Brain Hospital',
            'description': 'Video-EEG monitoring dataset with seizure and non-seizure segments',
            'statistics': stats_json,
            'processing': {
                'target_resolution': f"{self.target_size[0]}x{self.target_size[1]}",
                'target_fps': self.target_fps,
                'video_duration': '5 seconds per clip'
            },
            'classes': {
                '0': 'normal',
                '1': 'seizure'
            },
            'seizure_types': {
                'tonic_clonic': '334 segments (described in paper)',
                'absence': '69 segments (described in paper)'
            }
        }
        
        metadata_path = os.path.join(self.output_root, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n[OK] Metadata saved: {metadata_path}")
    
    def print_summary(self):
        """Print processing summary"""
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        
        print(f"\nNormal (Non-seizure) videos:")
        print(f"  Videos processed: {self.stats['normal']['videos']}")
        print(f"  Frames extracted: {self.stats['normal']['frames']}")
        print(f"  Unique patients:  {len(self.stats['normal']['patients'])}")
        
        print(f"\nSeizure videos:")
        print(f"  Videos processed: {self.stats['seizure']['videos']}")
        print(f"  Frames extracted: {self.stats['seizure']['frames']}")
        print(f"  Unique patients:  {len(self.stats['seizure']['patients'])}")
        
        print(f"\nTotal:")
        print(f"  Videos: {self.stats['normal']['videos'] + self.stats['seizure']['videos']}")
        print(f"  Frames: {self.stats['normal']['frames'] + self.stats['seizure']['frames']}")
        print(f"  Patients: {len(self.stats['normal']['patients'] | self.stats['seizure']['patients'])}")
        
        if self.stats['video_properties']:
            print(f"\nOriginal video resolutions:")
            for res, data in sorted(self.stats['video_properties'].items()):
                avg_duration = data['total_duration'] / data['count'] if data['count'] > 0 else 0
                print(f"  {res}: {data['count']} videos (avg {avg_duration:.1f}s)")
        
        print(f"\nOutput location:")
        print(f"  Normal videos:  {self.dirs['videos_normal']}")
        print(f"  Seizure videos: {self.dirs['videos_seizure']}")
        print(f"  Normal frames:  {self.dirs['frames_normal']}")
        print(f"  Seizure frames: {self.dirs['frames_seizure']}")
        
        print("\n" + "="*70)
        print("Next steps:")
        print("  1. Create train/val/test splits (run with --create-splits)")
        print("  2. Train seizure detection model")
        print("  3. Integrate with visual_guardian module")
        print("="*70 + "\n")


def main():
    """Main preprocessing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess unusual movement dataset')
    parser.add_argument('--extract-frames', action='store_true',
                       help='Extract frames from videos (for frame-based models)')
    parser.add_argument('--frame-rate', type=int, default=5,
                       help='Extract every Nth frame (default: 5)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test: process 10 videos per class only')
    parser.add_argument('--create-splits', action='store_true',
                       help='Create train/val/test splits after preprocessing')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                       help='Training set ratio (default: 0.7)')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                       help='Validation set ratio (default: 0.15)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = UnusualMovementPreprocessor()
    
    # Process dataset
    preprocessor.process_all(
        extract_frames=args.extract_frames,
        quick=args.quick
    )
    
    # Create splits if requested
    if args.create_splits:
        preprocessor.create_splits(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )


if __name__ == '__main__':
    main()

