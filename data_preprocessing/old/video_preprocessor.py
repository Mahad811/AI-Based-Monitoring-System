"""
Video Preprocessing Pipeline
Processes raw video data for fall detection training
"""

import cv2
import os
import numpy as np
from pathlib import Path
import json


class VideoPreprocessor:
    """Preprocess videos for fall detection"""
    
    def __init__(self, config):
        """
        Initialize video preprocessor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.target_size = (640, 480)  # Width x Height
        self.target_fps = 30
        
    def process_video(self, video_path, output_dir, extract_frames=False):
        """
        Process a single video
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save processed video
            extract_frames: If True, extract and save individual frames
            
        Returns:
            dict: Processing results
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            return {'success': False, 'error': 'Cannot open video'}
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Prepare output
        video_name = Path(video_path).stem
        output_path = os.path.join(output_dir, f"{video_name}_processed.mp4")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, self.target_fps, self.target_size)
        
        frames_saved = 0
        
        # Create frames directory if needed
        if extract_frames:
            frames_dir = os.path.join(output_dir, 'frames', video_name)
            os.makedirs(frames_dir, exist_ok=True)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            processed_frame = cv2.resize(frame, self.target_size)
            
            # Write to output video
            out.write(processed_frame)
            
            # Extract frames if requested
            if extract_frames and frame_idx % 10 == 0:  # Save every 10th frame
                frame_path = os.path.join(frames_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(frame_path, processed_frame)
                frames_saved += 1
            
            frame_idx += 1
        
        cap.release()
        out.release()
        
        return {
            'success': True,
            'input_path': video_path,
            'output_path': output_path,
            'original_fps': original_fps,
            'frame_count': frame_count,
            'frames_extracted': frames_saved if extract_frames else 0
        }
    
    def process_directory(self, input_dir, output_dir, extract_frames=False):
        """
        Process all videos in a directory
        
        Args:
            input_dir: Input directory containing videos
            output_dir: Output directory for processed videos
            extract_frames: Whether to extract frames
            
        Returns:
            list: Processing results for all videos
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Supported video formats
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        results = []
        
        for video_file in Path(input_dir).rglob('*'):
            if video_file.suffix.lower() in video_extensions:
                print(f"Processing: {video_file.name}")
                result = self.process_video(
                    str(video_file), 
                    output_dir, 
                    extract_frames
                )
                results.append(result)
        
        return results
    
    def create_train_val_test_split(self, dataset_dir, output_file, 
                                    train_ratio=0.7, val_ratio=0.15):
        """
        Create train/val/test splits
        
        Args:
            dataset_dir: Directory containing processed videos
            output_file: Path to save split information
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            
        Returns:
            dict: Split statistics
        """
        video_files = []
        for video_file in Path(dataset_dir).glob('*.mp4'):
            video_files.append(str(video_file))
        
        # Shuffle
        np.random.shuffle(video_files)
        
        # Calculate split indices
        total = len(video_files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)
        
        # Split
        train_files = video_files[:train_end]
        val_files = video_files[train_end:val_end]
        test_files = video_files[val_end:]
        
        # Save splits
        splits = {
            'train': train_files,
            'val': val_files,
            'test': test_files
        }
        
        with open(output_file, 'w') as f:
            json.dump(splits, f, indent=2)
        
        return {
            'total': total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }
    
    def augment_video(self, video_path, output_dir, augmentations=None):
        """
        Apply data augmentation to video
        
        Args:
            video_path: Input video path
            output_dir: Output directory
            augmentations: List of augmentations to apply
                         ['flip', 'brightness', 'noise']
        """
        if augmentations is None:
            augmentations = ['flip']
        
        cap = cv2.VideoCapture(video_path)
        video_name = Path(video_path).stem
        
        os.makedirs(output_dir, exist_ok=True)
        
        for aug in augmentations:
            output_path = os.path.join(output_dir, f"{video_name}_{aug}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 30, self.target_size)
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.resize(frame, self.target_size)
                
                # Apply augmentation
                if aug == 'flip':
                    frame = cv2.flip(frame, 1)  # Horizontal flip
                elif aug == 'brightness':
                    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)
                elif aug == 'noise':
                    noise = np.random.randint(0, 20, frame.shape, dtype='uint8')
                    frame = cv2.add(frame, noise)
                
                out.write(frame)
            
            out.release()
        
        cap.release()

