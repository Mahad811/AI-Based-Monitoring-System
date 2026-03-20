"""
Create Train/Val/Test Splits for Preprocessed Data
"""

import os
import json
import random
from pathlib import Path

def create_vision_splits(dataset_root='datasets'):
    """Create train/val/test splits for vision data"""
    
    # Collect all videos
    falls_dir = os.path.join(dataset_root, 'vision/processed/falls')
    normal_dir = os.path.join(dataset_root, 'vision/processed/normal')
    
    falls_videos = [str(f) for f in Path(falls_dir).glob('*.mp4')]
    normal_videos = [str(f) for f in Path(normal_dir).glob('*.mp4')]
    
    print(f"Found {len(falls_videos)} fall videos")
    print(f"Found {len(normal_videos)} normal videos")
    
    # Shuffle
    random.seed(42)
    random.shuffle(falls_videos)
    random.shuffle(normal_videos)
    
    # Split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    # test_ratio = 0.15
    
    # Split falls
    falls_train_end = int(len(falls_videos) * train_ratio)
    falls_val_end = falls_train_end + int(len(falls_videos) * val_ratio)
    
    falls_train = falls_videos[:falls_train_end]
    falls_val = falls_videos[falls_train_end:falls_val_end]
    falls_test = falls_videos[falls_val_end:]
    
    # Split normal
    normal_train_end = int(len(normal_videos) * train_ratio)
    normal_val_end = normal_train_end + int(len(normal_videos) * val_ratio)
    
    normal_train = normal_videos[:normal_train_end]
    normal_val = normal_videos[normal_train_end:normal_val_end]
    normal_test = normal_videos[normal_val_end:]
    
    # Combine
    train = falls_train + normal_train
    val = falls_val + normal_val
    test = falls_test + normal_test
    
    # Shuffle combined sets
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    # Create labels (0=normal, 1=fall)
    train_labels = {}
    val_labels = {}
    test_labels = {}
    
    for video in train:
        train_labels[video] = 1 if 'falls' in video else 0
    for video in val:
        val_labels[video] = 1 if 'falls' in video else 0
    for video in test:
        test_labels[video] = 1 if 'falls' in video else 0
    
    # Save splits
    splits = {
        'train': {'videos': train, 'labels': train_labels},
        'val': {'videos': val, 'labels': val_labels},
        'test': {'videos': test, 'labels': test_labels},
        'stats': {
            'train_count': len(train),
            'val_count': len(val),
            'test_count': len(test),
            'total': len(train) + len(val) + len(test)
        }
    }
    
    output_file = os.path.join(dataset_root, 'vision/splits/splits.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"\n[OK] Splits created:")
    print(f"  Train: {len(train)} videos ({len(falls_train)} falls, {len(normal_train)} normal)")
    print(f"  Val:   {len(val)} videos ({len(falls_val)} falls, {len(normal_val)} normal)")
    print(f"  Test:  {len(test)} videos ({len(falls_test)} falls, {len(normal_test)} normal)")
    print(f"  Total: {len(train) + len(val) + len(test)} videos")
    print(f"\nSaved to: {output_file}")

if __name__ == '__main__':
    create_vision_splits()

