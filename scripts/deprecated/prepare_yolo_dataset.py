"""
Prepare Dataset for YOLOv8 Training
Extracts frames from videos and creates YOLO format labels

Usage:
    python scripts/prepare_yolo_dataset.py

Output:
    - datasets/vision/yolo/train/images/ (frames)
    - datasets/vision/yolo/train/labels/ (YOLO annotations)
    - datasets/vision/yolo/val/images/
    - datasets/vision/yolo/val/labels/
    - datasets/vision/yolo/test/images/
    - datasets/vision/yolo/test/labels/
"""

import cv2
import os
import json
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    print("[!] tqdm not installed. Install with: pip install tqdm")
    print("[!] Continuing without progress bars...")
    tqdm = lambda x, **kwargs: x

def extract_frames(video_path, output_dir, max_frames=10, label=None):
    """
    Extract frames from video
    
    Args:
        video_path: Path to video
        output_dir: Where to save frames
        max_frames: Max frames to extract per video
        label: Class label (0=normal, 1=fall)
    """
    try:
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"[!] Failed to open: {video_path}")
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            print(f"[!] No frames in: {video_path}")
            cap.release()
            return []
        
        # Sample frames evenly
        frame_indices = [int(i * total_frames / max_frames) for i in range(max_frames)]
        
        video_name = Path(video_path).stem
        extracted = []
        
        for i, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Save frame
            frame_name = f"{video_name}_f{i:03d}.jpg"
            frame_path = os.path.join(output_dir, 'images', frame_name)
            
            try:
                cv2.imwrite(frame_path, frame)
                
                # Create YOLO label (full frame = person detected)
                if label is not None:
                    label_path = os.path.join(output_dir, 'labels', frame_name.replace('.jpg', '.txt'))
                    # YOLO format: class x_center y_center width height (normalized 0-1)
                    # Assume person fills ~80% of frame center
                    with open(label_path, 'w') as f:
                        f.write(f"{label} 0.5 0.5 0.8 0.8\n")
                
                extracted.append(frame_name)
            except Exception as e:
                print(f"[!] Error saving frame {frame_name}: {e}")
                continue
        
        cap.release()
        return extracted
    
    except Exception as e:
        print(f"[!] Error processing {video_path}: {e}")
        return []

def prepare_yolo_dataset(dataset_root='datasets'):
    """Prepare complete YOLO dataset from processed videos"""
    
    print("\n" + "="*70)
    print("PREPARING YOLO DATASET FOR TRAINING")
    print("="*70)
    
    # Load splits
    splits_file = os.path.join(dataset_root, 'vision/splits/splits.json')
    
    if not os.path.exists(splits_file):
        print(f"\n[X] ERROR: Splits file not found: {splits_file}")
        print("[!] Run: python scripts/create_splits.py first")
        return
    
    print(f"\n[*] Loading splits from: {splits_file}")
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    
    # Create output directories
    print("\n[*] Creating YOLO directory structure...")
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_root, 'vision/yolo', split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dataset_root, 'vision/yolo', split, 'labels'), exist_ok=True)
    print("[OK] Directories created")
    
    # Statistics
    stats = {}
    
    # Process each split
    for split_name in ['train', 'val', 'test']:
        print(f"\n" + "-"*70)
        print(f"[*] Processing {split_name.upper()} split")
        print("-"*70)
        
        videos = splits[split_name]['videos']
        labels = splits[split_name]['labels']
        
        print(f"[*] Total videos: {len(videos)}")
        
        output_dir = os.path.join(dataset_root, 'vision/yolo', split_name)
        
        total_frames = 0
        failed_videos = 0
        
        for video_path in tqdm(videos, desc=f"Extracting {split_name}"):
            label = labels[video_path]
            frames = extract_frames(video_path, output_dir, max_frames=10, label=label)
            
            if len(frames) == 0:
                failed_videos += 1
            else:
                total_frames += len(frames)
        
        stats[split_name] = {
            'videos': len(videos),
            'frames': total_frames,
            'failed': failed_videos
        }
        
        print(f"\n[OK] {split_name.upper()} complete:")
        print(f"    Videos processed: {len(videos) - failed_videos}/{len(videos)}")
        print(f"    Frames extracted: {total_frames}")
        if failed_videos > 0:
            print(f"    [!] Failed videos: {failed_videos}")
    
    # Final summary
    print("\n" + "="*70)
    print("YOLO DATASET PREPARATION COMPLETE")
    print("="*70)
    print(f"\nLocation: {dataset_root}/vision/yolo/\n")
    print(f"Train:  {stats['train']['frames']:,} frames from {stats['train']['videos']:,} videos")
    print(f"Val:    {stats['val']['frames']:,} frames from {stats['val']['videos']:,} videos")
    print(f"Test:   {stats['test']['frames']:,} frames from {stats['test']['videos']:,} videos")
    print(f"\nTotal:  {sum(s['frames'] for s in stats.values()):,} frames")
    print("\n" + "="*70)
    print("\nNext steps:")
    print("  1. Compress datasets/vision/yolo/ folder")
    print("  2. Upload to Kaggle as dataset")
    print("  3. Follow KAGGLE_TRAINING_GUIDE.md")
    print("="*70 + "\n")

if __name__ == '__main__':
    prepare_yolo_dataset()

