"""
SLP Dataset Preprocessing Script
Converts SLP dataset to YOLO format for normal pose training

Usage:
    python scripts/preprocess_slp.py

Output:
    - datasets/vision/processed/slp/images/ (RGB images)
    - datasets/vision/processed/slp/labels/ (YOLO format annotations)
    - datasets/vision/processed/slp/metadata.json (dataset info)
"""

import os
import json
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
import scipy.io

try:
    from tqdm import tqdm
except ImportError:
    print("[!] tqdm not installed. Install with: pip install tqdm")
    tqdm = lambda x, **kwargs: x


class SLPPreprocessor:
    """Preprocess SLP dataset for YOLO training"""
    
    def __init__(self, 
                 slp_root='datasets/vision/raw/normal/SLP',
                 output_root='datasets/vision/processed/slp'):
        self.slp_root = slp_root
        self.output_root = output_root
        
        # Create output directories
        os.makedirs(os.path.join(output_root, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_root, 'labels'), exist_ok=True)
        
        self.stats = {
            'total_images': 0,
            'subjects': 0,
            'poses': {},
            'cover_conditions': {'uncover': 0, 'cover1': 0, 'cover2': 0}
        }
    
    def load_pose_annotations(self, subject_dir):
        """Load pose annotations from joints_gt_RGB.mat"""
        mat_file = os.path.join(subject_dir, 'joints_gt_RGB.mat')
        
        if not os.path.exists(mat_file):
            return None
        
        try:
            mat_data = scipy.io.loadmat(mat_file)
            # Format: <x,y,if_occluded> x n_joints x n_frames
            joints = mat_data['joints_gt']  # Shape: (3, 14, n_frames)
            return joints
        except Exception as e:
            print(f"[!] Error loading {mat_file}: {e}")
            return None
    
    def joints_to_yolo_bbox(self, joints_frame, img_width, img_height):
        """
        Convert joint coordinates to YOLO bounding box
        
        Args:
            joints_frame: (3, 14) array [x, y, occluded]
            img_width, img_height: Image dimensions
            
        Returns:
            YOLO format: class x_center y_center width height (normalized)
        """
        # Extract x, y coordinates (ignore occlusion flag)
        x_coords = joints_frame[0, :]  # All x coordinates
        y_coords = joints_frame[1, :]  # All y coordinates
        
        # Filter out invalid/occluded points (typically marked as 0 or negative)
        valid_mask = (x_coords > 0) & (y_coords > 0)
        
        if not valid_mask.any():
            # No valid joints, use full frame as fallback
            return "0 0.5 0.5 0.8 0.8\n"
        
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        
        # Calculate bounding box
        x_min, x_max = x_valid.min(), x_valid.max()
        y_min, y_max = y_valid.min(), y_valid.max()
        
        # Add padding (10%)
        padding_x = (x_max - x_min) * 0.1
        padding_y = (y_max - y_min) * 0.1
        
        x_min = max(0, x_min - padding_x)
        x_max = min(img_width, x_max + padding_x)
        y_min = max(0, y_min - padding_y)
        y_max = min(img_height, y_max + padding_y)
        
        # Convert to YOLO format (normalized)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height
        
        # Class 0 = normal pose (person in bed)
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    
    def process_subject(self, subject_dir, subject_id):
        """Process all images for one subject"""
        rgb_dir = os.path.join(subject_dir, 'RGB')
        
        if not os.path.exists(rgb_dir):
            return 0
        
        # Load pose annotations
        joints = self.load_pose_annotations(subject_dir)
        
        processed = 0
        cover_conditions = ['uncover', 'cover1', 'cover2']
        
        for cover in cover_conditions:
            cover_path = os.path.join(rgb_dir, cover)
            
            if not os.path.exists(cover_path):
                continue
            
            # Get all images in this cover condition
            images = sorted([f for f in os.listdir(cover_path) if f.endswith('.png')])
            
            for idx, img_name in enumerate(images):
                img_path = os.path.join(cover_path, img_name)
                
                # Generate unique output name
                output_name = f"{subject_id}_{cover}_{img_name}"
                output_img = os.path.join(self.output_root, 'images', output_name)
                output_label = os.path.join(self.output_root, 'labels', 
                                           output_name.replace('.png', '.txt'))
                
                try:
                    # Copy image
                    shutil.copy2(img_path, output_img)
                    
                    # Get image dimensions
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    
                    # Create YOLO label
                    if joints is not None and idx < joints.shape[2]:
                        # Use actual pose annotations
                        joints_frame = joints[:, :, idx]
                        yolo_label = self.joints_to_yolo_bbox(joints_frame, img_width, img_height)
                    else:
                        # Fallback: assume person occupies center 80% of frame
                        yolo_label = "0 0.5 0.5 0.8 0.8\n"
                    
                    # Write label file
                    with open(output_label, 'w') as f:
                        f.write(yolo_label)
                    
                    processed += 1
                    self.stats['cover_conditions'][cover] += 1
                    
                except Exception as e:
                    print(f"[!] Error processing {img_path}: {e}")
                    continue
        
        return processed
    
    def process_all(self, max_subjects=None, sample_rate=1):
        """
        Process entire SLP dataset
        
        Args:
            max_subjects: Limit number of subjects (None = all)
            sample_rate: Sample every Nth frame (1 = all frames)
        """
        print("\n" + "="*70)
        print("SLP DATASET PREPROCESSING")
        print("="*70)
        
        # Process danaLab (main dataset)
        danalab_path = os.path.join(self.slp_root, 'danaLab')
        
        if not os.path.exists(danalab_path):
            print(f"[X] ERROR: danaLab not found at {danalab_path}")
            return
        
        # Get all subject directories
        subjects = sorted([d for d in os.listdir(danalab_path) 
                          if os.path.isdir(os.path.join(danalab_path, d)) 
                          and d.isdigit()])
        
        if max_subjects:
            subjects = subjects[:max_subjects]
        
        print(f"\n[*] Found {len(subjects)} subjects")
        print(f"[*] Sample rate: 1/{sample_rate} frames")
        print(f"[*] Output: {self.output_root}\n")
        
        # Process each subject
        for subject_id in tqdm(subjects, desc="Processing subjects"):
            subject_dir = os.path.join(danalab_path, subject_id)
            
            processed = self.process_subject(subject_dir, subject_id)
            
            if processed > 0:
                self.stats['subjects'] += 1
                self.stats['total_images'] += processed
        
        # Save metadata
        self.save_metadata()
        
        # Print summary
        self.print_summary()
    
    def save_metadata(self):
        """Save preprocessing metadata"""
        metadata = {
            'dataset': 'SLP (Simulated Lying Postures)',
            'source': 'datasets/vision/raw/normal/SLP',
            'processed_date': str(Path(__file__).stat().st_mtime),
            'statistics': self.stats,
            'format': 'YOLO (class x_center y_center width height)',
            'class_mapping': {
                '0': 'normal_pose'
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
        print(f"\nSubjects processed:  {self.stats['subjects']}")
        print(f"Total images:        {self.stats['total_images']:,}")
        print(f"\nCover conditions:")
        for cover, count in self.stats['cover_conditions'].items():
            print(f"  {cover:10s}: {count:,} images")
        
        print(f"\nOutput location:")
        print(f"  Images: {self.output_root}/images/")
        print(f"  Labels: {self.output_root}/labels/")
        
        print("\n" + "="*70)
        print("Next steps:")
        print("  1. Create train/val/test splits")
        print("  2. Merge with fall detection dataset")
        print("  3. Train YOLO model")
        print("="*70 + "\n")


def main():
    """Main preprocessing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess SLP dataset')
    parser.add_argument('--max-subjects', type=int, default=None,
                       help='Maximum number of subjects to process (default: all)')
    parser.add_argument('--sample-rate', type=int, default=1,
                       help='Sample every Nth frame (default: 1 = all frames)')
    parser.add_argument('--quick', action='store_true',
                       help='Quick test: process 5 subjects only')
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.max_subjects = 5
        print("[*] QUICK TEST MODE: Processing 5 subjects only")
    
    # Initialize preprocessor
    preprocessor = SLPPreprocessor()
    
    # Process dataset
    preprocessor.process_all(
        max_subjects=args.max_subjects,
        sample_rate=args.sample_rate
    )


if __name__ == '__main__':
    main()

