"""
Fall Detector Test Script

Tests the new vision pipeline on test videos and reports metrics.
"""

import sys
import yaml
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import argparse
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from visual_guardian import VisionPipeline


class FallDetectorTester:
    """Tests fall detection pipeline on video files"""
    
    def __init__(self, config_path, test_videos_root):
        """
        Args:
            config_path: Path to config.yaml
            test_videos_root: Path to test videos (datasets/vision/raw/fall/)
        """
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        self.test_videos_root = Path(test_videos_root)
        
        # Initialize pipeline
        print("Initializing vision pipeline...")
        self.pipeline = VisionPipeline(config['vision'])
        
        # Stats
        self.stats = {
            'videos_tested': 0,
            'videos_failed': 0,
            'per_video_results': [],
            'overall_metrics': {},
            'timing': {}
        }
    
    def discover_test_videos(self):
        """
        Discover test videos (fall and normal)
        
        Returns:
            dict with 'fall' and 'normal' lists of video paths
        """
        print("\nDiscovering test videos...")
        
        videos = {'fall': [], 'normal': []}
        
        # Fall test videos: raw/fall/falls/Fall/Raw_Video/*.mp4
        fall_dir = self.test_videos_root / 'falls' / 'Fall' / 'Raw_Video'
        if fall_dir.exists():
            fall_videos = sorted(list(fall_dir.glob('*.mp4')))
            # Take a subset for testing (e.g., every 5th video to save time)
            videos['fall'] = fall_videos[::5][:20]  # Max 20 fall videos
            print(f"Found {len(videos['fall'])} fall test videos (sampled)")
        
        # Normal test videos: raw/fall/normal/No_Fall/Raw_Video/*.mp4
        normal_dir = self.test_videos_root / 'normal' / 'No_Fall' / 'Raw_Video'
        if normal_dir.exists():
            normal_videos = sorted(list(normal_dir.glob('*.mp4')))
            videos['normal'] = normal_videos[::5][:20]  # Max 20 normal videos
            print(f"Found {len(videos['normal'])} normal test videos (sampled)")
        
        total = len(videos['fall']) + len(videos['normal'])
        print(f"Total test videos: {total}\n")
        
        return videos
    
    def test_video(self, video_path, true_label):
        """
        Test pipeline on a single video
        
        Args:
            video_path: Path to video file
            true_label: 'fall' or 'normal'
            
        Returns:
            dict with test results
        """
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return None
        
        frame_count = 0
        fall_detections = 0
        total_fall_prob = 0.0
        processing_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            start_time = datetime.now()
            event = self.pipeline.process_frame(frame)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000  # ms
            
            processing_times.append(processing_time)
            frame_count += 1
            
            if event['event_type'] == 'fall':
                fall_detections += 1
            
            total_fall_prob += event['fall_smoothed']
        
        cap.release()
        
        if frame_count == 0:
            return None
        
        # Video-level prediction: fall if >= 10% of frames detect fall
        avg_fall_prob = total_fall_prob / frame_count
        fall_frame_ratio = fall_detections / frame_count
        predicted_label = 'fall' if fall_frame_ratio >= 0.1 else 'normal'
        
        result = {
            'video': video_path.name,
            'true_label': true_label,
            'predicted_label': predicted_label,
            'correct': predicted_label == true_label,
            'frame_count': frame_count,
            'fall_detections': fall_detections,
            'fall_frame_ratio': fall_frame_ratio,
            'avg_fall_prob': avg_fall_prob,
            'avg_processing_time_ms': np.mean(processing_times),
            'fps': 1000.0 / np.mean(processing_times) if len(processing_times) > 0 else 0.0
        }
        
        return result
    
    def test_all(self, videos):
        """Test all videos and compute metrics"""
        print("=" * 80)
        print("TESTING FALL DETECTION PIPELINE")
        print("=" * 80)
        
        all_results = []
        
        for class_name in ['fall', 'normal']:
            video_list = videos[class_name]
            print(f"\nTesting {class_name.upper()} videos...")
            
            pbar = tqdm(video_list, desc=f"  Processing")
            
            for video_path in pbar:
                try:
                    result = self.test_video(video_path, class_name)
                    
                    if result:
                        all_results.append(result)
                        self.stats['videos_tested'] += 1
                        pbar.set_postfix({
                            'acc': f"{100*sum(r['correct'] for r in all_results)/len(all_results):.1f}%"
                        })
                    else:
                        self.stats['videos_failed'] += 1
                        
                except Exception as e:
                    self.stats['videos_failed'] += 1
                    pbar.write(f"    ✗ Failed: {video_path.name} ({str(e)})")
                    continue
        
        self.stats['per_video_results'] = all_results
        
        # Compute overall metrics
        self.compute_metrics(all_results)
        
        return all_results
    
    def compute_metrics(self, results):
        """Compute overall classification metrics"""
        if len(results) == 0:
            print("No results to compute metrics")
            return
        
        # Confusion matrix
        tp = sum(1 for r in results if r['true_label'] == 'fall' and r['predicted_label'] == 'fall')
        tn = sum(1 for r in results if r['true_label'] == 'normal' and r['predicted_label'] == 'normal')
        fp = sum(1 for r in results if r['true_label'] == 'normal' and r['predicted_label'] == 'fall')
        fn = sum(1 for r in results if r['true_label'] == 'fall' and r['predicted_label'] == 'normal')
        
        total = len(results)
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        # Fall-specific metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Timing
        avg_processing_time = np.mean([r['avg_processing_time_ms'] for r in results])
        avg_fps = np.mean([r['fps'] for r in results])
        
        self.stats['overall_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': {
                'true_positive': tp,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn
            }
        }
        
        self.stats['timing'] = {
            'avg_processing_time_ms': avg_processing_time,
            'avg_fps': avg_fps
        }
        
        # Print results
        print("\n" + "=" * 80)
        print("TEST RESULTS")
        print("=" * 80)
        
        print(f"\nVideos tested: {self.stats['videos_tested']}")
        print(f"Videos failed: {self.stats['videos_failed']}")
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall (Fall): {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        
        print(f"\nConfusion Matrix:")
        print(f"  TP (fall detected as fall): {tp}")
        print(f"  TN (normal detected as normal): {tn}")
        print(f"  FP (normal detected as fall): {fp}")
        print(f"  FN (fall detected as normal): {fn}")
        
        print(f"\nPerformance:")
        print(f"  Avg processing time: {avg_processing_time:.2f} ms/frame")
        print(f"  Avg FPS: {avg_fps:.2f}")
    
    def save_results(self, output_dir):
        """Save test results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / 'test_results.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Test fall detection pipeline')
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--test_videos',
        type=str,
        default='datasets/vision/raw/fall',
        help='Path to test videos root'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='fall_detection/report_eval',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Run tests
    tester = FallDetectorTester(args.config, args.test_videos)
    videos = tester.discover_test_videos()
    
    if sum(len(v) for v in videos.values()) == 0:
        print("ERROR: No test videos found!")
        return
    
    tester.test_all(videos)
    tester.save_results(args.output)


if __name__ == '__main__':
    main()
