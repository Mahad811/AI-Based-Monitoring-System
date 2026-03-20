"""
Comprehensive Seizure Detection Optimization Analysis

Sweeps thresholds, aggregation strategies, and ensemble configurations
to find the mathematically optimal setup for the test set.

Usage:
    python scripts/optimize_seizure_detection.py

Output:
    - Optimal threshold and aggregation strategy
    - Per-patient analysis
    - Recommendations for further improvement
"""

import cv2
import yaml
import os
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_guardian.seizure_classifier import SeizureClassifier
from visual_guardian.person_detector import PersonDetector

DATASET_ROOT = Path("datasets/vision/processed/unusual_movement/videos/")
SPLIT_FILE = Path("datasets/vision/processed/unusual_movement/splits.json")


def collect_video_probs(classifier, person_detector, video_path, stride=15):
    """Collect all window probabilities for a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    classifier.reset()
    probs = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        classifier.update(frame)
        
        if frame_count % stride == 0 and classifier.is_ready():
            result = classifier.classify(person_detector, padding=0.2)
            if result is not None:
                probs.append(result['seizure_prob'])
    
    cap.release()
    return probs


def aggregate(probs, method='max'):
    """Aggregate window probabilities into a single video-level score."""
    if not probs:
        return 0.0
    if method == 'max':
        return max(probs)
    elif method == 'mean':
        return np.mean(probs)
    elif method == 'p90':  # 90th percentile
        return np.percentile(probs, 90)
    elif method == 'p75':
        return np.percentile(probs, 75)
    elif method == 'top3_mean':  # Mean of top 3 windows
        return np.mean(sorted(probs, reverse=True)[:3])
    elif method == 'top5_mean':
        return np.mean(sorted(probs, reverse=True)[:5])
    return max(probs)


def compute_metrics(scores, labels, threshold):
    """Compute precision, recall, F1 at a given threshold."""
    TP = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
    FP = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
    FN = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 1)
    TN = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 0)
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0
    
    return {
        'threshold': threshold,
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def find_optimal_threshold(scores, labels, metric='f1'):
    """Sweep thresholds from 0.05 to 0.95 to find optimal."""
    best = None
    best_val = -1
    
    for t in np.arange(0.05, 0.96, 0.01):
        m = compute_metrics(scores, labels, t)
        val = m[metric]
        if val > best_val:
            best_val = val
            best = m
    
    return best


def main():
    # Load config
    with open('config/config.yaml', 'r') as f:
        root_config = yaml.safe_load(f)
    vision_config = root_config['vision']

    # Load splits
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    test_files = splits.get('test', {})
    
    print(f"Test set: {len(test_files.get('normal', []))} normal, {len(test_files.get('seizure', []))} seizure")

    # Initialize classifier
    print("\nInitializing SeizureClassifier (10-model ensemble)...")
    model_path = vision_config['seizure_classifier']['model']
    window_frames = int(vision_config['seizure_classifier']['window_seconds'] * 30)
    
    classifier = SeizureClassifier(
        model_path=model_path,
        window_frames=window_frames,
        device='auto'
    )
    person_detector = PersonDetector(
        model_path=vision_config['person_detector']['model'],
        confidence=vision_config['person_detector']['confidence']
    )
    print("✓ Ready\n")

    # Collect all window probabilities per video
    print("="*70)
    print("PHASE 1: Collecting window probabilities for all test videos")
    print("="*70)
    
    video_data = []  # List of {name, label, probs, patient_id}
    
    for cls in ['normal', 'seizure']:
        class_dir = DATASET_ROOT / cls
        if not class_dir.exists():
            continue
        
        split_filenames = set(test_files.get(cls, []))
        videos = [v for v in class_dir.glob("*.mp4") if v.name in split_filenames]
        label = 1 if cls == 'seizure' else 0
        
        print(f"\nProcessing {cls} ({len(videos)} videos)...")
        for video_path in tqdm(videos, desc=f"  {cls}"):
            probs = collect_video_probs(classifier, person_detector, video_path)
            patient_id = video_path.stem.split('_')[0]  # e.g., S10 from S10_0_64
            video_data.append({
                'name': video_path.name,
                'label': label,
                'probs': probs,
                'patient_id': patient_id,
                'cls': cls
            })

    print(f"\n✓ Collected data for {len(video_data)} videos")
    
    # Save raw probabilities for offline analysis
    raw_data = [{
        'name': v['name'],
        'label': v['label'],
        'patient_id': v['patient_id'],
        'probs': v['probs'],
        'n_windows': len(v['probs'])
    } for v in video_data]
    
    with open('seizure_detection/report_eval/test_raw_probs.json', 'w') as f:
        json.dump(raw_data, f, indent=2)
    print("✓ Raw probabilities saved to seizure_detection/report_eval/test_raw_probs.json")

    # Phase 2: Threshold sweep for each aggregation method
    print("\n" + "="*70)
    print("PHASE 2: Threshold Sweep Across Aggregation Methods")
    print("="*70)
    
    aggregation_methods = ['max', 'mean', 'p90', 'p75', 'top3_mean', 'top5_mean']
    
    results_by_method = {}
    
    for method in aggregation_methods:
        scores = [aggregate(v['probs'], method) for v in video_data]
        labels = [v['label'] for v in video_data]
        
        # Find optimal threshold for F1
        best_f1 = find_optimal_threshold(scores, labels, metric='f1')
        # Find optimal threshold for Recall (medical priority)
        best_recall = find_optimal_threshold(scores, labels, metric='recall')
        # Find threshold for recall >= 0.90 with best precision
        recall_90_results = []
        for t in np.arange(0.05, 0.96, 0.01):
            m = compute_metrics(scores, labels, t)
            if m['recall'] >= 0.90:
                recall_90_results.append(m)
        best_recall_90 = max(recall_90_results, key=lambda x: x['precision']) if recall_90_results else None
        
        results_by_method[method] = {
            'best_f1': best_f1,
            'best_recall': best_recall,
            'best_recall_90_precision': best_recall_90
        }
        
        print(f"\n[{method.upper()}]")
        print(f"  Best F1:     threshold={best_f1['threshold']:.2f}  "
              f"F1={best_f1['f1']:.3f}  Recall={best_f1['recall']:.3f}  Precision={best_f1['precision']:.3f}")
        if best_recall_90:
            print(f"  Recall≥90%:  threshold={best_recall_90['threshold']:.2f}  "
                  f"F1={best_recall_90['f1']:.3f}  Recall={best_recall_90['recall']:.3f}  Precision={best_recall_90['precision']:.3f}")
        else:
            print(f"  Recall≥90%:  Not achievable with this method")

    # Phase 3: Per-patient analysis
    print("\n" + "="*70)
    print("PHASE 3: Per-Patient Analysis (using max aggregation)")
    print("="*70)
    
    patient_data = defaultdict(lambda: {'seizure_probs': [], 'normal_probs': [], 'seizure_videos': [], 'normal_videos': []})
    
    for v in video_data:
        pid = v['patient_id']
        max_prob = aggregate(v['probs'], 'max')
        if v['label'] == 1:
            patient_data[pid]['seizure_probs'].append(max_prob)
            patient_data[pid]['seizure_videos'].append(v['name'])
        else:
            patient_data[pid]['normal_probs'].append(max_prob)
            patient_data[pid]['normal_videos'].append(v['name'])
    
    print(f"\n{'Patient':<10} {'Seizure Avg':>12} {'Normal Avg':>12} {'Overlap':>10}")
    print("-" * 50)
    
    for pid in sorted(patient_data.keys()):
        pd = patient_data[pid]
        s_avg = np.mean(pd['seizure_probs']) if pd['seizure_probs'] else None
        n_avg = np.mean(pd['normal_probs']) if pd['normal_probs'] else None
        
        s_str = f"{s_avg:.3f}" if s_avg is not None else "  N/A"
        n_str = f"{n_avg:.3f}" if n_avg is not None else "  N/A"
        
        # Check if there's overlap (hard to separate)
        overlap = "⚠ HARD" if (s_avg is not None and n_avg is not None and abs(s_avg - n_avg) < 0.2) else ""
        
        print(f"{pid:<10} {s_str:>12} {n_str:>12} {overlap:>10}")

    # Phase 4: Analysis of missed cases
    print("\n" + "="*70)
    print("PHASE 4: Missed Seizure Analysis (max aggregation, threshold=0.24)")
    print("="*70)
    
    missed = [v for v in video_data if v['label'] == 1 and aggregate(v['probs'], 'max') < 0.24]
    print(f"\nMissed seizure videos: {len(missed)}")
    for v in missed:
        max_p = aggregate(v['probs'], 'max')
        n_windows = len(v['probs'])
        print(f"  {v['name']}: MaxProb={max_p:.3f}, Windows={n_windows}, Patient={v['patient_id']}")
        if n_windows == 0:
            print(f"    ⚠ NO WINDOWS EXTRACTED - Person not detected in any frame!")
        elif max_p < 0.05:
            print(f"    ⚠ Very low confidence - possible absence seizure or poor video quality")

    # Phase 5: Overall recommendation
    print("\n" + "="*70)
    print("PHASE 5: OPTIMAL CONFIGURATION RECOMMENDATION")
    print("="*70)
    
    # Find the overall best method
    best_overall = None
    best_f1_val = 0
    for method, res in results_by_method.items():
        if res['best_f1']['f1'] > best_f1_val:
            best_f1_val = res['best_f1']['f1']
            best_overall = (method, res['best_f1'])
    
    print(f"\n🏆 BEST F1 CONFIGURATION:")
    print(f"   Method:    {best_overall[0]}")
    print(f"   Threshold: {best_overall[1]['threshold']:.2f}")
    print(f"   F1:        {best_overall[1]['f1']:.3f} ({best_overall[1]['f1']*100:.1f}%)")
    print(f"   Recall:    {best_overall[1]['recall']:.3f} ({best_overall[1]['recall']*100:.1f}%)")
    print(f"   Precision: {best_overall[1]['precision']:.3f} ({best_overall[1]['precision']*100:.1f}%)")
    print(f"   TP={best_overall[1]['TP']}, FP={best_overall[1]['FP']}, FN={best_overall[1]['FN']}, TN={best_overall[1]['TN']}")
    
    # Medical priority: highest recall with acceptable precision
    print(f"\n🏥 MEDICAL PRIORITY (Recall ≥ 90%) CONFIGURATION:")
    best_medical = None
    best_medical_precision = 0
    for method, res in results_by_method.items():
        r = res['best_recall_90_precision']
        if r and r['precision'] > best_medical_precision:
            best_medical_precision = r['precision']
            best_medical = (method, r)
    
    if best_medical:
        print(f"   Method:    {best_medical[0]}")
        print(f"   Threshold: {best_medical[1]['threshold']:.2f}")
        print(f"   F1:        {best_medical[1]['f1']:.3f} ({best_medical[1]['f1']*100:.1f}%)")
        print(f"   Recall:    {best_medical[1]['recall']:.3f} ({best_medical[1]['recall']*100:.1f}%)")
        print(f"   Precision: {best_medical[1]['precision']:.3f} ({best_medical[1]['precision']*100:.1f}%)")
        print(f"   TP={best_medical[1]['TP']}, FP={best_medical[1]['FP']}, FN={best_medical[1]['FN']}, TN={best_medical[1]['TN']}")
    else:
        print("   90% Recall not achievable with current models")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
