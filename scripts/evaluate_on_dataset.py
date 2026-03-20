"""
Full Pipeline Evaluation Script
Tests the complete VisionPipeline end-to-end including:
  - Seizure Classifier (10-model ensemble)
  - Pose Analyzer (MediaPipe)
  - Rhythm Verification (check_seizure_rhythm)
  - State Machine

This is the most realistic evaluation — matches production behavior.

Usage:
    python scripts/evaluate_on_dataset.py

Key config (config.yaml):
    person_detector.confidence: 0.10   (lowered to catch partial detections)
    seizure_classifier.threshold: 0.24 (video-level max aggregation)
"""

import os
# Suppress TensorFlow/MediaPipe info and warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'       # Suppress TF C++ logs (INFO, WARNING)
os.environ['GLOG_minloglevel'] = '3'            # Suppress MediaPipe/glog messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # Suppress oneDNN info

import cv2
import yaml
import sys
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Suppress Python-level TF warnings
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_guardian.pipeline import VisionPipeline

DATASET_ROOT = Path("datasets/vision/processed/unusual_movement/videos/")
SPLIT_FILE   = Path("datasets/vision/processed/unusual_movement/splits.json")

# ── Aggregation: video is "seizure" if ANY frame triggers event_type='seizure'
# ── We also track max smoothed probability for threshold-sweep analysis


def evaluate_video(pipeline, video_path):
    """
    Run the full pipeline on a video.

    Returns:
        dict with:
            detected      : bool  — did 'seizure' event fire at any point?
            max_prob      : float — highest seizure_smoothed seen
            rhythm_fires  : int   — how many times rhythm check confirmed
            rhythm_suppressed: int — how many times rhythm check suppressed
            n_frames      : int
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'detected': False, 'max_prob': 0.0,
                'rhythm_fires': 0, 'rhythm_suppressed': 0, 'n_frames': 0}

    pipeline.reset()
    pipeline.pose_history.clear()
    # Force state to OUT_OF_BED so fall + seizure detection both run
    # (IN_BED skips fall detection but seizure still runs — keep default)
    pipeline.patient_state = 'OUT_OF_BED'

    detected        = False
    max_prob        = 0.0
    rhythm_fires    = 0
    rhythm_suppressed = 0
    n_frames        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n_frames += 1

        event = pipeline.process_frame(frame)

        prob = event.get('seizure_smoothed', 0.0)
        if prob > max_prob:
            max_prob = prob

        if event['event_type'] == 'seizure':
            detected = True
            rhythm_fires += 1

        # Count rhythm suppressions from debug_info
        debug = event.get('debug_info', '')
        if 'Suppressed' in debug:
            rhythm_suppressed += 1

    cap.release()
    return {
        'detected':          detected,
        'max_prob':          max_prob,
        'rhythm_fires':      rhythm_fires,
        'rhythm_suppressed': rhythm_suppressed,
        'n_frames':          n_frames,
    }


def compute_metrics(results):
    TP = sum(1 for r in results if r['label'] == 1 and r['detected'])
    FP = sum(1 for r in results if r['label'] == 0 and r['detected'])
    FN = sum(1 for r in results if r['label'] == 1 and not r['detected'])
    TN = sum(1 for r in results if r['label'] == 0 and not r['detected'])
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    return TP, FP, FN, TN, precision, recall, f1, accuracy


def main():
    # ── Config ──────────────────────────────────────────────────────────────
    with open('config/config.yaml', 'r') as f:
        root_config = yaml.safe_load(f)
    vision_config = root_config['vision']

    print("=" * 70)
    print("FULL PIPELINE EVALUATION")
    print("  Seizure Classifier  +  Pose Analyzer  +  Rhythm Verification")
    print("=" * 70)
    print(f"  Person detector confidence : {vision_config['person_detector']['confidence']}")
    print(f"  Seizure threshold          : {vision_config['seizure_classifier']['threshold']}")
    print(f"  Bed-exit enabled           : {vision_config.get('bed_exit', {}).get('enabled', False)}")

    # ── Splits ───────────────────────────────────────────────────────────────
    with open(SPLIT_FILE, 'r') as f:
        splits = json.load(f)
    test_files = splits.get('test', {})
    n_normal  = len(test_files.get('normal',  []))
    n_seizure = len(test_files.get('seizure', []))
    print(f"\nTest split: {n_normal} normal, {n_seizure} seizure videos\n")

    # ── Pipeline ─────────────────────────────────────────────────────────────
    print("Initializing full VisionPipeline...")
    pipeline = VisionPipeline(vision_config)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    all_results = []

    for cls in ['normal', 'seizure']:
        class_dir = DATASET_ROOT / cls
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        split_filenames = set(test_files.get(cls, []))
        videos = [v for v in class_dir.glob("*.mp4") if v.name in split_filenames]
        label  = 1 if cls == 'seizure' else 0

        print(f"Testing {cls} ({len(videos)} videos)...")
        for video_path in tqdm(videos, desc=f"  {cls}"):
            res = evaluate_video(pipeline, video_path)
            res['name']       = video_path.name
            res['label']      = label
            res['cls']        = cls
            res['patient_id'] = video_path.stem.split('_')[0]
            all_results.append(res)

    # ── Metrics ───────────────────────────────────────────────────────────────
    TP, FP, FN, TN, precision, recall, f1, accuracy = compute_metrics(all_results)

    print("\n" + "=" * 70)
    print("FULL PIPELINE RESULTS")
    print("=" * 70)
    print(f"Total Videos : {len(all_results)}")
    print(f"Accuracy     : {accuracy:.1%}")
    print(f"Precision    : {precision:.1%}  (low → false alarms)")
    print(f"Recall       : {recall:.1%}  (low → missed seizures)")
    print(f"F1-Score     : {f1:.1%}")
    print("-" * 70)
    print(f"True Positives  (Seizure Correct) : {TP}")
    print(f"False Negatives (Seizure Missed)  : {FN}")
    print(f"True Negatives  (Normal Correct)  : {TN}")
    print(f"False Positives (False Alarm)     : {FP}")

    # Rhythm stats
    total_rhythm_fires     = sum(r['rhythm_fires']     for r in all_results)
    total_rhythm_suppressed = sum(r['rhythm_suppressed'] for r in all_results)
    print(f"\nRhythm Verification:")
    print(f"  Confirmed seizures : {total_rhythm_fires}")
    print(f"  Suppressed alarms  : {total_rhythm_suppressed}")

    # ── Per-patient breakdown ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PER-PATIENT BREAKDOWN")
    print("=" * 70)
    patient_stats = defaultdict(lambda: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0})
    for r in all_results:
        pid = r['patient_id']
        if r['label'] == 1 and r['detected']:     patient_stats[pid]['TP'] += 1
        elif r['label'] == 0 and r['detected']:   patient_stats[pid]['FP'] += 1
        elif r['label'] == 1 and not r['detected']:patient_stats[pid]['FN'] += 1
        else:                                      patient_stats[pid]['TN'] += 1

    print(f"\n{'Patient':<10} {'TP':>4} {'FP':>4} {'FN':>4} {'TN':>4}  {'Status'}")
    print("-" * 50)
    for pid in sorted(patient_stats.keys()):
        ps = patient_stats[pid]
        issues = []
        if ps['FN'] > 0: issues.append(f"⚠ {ps['FN']} missed")
        if ps['FP'] > 0: issues.append(f"⚠ {ps['FP']} false alarms")
        status = ', '.join(issues) if issues else "✓ clean"
        print(f"{pid:<10} {ps['TP']:>4} {ps['FP']:>4} {ps['FN']:>4} {ps['TN']:>4}  {status}")

    # ── Failures ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FAILURES")
    print("=" * 70)

    missed = [r for r in all_results if r['label'] == 1 and not r['detected']]
    false_alarms = [r for r in all_results if r['label'] == 0 and r['detected']]

    print(f"\nMissed Seizures ({len(missed)}):")
    for r in missed:
        print(f"  {r['name']:<30}  MaxProb={r['max_prob']:.3f}  "
              f"Frames={r['n_frames']}  RhythmSuppressed={r['rhythm_suppressed']}")

    print(f"\nFalse Alarms ({len(false_alarms)}):")
    for r in false_alarms:
        print(f"  {r['name']:<30}  MaxProb={r['max_prob']:.3f}  "
              f"RhythmFires={r['rhythm_fires']}  Patient={r['patient_id']}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path("seizure_detection/report_eval/full_pipeline_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_data = {
        'config': {
            'person_detector_confidence': vision_config['person_detector']['confidence'],
            'seizure_threshold': vision_config['seizure_classifier']['threshold'],
        },
        'metrics': {
            'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
            'precision': precision, 'recall': recall, 'f1': f1, 'accuracy': accuracy,
        },
        'per_video': all_results
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
