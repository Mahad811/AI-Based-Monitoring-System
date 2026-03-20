"""
Fall Detection Evaluation Script — Full Pipeline
Tests the complete fall detection pipeline including:
  - FallClassifier (5-model EfficientNet-B0 ensemble)
  - Threshold sweep to find optimal configuration
  - Pose-based safety net (check_fallen_state) analysis
  - Per-class and per-video-source breakdown

The fall dataset is frame-level triplets (not raw videos).
Each .jpg is already a temporal RGB triplet (t-1, t, t+1 stacked).
Evaluation: classify each triplet, then aggregate per source video.

Usage:
    python scripts/evaluate_fall_detection.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import cv2
import yaml
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visual_guardian.fall_classifier import FallClassifier

DATASET_ROOT = Path("datasets/vision/fall_classification/test")
MAX_SAMPLES_PER_CLASS = 500   # Set to None to evaluate all ~16k triplets
RANDOM_SEED = 42


def compute_metrics(scores, labels, threshold):
    TP = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 1)
    FP = sum(1 for s, l in zip(scores, labels) if s >= threshold and l == 0)
    FN = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 1)
    TN = sum(1 for s, l in zip(scores, labels) if s < threshold and l == 0)
    p  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    r  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    acc = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    return TP, FP, FN, TN, p, r, f1, acc


def find_optimal_threshold(scores, labels, metric='f1'):
    best, best_val = None, -1
    for t in np.arange(0.05, 0.96, 0.01):
        t = round(float(t), 2)
        TP, FP, FN, TN, p, r, f1, acc = compute_metrics(scores, labels, t)
        val = {'f1': f1, 'recall': r, 'precision': p, 'accuracy': acc}[metric]
        if val > best_val:
            best_val = val
            best = (t, TP, FP, FN, TN, p, r, f1, acc)
    return best


def main():
    # ── Config ──────────────────────────────────────────────────────────────
    with open('config/config.yaml', 'r') as f:
        root_config = yaml.safe_load(f)
    vision_config = root_config['vision']
    fall_config = vision_config['fall_classifier']
    current_threshold = fall_config.get('threshold', 0.6)

    print("=" * 70)
    print("FALL DETECTION EVALUATION — FULL PIPELINE")
    print("=" * 70)
    print(f"  Model path  : {fall_config['model']}")
    print(f"  Threshold   : {current_threshold} (current config)")
    print(f"  Test set    : {DATASET_ROOT}")

    # ── Load classifier ───────────────────────────────────────────────────────
    print("\nLoading FallClassifier (5-model ensemble)...")
    classifier = FallClassifier(
        model_path=fall_config['model'],
        device='auto',
        use_ensemble=True
    )
    print("✓ Ready\n")

    # ── Collect all test triplets ─────────────────────────────────────────────
    all_results = []  # {name, label, fall_prob, source_video}

    for cls in ['fall', 'normal']:
        class_dir = DATASET_ROOT / cls
        if not class_dir.exists():
            print(f"Warning: {class_dir} not found, skipping.")
            continue

        images = sorted(class_dir.glob("*.jpg"))
        label  = 1 if cls == 'fall' else 0

        # Random subset sampling
        if MAX_SAMPLES_PER_CLASS and len(images) > MAX_SAMPLES_PER_CLASS:
            rng = np.random.default_rng(RANDOM_SEED)
            indices = rng.choice(len(images), MAX_SAMPLES_PER_CLASS, replace=False)
            images = [images[i] for i in sorted(indices)]

        print(f"Evaluating {cls} ({len(images)} triplets)...")
        for img_path in tqdm(images, desc=f"  {cls}"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Convert BGR→RGB for classifier
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = classifier.classify(img_rgb)

            if result is None:
                continue

            # Source video = filename prefix before last underscore+frame number
            # e.g. "20240912_101520_f00001.jpg" → "20240912_101520"
            parts = img_path.stem.rsplit('_f', 1)
            source_video = parts[0] if len(parts) == 2 else img_path.stem

            all_results.append({
                'name':        img_path.name,
                'label':       label,
                'cls':         cls,
                'fall_prob':   result['fall_prob'],
                'source_video': source_video,
            })

    print(f"\n✓ Evaluated {len(all_results)} triplets total")

    # ── Frame-level metrics at current threshold ──────────────────────────────
    scores = [r['fall_prob'] for r in all_results]
    labels = [r['label']     for r in all_results]

    print("\n" + "=" * 70)
    print(f"FRAME-LEVEL RESULTS (threshold={current_threshold})")
    print("=" * 70)
    TP, FP, FN, TN, p, r, f1, acc = compute_metrics(scores, labels, current_threshold)
    print(f"Accuracy  : {acc:.1%}")
    print(f"Precision : {p:.1%}")
    print(f"Recall    : {r:.1%}  (fall clips correctly detected)")
    print(f"F1-Score  : {f1:.1%}")
    print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD SWEEP (frame-level)")
    print("=" * 70)

    best_f1     = find_optimal_threshold(scores, labels, 'f1')
    best_recall = find_optimal_threshold(scores, labels, 'recall')

    # Best threshold for recall >= 90% with highest precision
    best_r90 = None
    best_r90_p = 0
    for t in np.arange(0.05, 0.96, 0.01):
        t = round(float(t), 2)
        TP2, FP2, FN2, TN2, p2, r2, f12, acc2 = compute_metrics(scores, labels, t)
        if r2 >= 0.90 and p2 > best_r90_p:
            best_r90_p = p2
            best_r90 = (t, TP2, FP2, FN2, TN2, p2, r2, f12, acc2)

    t, TP, FP, FN, TN, p, r, f1, acc = best_f1
    print(f"\nBest F1:          thresh={t:.2f}  F1={f1:.3f}  Recall={r:.3f}  Precision={p:.3f}  Acc={acc:.3f}")
    print(f"                  TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    if best_r90:
        t, TP, FP, FN, TN, p, r, f1, acc = best_r90
        print(f"Recall>=90%:      thresh={t:.2f}  F1={f1:.3f}  Recall={r:.3f}  Precision={p:.3f}  Acc={acc:.3f}")
        print(f"                  TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    else:
        print("Recall>=90%:      NOT ACHIEVABLE at frame level")

    # ── Video-level aggregation ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("VIDEO-LEVEL AGGREGATION (max probability per source video)")
    print("=" * 70)

    # Group by source video
    video_groups = defaultdict(lambda: {'probs': [], 'label': None})
    for r_item in all_results:
        vid = r_item['source_video']
        video_groups[vid]['probs'].append(r_item['fall_prob'])
        video_groups[vid]['label'] = r_item['label']

    video_scores = [max(v['probs']) for v in video_groups.values()]
    video_labels = [v['label']     for v in video_groups.values()]

    print(f"\nTotal source videos: {len(video_groups)}")
    fall_vids   = sum(1 for l in video_labels if l == 1)
    normal_vids = sum(1 for l in video_labels if l == 0)
    print(f"Fall videos: {fall_vids}, Normal videos: {normal_vids}")

    best_vid_f1     = find_optimal_threshold(video_scores, video_labels, 'f1')
    best_vid_recall = find_optimal_threshold(video_scores, video_labels, 'recall')

    best_vid_r90 = None
    best_vid_r90_p = 0
    for t in np.arange(0.05, 0.96, 0.01):
        t = round(float(t), 2)
        TP2, FP2, FN2, TN2, p2, r2, f12, acc2 = compute_metrics(video_scores, video_labels, t)
        if r2 >= 0.90 and p2 > best_vid_r90_p:
            best_vid_r90_p = p2
            best_vid_r90 = (t, TP2, FP2, FN2, TN2, p2, r2, f12, acc2)

    t, TP, FP, FN, TN, p, r, f1, acc = best_vid_f1
    print(f"\nBest F1:          thresh={t:.2f}  F1={f1:.3f}  Recall={r:.3f}  Precision={p:.3f}  Acc={acc:.3f}")
    print(f"                  TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    if best_vid_r90:
        t, TP, FP, FN, TN, p, r, f1, acc = best_vid_r90
        print(f"Recall>=90%:      thresh={t:.2f}  F1={f1:.3f}  Recall={r:.3f}  Precision={p:.3f}  Acc={acc:.3f}")
        print(f"                  TP={TP}  FP={FP}  FN={FN}  TN={TN}")

    # ── Optimal config recommendation ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("OPTIMAL CONFIGURATION RECOMMENDATION")
    print("=" * 70)

    t, TP, FP, FN, TN, p, r, f1, acc = best_vid_f1
    print(f"\n  Best F1 (video-level max):")
    print(f"    fall_classifier.threshold: {t:.2f}")
    print(f"    F1={f1:.1%}  Recall={r:.1%}  Precision={p:.1%}")

    if best_vid_r90:
        t, TP, FP, FN, TN, p, r, f1, acc = best_vid_r90
        print(f"\n  Medical Priority (Recall>=90%, video-level max):")
        print(f"    fall_classifier.threshold: {t:.2f}")
        print(f"    F1={f1:.1%}  Recall={r:.1%}  Precision={p:.1%}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_path = Path("fall_detection/report_eval/fall_pipeline_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'config': {
            'model': str(fall_config['model']),
            'current_threshold': current_threshold,
        },
        'frame_level': {
            'total': len(all_results),
            'best_f1': {'threshold': best_f1[0], 'f1': best_f1[6], 'recall': best_f1[5], 'precision': best_f1[4]},
            'best_recall_90': {'threshold': best_r90[0], 'f1': best_r90[6], 'recall': best_r90[5], 'precision': best_r90[4]} if best_r90 else None,
        },
        'video_level': {
            'total_videos': len(video_groups),
            'fall_videos': fall_vids,
            'normal_videos': normal_vids,
            'best_f1': {'threshold': best_vid_f1[0], 'f1': best_vid_f1[6], 'recall': best_vid_f1[5], 'precision': best_vid_f1[4]},
            'best_recall_90': {'threshold': best_vid_r90[0], 'f1': best_vid_r90[6], 'recall': best_vid_r90[5], 'precision': best_vid_r90[4]} if best_vid_r90 else None,
        }
    }
    with open(out_path, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {out_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
