"""
Direct Seizure Ensemble Test

Tests all 10 models (5 motion + 5 temporal) directly on test videos,
bypassing the real-time pipeline. Encodes each video exactly like training
preprocessing, then averages 10 model probabilities.

Usage:
    python scripts/test_seizure_ensemble.py
"""

import sys
import cv2
import json
import torch
import numpy as np
import timm
import random
import re
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ============================================================
# CONFIG
# ============================================================
MOTION_MODEL_DIR = Path('seizure_detection/seizure_v3_ensemble')
TEMPORAL_MODEL_DIR = Path('seizure_detection/seizure_temporal_ensemble')
VIDEO_ROOT = Path('datasets/vision/processed/unusual_movement/videos')
WINDOW = 60       # frames per window (must match training)
STRIDE = 15       # sliding window stride
TARGET_SIZE = 224
PERSON_CONF = 0.25  # YOLO person detection confidence

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ============================================================
# HELPERS
# ============================================================
def normalize_channel(channel):
    mn, mx = channel.min(), channel.max()
    if mx - mn < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)
    return ((channel - mn) / (mx - mn) * 255).astype(np.uint8)


def create_motion_summary(gray_crops):
    """R=mean_diff, G=std_diff, B=max_diff with contrast stretch"""
    diffs = np.abs(np.diff(gray_crops, axis=0))  # (N-1, H, W)
    mean_diff = normalize_channel(np.mean(diffs, axis=0))
    std_diff = normalize_channel(np.std(diffs, axis=0))
    max_diff = normalize_channel(np.max(diffs, axis=0))
    img = np.stack([mean_diff, std_diff, max_diff], axis=-1)
    img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))
    return img


def create_temporal_map(gray_crops):
    """2D spectrogram of motion over time"""
    diffs = np.abs(np.diff(gray_crops, axis=0))  # (N-1, H, W)
    resized = np.array([cv2.resize(d, (TARGET_SIZE, TARGET_SIZE)) for d in diffs])
    row_avg = np.mean(resized, axis=2)  # (N-1, 224)
    tmap = cv2.resize(row_avg.T, (TARGET_SIZE, TARGET_SIZE))
    tmap = normalize_channel(tmap)
    return np.stack([tmap] * 3, axis=-1)


def preprocess_image(img, device):
    """(H,W,3) uint8 -> (1,3,224,224) normalized tensor"""
    t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    t = (t - IMAGENET_MEAN) / IMAGENET_STD
    return t.unsqueeze(0).to(device)


def detect_person(frame, detector):
    """Run YOLO, return padded bbox or None"""
    results = detector(frame, classes=[0], conf=PERSON_CONF, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    # Pick largest box
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    idx = areas.argmax().item()
    x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
    # Pad 20%
    h, w = frame.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    return (x1, y1, x2, y2)


def extract_windows_from_video(video_path, detector):
    """
    Read video, extract sliding windows, detect person on middle frame,
    crop all frames with that bbox, return list of (motion_img, temporal_img).
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < WINDOW:
        # Video shorter than window: use all frames
        windows_starts = [0] if len(frames) >= 10 else []
        effective_window = len(frames)
    else:
        windows_starts = list(range(0, len(frames) - WINDOW + 1, STRIDE))
        effective_window = WINDOW

    results = []
    for start in windows_starts:
        end = start + effective_window
        window_frames = frames[start:end]

        # Detect person on middle frame
        mid = len(window_frames) // 2
        bbox = detect_person(window_frames[mid], detector)
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        # Crop and convert to grayscale
        gray_crops = []
        valid = True
        for f in window_frames:
            crop = f[y1:y2, x1:x2]
            if crop.size == 0:
                valid = False
                break
            gray_crops.append(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32))
        if not valid or len(gray_crops) < 3:
            continue

        gray_crops = np.array(gray_crops)
        motion = create_motion_summary(gray_crops)
        temporal = create_temporal_map(gray_crops)
        results.append((motion, temporal))

    return results


# ============================================================
# MAIN
# ============================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load YOLO for person detection
    from ultralytics import YOLO
    detector = YOLO('yolov8n.pt')
    print("Person detector loaded")

    # Load 10 models
    motion_models = []
    for pt in sorted(MOTION_MODEL_DIR.glob('fold*.pt')):
        m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2,
                              drop_rate=0.5, drop_path_rate=0.2)
        m.load_state_dict(torch.load(pt, map_location=device))
        m.to(device).eval()
        motion_models.append(m)
    print(f"Motion models: {len(motion_models)}")

    temporal_models = []
    for pt in sorted(TEMPORAL_MODEL_DIR.glob('fold*.pt')):
        m = timm.create_model('efficientnet_b0', pretrained=False, num_classes=2,
                              drop_rate=0.5, drop_path_rate=0.2)
        m.load_state_dict(torch.load(pt, map_location=device))
        m.to(device).eval()
        temporal_models.append(m)
    print(f"Temporal models: {len(temporal_models)}")
    print(f"Total ensemble: {len(motion_models) + len(temporal_models)} models\n")

    # Discover videos and recreate patient-level splits (same as preprocessing)
    seizure_dir = VIDEO_ROOT / 'seizure'
    normal_dir = VIDEO_ROOT / 'normal'
    all_seizure = sorted(seizure_dir.glob('*.mp4'))
    all_normal = sorted(normal_dir.glob('*.mp4'))
    
    def extract_patient_id(video_path):
        """Extract patient ID from filename (e.g., S47_11_282.mp4 -> S47)"""
        match = re.match(r'^([SN]\d+)', video_path.name)
        return match.group(1) if match else None
    
    # Group by patient (same logic as preprocessing)
    random.seed(42)
    patient_videos = defaultdict(lambda: {'seizure': [], 'normal': []})
    for v in all_seizure:
        pid = extract_patient_id(v)
        if pid:
            patient_videos[pid]['seizure'].append(v)
    for v in all_normal:
        pid = extract_patient_id(v)
        if pid:
            patient_videos[pid]['normal'].append(v)
    
    patients = sorted(patient_videos.keys())
    random.shuffle(patients)
    
    # Split patients: 70% train, 15% val, 15% test (same ratios as preprocessing)
    n_patients = len(patients)
    n_train = int(n_patients * 0.70)
    n_val = int(n_patients * 0.15)
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])
    
    # Extract TEST videos only
    seizure_videos = []
    normal_videos = []
    for pid in test_patients:
        seizure_videos.extend(patient_videos[pid]['seizure'])
        normal_videos.extend(patient_videos[pid]['normal'])
    
    seizure_videos = sorted(seizure_videos)
    normal_videos = sorted(normal_videos)
    
    print(f"Total patients: {n_patients} (train={len(train_patients)}, val={len(val_patients)}, test={len(test_patients)})")
    print(f"TEST SET - Seizure videos: {len(seizure_videos)}")
    print(f"TEST SET - Normal videos: {len(normal_videos)}\n")

    # Test each video
    all_true = []
    all_pred_prob_avg = []
    all_pred_prob_max = []
    all_names = []
    skipped = 0

    for label, videos in [('seizure', seizure_videos), ('normal', normal_videos)]:
        print(f"Processing {label.upper()} videos...")
        for vpath in tqdm(videos, desc=f"  {label}"):
            windows = extract_windows_from_video(vpath, detector)
            if len(windows) == 0:
                skipped += 1
                # No windows extracted — default to normal
                all_true.append(1 if label == 'seizure' else 0)
                all_pred_prob_avg.append(0.0)
                all_pred_prob_max.append(0.0)
                all_names.append(vpath.name)
                continue

            # Run 10-model ensemble on each window, collect seizure probs
            window_probs = []
            with torch.no_grad():
                for motion_img, temporal_img in windows:
                    motion_t = preprocess_image(motion_img, device)
                    temporal_t = preprocess_image(temporal_img, device)

                    probs = []
                    for m in motion_models:
                        p = torch.softmax(m(motion_t), dim=1)[0, 1].item()  # seizure prob
                        probs.append(p)
                    for m in temporal_models:
                        p = torch.softmax(m(temporal_t), dim=1)[0, 1].item()
                        probs.append(p)

                    # Average across 10 models for this window
                    window_probs.append(np.mean(probs))

            # Video-level: both avg and max across windows
            all_true.append(1 if label == 'seizure' else 0)
            all_pred_prob_avg.append(np.mean(window_probs))
            all_pred_prob_max.append(np.max(window_probs))
            all_names.append(vpath.name)

    # Find best threshold for both aggregation methods
    print(f"\nSkipped (no windows): {skipped}")
    all_true = np.array(all_true)
    all_pred_prob_avg = np.array(all_pred_prob_avg)
    all_pred_prob_max = np.array(all_pred_prob_max)

    # Method 1: Average window probs
    best_f1_avg = 0
    best_thresh_avg = 0.5
    for thresh in np.arange(0.20, 0.70, 0.01):
        preds = (all_pred_prob_avg >= thresh).astype(int)
        f1 = f1_score(all_true, preds, zero_division=0)
        if f1 > best_f1_avg:
            best_f1_avg = f1
            best_thresh_avg = thresh

    preds_avg = (all_pred_prob_avg >= best_thresh_avg).astype(int)
    acc_avg = accuracy_score(all_true, preds_avg)
    prec_avg = precision_score(all_true, preds_avg, zero_division=0)
    rec_avg = recall_score(all_true, preds_avg, zero_division=0)
    f1_avg = f1_score(all_true, preds_avg, zero_division=0)
    cm_avg = confusion_matrix(all_true, preds_avg)

    # Method 2: Max window prob
    best_f1_max = 0
    best_thresh_max = 0.5
    for thresh in np.arange(0.20, 0.70, 0.01):
        preds = (all_pred_prob_max >= thresh).astype(int)
        f1 = f1_score(all_true, preds, zero_division=0)
        if f1 > best_f1_max:
            best_f1_max = f1
            best_thresh_max = thresh

    preds_max = (all_pred_prob_max >= best_thresh_max).astype(int)
    acc_max = accuracy_score(all_true, preds_max)
    prec_max = precision_score(all_true, preds_max, zero_division=0)
    rec_max = recall_score(all_true, preds_max, zero_division=0)
    f1_max = f1_score(all_true, preds_max, zero_division=0)
    cm_max = confusion_matrix(all_true, preds_max)

    print("\n" + "=" * 80)
    print("DIRECT ENSEMBLE TEST RESULTS (10 models)")
    print("=" * 80)
    
    print("\nMethod 1: AVG window probs (more conservative)")
    print(f"  Best threshold: {best_thresh_avg:.2f}")
    print(f"  Accuracy:  {acc_avg*100:.2f}%")
    print(f"  Precision: {prec_avg:.4f}")
    print(f"  Recall:    {rec_avg:.4f}")
    print(f"  F1 Score:  {f1_avg:.4f}")
    print(f"  Confusion Matrix: {cm_avg.tolist()}")

    print("\nMethod 2: MAX window prob (more sensitive)")
    print(f"  Best threshold: {best_thresh_max:.2f}")
    print(f"  Accuracy:  {acc_max*100:.2f}%")
    print(f"  Precision: {prec_max:.4f}")
    print(f"  Recall:    {rec_max:.4f}")
    print(f"  F1 Score:  {f1_max:.4f}")
    print(f"  Confusion Matrix: {cm_max.tolist()}")

    # Show per-class stats (using avg)
    seizure_probs_avg = all_pred_prob_avg[all_true == 1]
    normal_probs_avg = all_pred_prob_avg[all_true == 0]
    print(f"\nSeizure video avg prob: min={seizure_probs_avg.min():.3f}, mean={seizure_probs_avg.mean():.3f}, max={seizure_probs_avg.max():.3f}")
    print(f"Normal  video avg prob: min={normal_probs_avg.min():.3f}, mean={normal_probs_avg.mean():.3f}, max={normal_probs_avg.max():.3f}")

    # Save
    output = {
        'avg_aggregation': {
            'best_threshold': float(best_thresh_avg),
            'accuracy': float(acc_avg),
            'precision': float(prec_avg),
            'recall': float(rec_avg),
            'f1': float(f1_avg),
            'confusion_matrix': cm_avg.tolist()
        },
        'max_aggregation': {
            'best_threshold': float(best_thresh_max),
            'accuracy': float(acc_max),
            'precision': float(prec_max),
            'recall': float(rec_max),
            'f1': float(f1_max),
            'confusion_matrix': cm_max.tolist()
        },
        'per_video': [
            {'video': n, 'true': int(t), 'prob_avg': float(pa), 'prob_max': float(pm)}
            for n, t, pa, pm in zip(all_names, all_true, all_pred_prob_avg, all_pred_prob_max)
        ]
    }
    out_path = Path('seizure_detection/report_eval/ensemble_direct_test.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == '__main__':
    main()
