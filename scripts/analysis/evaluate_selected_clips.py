"""
Targeted pipeline evaluation on demo clips for SEIZURE detection.
Uses Kaggle cloud API for MoViNet inference (same path as live demo).
Tests S37 and S15 clips: 4 normal (no seizure) + 4 seizure.

Usage (from repo root):
    python scripts/analysis/evaluate_selected_clips.py
"""

import os
import sys
import time
import base64
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

import cv2
import numpy as np
import requests
import yaml

# ── All demo seizure/normal videos ─────────────────────────────────────────────
RAW_ROOT = Path(__file__).resolve().parent.parent.parent / "demo_dataset/unusual_movement/data"

CLIPS = [
    # patient, label, filename
    ("S37", "normal",  "S37_0_39.mp4"),
    ("S37", "normal",  "S37_0_170.mp4"),
    ("S15", "normal",  "S15_1_140.mp4"),
    ("S15", "normal",  "S15_2_2.mp4"),
    ("S37", "seizure", "S37_0_80.mp4"),
    ("S37", "seizure", "S37_0_75.mp4"),
    ("S15", "seizure", "S15_3_88.mp4"),
    ("S15", "seizure", "S15_3_89.mp4"),
]

# Seizure model: 64-frame rolling buffer → stride-2 → 32 model frames.
# Evaluate one window every SEIZURE_EVAL_STRIDE new frames.
SEIZURE_EVAL_STRIDE = 32
SEIZURE_BUFFER      = 64
FRAME_SIZE          = 224


# ── Synchronous Kaggle API call ────────────────────────────────────────────────
def _call_seizure_api(frames_float32: np.ndarray, endpoint: str, timeout: float = 15.0) -> float:
    """
    Send a (32, 224, 224, 3) float32 RGB clip to the Kaggle seizure endpoint.
    Returns seizure probability as float.
    """
    frames_b64 = []
    for i in range(frames_float32.shape[0]):
        img_uint8 = (frames_float32[i] * 255.0).astype(np.uint8)
        img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
        _, buf    = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frames_b64.append(base64.b64encode(buf).decode('utf-8'))

    try:
        resp = requests.post(
            f"{endpoint.rstrip('/')}/predict/seizure",
            json={"frames_b64": frames_b64},
            timeout=timeout
        )
        if resp.status_code == 200:
            return float(resp.json().get("seizure_prob", 0.0))
        print(f"\n  [API] HTTP {resp.status_code}: {resp.text[:120]}")
    except Exception as e:
        print(f"\n  [API] Request failed: {e}")
    return 0.0


def _build_seizure_clip(frame_buffer: list, detection) -> np.ndarray:
    """
    Convert last 64 frames → (32, 224, 224, 3) float32 RGB clip (stride-2),
    cropped to person bbox if available.
    """
    clip = []
    for i in range(0, SEIZURE_BUFFER, 2):   # indices 0,2,4,...,62 → 32 frames
        frm = frame_buffer[i]
        if detection is not None:
            x1, y1, x2, y2 = detection['bbox']
            x1, y1 = max(0, x1), max(0, y1)
            if x2 > x1 and y2 > y1:
                crop = frm[y1:y2, x1:x2]
                frm  = cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE)) if crop.size > 0 \
                       else cv2.resize(frm, (FRAME_SIZE, FRAME_SIZE))
            else:
                frm = cv2.resize(frm, (FRAME_SIZE, FRAME_SIZE))
        else:
            frm = cv2.resize(frm, (FRAME_SIZE, FRAME_SIZE))
        rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clip.append(rgb)
    return np.stack(clip)   # (32, 224, 224, 3)


# ── Main evaluation function ───────────────────────────────────────────────────
def evaluate_video(detector, video_path: Path, endpoint: str, threshold: float):
    """
    Bypass the async pipeline classifier entirely.
    - Use PersonDetector for YOLO bounding boxes.
    - Maintain our own 64-frame rolling buffer.
    - Every SEIZURE_EVAL_STRIDE frames, send one synchronous Kaggle request.
    - Collect all probabilities and return max.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    frame_buffer = []
    all_probs    = []
    n_frames     = 0
    api_calls    = 0
    t_start      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        n_frames += 1
        frame_buffer.append(frame)

        if len(frame_buffer) > SEIZURE_BUFFER:
            frame_buffer.pop(0)

        # Evaluate when buffer is full AND at a stride boundary
        if len(frame_buffer) == SEIZURE_BUFFER and (n_frames - SEIZURE_BUFFER) % SEIZURE_EVAL_STRIDE == 0:
            detection = detector.detect(frame)
            clip      = _build_seizure_clip(frame_buffer, detection)
            prob      = _call_seizure_api(clip, endpoint)
            all_probs.append(prob)
            api_calls += 1
            print(f".", end="", flush=True)

    cap.release()
    elapsed  = time.time() - t_start
    max_prob = max(all_probs) if all_probs else 0.0
    detected = max_prob >= threshold

    return {
        'detected':  detected,
        'max_prob':  max_prob,
        'all_probs': all_probs,
        'n_frames':  n_frames,
        'api_calls': api_calls,
        'elapsed_s': elapsed,
    }


def main():
    endpoint  = os.getenv("KAGGLE_ENDPOINT", "")
    mode      = os.getenv("INFERENCE_MODE", "LOCAL").upper()

    with open('config/config.yaml') as f:
        cfg = yaml.safe_load(f)

    threshold = cfg['vision']['seizure_classifier']['threshold']   # 0.30 from config

    print("=" * 70)
    print("SEIZURE DETECTION — DEMO DATASET EVALUATION  [S37 & S15, 8 clips]")
    print("=" * 70)
    print(f"  Inference mode   : {mode}")
    print(f"  Kaggle endpoint  : {endpoint or '(not set!)'}")
    print(f"  Seizure threshold: {threshold}  (from config)")
    print(f"  Eval stride      : every {SEIZURE_EVAL_STRIDE} frames (non-overlapping clips)")
    print()

    if mode == 'KAGGLE' and not endpoint:
        print("ERROR: INFERENCE_MODE=KAGGLE but KAGGLE_ENDPOINT is not set in .env")
        sys.exit(1)

    # Only need PersonDetector — no TF models loaded
    from visual_guardian.person_detector import PersonDetector
    detector = PersonDetector(
        model_path=cfg['vision']['person_detector']['model'],
        confidence=cfg['vision']['person_detector']['confidence'],
        device=cfg['vision']['person_detector'].get('device', 'intel:cpu'),
        process_every=1,   # Always detect for accurate crop in eval mode
    )
    print("Person detector ready.\n")

    results = []
    for patient, label, fname in CLIPS:
        folder = "Normal" if label == "normal" else "Seizure"
        path   = RAW_ROOT / folder / fname

        if not path.exists():
            print(f"  [MISSING] {fname}  ({path})")
            results.append((patient, label, fname, None))
            continue

        print(f"  Running: {fname} ({patient} / {label}) ... ", end="", flush=True)
        res = evaluate_video(detector, path, endpoint, threshold)
        if res is None:
            print("FAILED TO OPEN")
            results.append((patient, label, fname, None))
            continue

        results.append((patient, label, fname, res))
        detected_str = "DETECTED" if res['detected'] else "not detected"
        probs_str    = "  ".join(f"{p:.2f}" for p in res['all_probs'])
        print(
            f"\n    max={res['max_prob']:.3f}  [{probs_str}]"
            f"  calls={res['api_calls']}  frames={res['n_frames']}"
            f"  time={res['elapsed_s']:.1f}s  →  {detected_str}"
        )

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for patient in ("S37", "S15"):
        for label in ("normal", "seizure"):
            clips = [(f, r) for (p, l, f, r) in results if p == patient and l == label and r is not None]
            if not clips:
                continue
            print(f"\n  [{patient}] {label.upper()} clips:")
            correct = 0
            for fname, r in clips:
                expected_detect = (label == "seizure")
                actual_detect   = r['detected']
                correct_flag    = (expected_detect == actual_detect)
                correct        += int(correct_flag)
                status = "CORRECT" if correct_flag else "WRONG"
                print(
                    f"    {fname:<22}  max_prob={r['max_prob']:.3f}  "
                    f"detected={str(r['detected']):<5}  [{status}]"
                )
            print(f"    --> {correct}/{len(clips)} correct")

    valid         = [(p, l, f, r) for p, l, f, r in results if r is not None]
    total_correct = sum(1 for p, l, f, r in valid if (l == "seizure") == r['detected'])
    sz            = [(f, r) for p, l, f, r in valid if l == "seizure"]
    nr            = [(f, r) for p, l, f, r in valid if l == "normal"]

    print()
    print(f"  OVERALL: {total_correct}/{len(valid)} clips correct")
    if sz:
        detected_sz = sum(1 for _, r in sz if r['detected'])
        print(f"  Seizure recall  : {detected_sz}/{len(sz)} detected")
    if nr:
        false_alarms = sum(1 for _, r in nr if r['detected'])
        print(f"  Normal (no FA)  : {len(nr) - false_alarms}/{len(nr)} clean  ({false_alarms} false alarm(s))")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
