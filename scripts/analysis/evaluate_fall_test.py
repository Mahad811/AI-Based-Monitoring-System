"""
Fall model evaluation on demo_dataset/fall_test/.

Auto-discovers every video in:
  demo_dataset/fall_test/fall/   → ground truth = fall
  demo_dataset/fall_test/nofall/ → ground truth = no fall

Preprocessing exactly matches the training notebook (v2-03-fall-preprocess.ipynb):
  • fix_orientation  (rotate portrait→landscape)
  • YOLO11n per-frame crop with 20% padding
  • stride-2 sampling → 16 frames per clip
  • tail evaluation  (catches falls at end of short clips)

Usage (from repo root):
    python scripts/analysis/evaluate_fall_test.py
"""

import os
import sys
import time
import base64
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel']      = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / '.env')

import cv2
import numpy as np
import requests
import yaml

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent.parent / "demo_dataset" / "fall_test"
FALL_DIR   = ROOT / "fall"
NOFALL_DIR = ROOT / "nofall"

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv'}

# ── Constants (must match training notebook) ───────────────────────────────────
FALL_BUFFER      = 32
FALL_EVAL_STRIDE = 16
FRAME_SIZE       = 224
TRAINING_PADDING = 0.20   # v2-03-fall-preprocess.ipynb Cell 2: PADDING = 0.20
CLIP_FRAMES      = 16     # stride-2 through FALL_BUFFER → 16 frames sent to model


# ── Kaggle API call ────────────────────────────────────────────────────────────
def _call_fall_api(clip_rgb: np.ndarray, endpoint: str, timeout: float = 20.0) -> float:
    """Send (16, 224, 224, 3) float32 RGB clip → Kaggle /predict/fall."""
    frames_b64 = []
    for i in range(clip_rgb.shape[0]):
        u8  = (clip_rgb[i] * 255.0).astype(np.uint8)
        bgr = cv2.cvtColor(u8, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        frames_b64.append(base64.b64encode(buf).decode('utf-8'))
    try:
        r = requests.post(
            f"{endpoint.rstrip('/')}/predict/fall",
            json={"frames_b64": frames_b64},
            timeout=timeout,
        )
        if r.status_code == 200:
            return float(r.json().get("fall_prob", 0.0))
        print(f"\n  [API] HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        print(f"\n  [API] Request failed: {e}")
    return -1.0


# ── Preprocessing helpers (exact replica of training notebook) ─────────────────
def _fix_orientation(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if h > w * 1.5:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _get_crop(frame: np.ndarray, yolo_model, device: str) -> np.ndarray:
    """YOLO11n person crop with 20% padding. Falls back to full-frame resize."""
    results = yolo_model(frame, verbose=False, classes=[0], device=device)
    boxes   = results[0].boxes
    if boxes is not None and len(boxes) > 0:
        idx             = int(boxes.conf.argmax())
        x1, y1, x2, y2 = boxes.xyxy[idx].cpu().numpy().astype(int)
        h, w            = frame.shape[:2]
        pad_x           = int((x2 - x1) * TRAINING_PADDING)
        pad_y           = int((y2 - y1) * TRAINING_PADDING)
        x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x);  y2 = min(h, y2 + pad_y)
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0:
            return cv2.resize(crop, (FRAME_SIZE, FRAME_SIZE))
    return cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))


def _build_clip(frame_buffer: list, yolo_model, device: str) -> np.ndarray:
    """Build (16, 224, 224, 3) float32 RGB clip from 32-frame buffer."""
    clip = []
    for i in range(0, FALL_BUFFER, 2):           # stride-2 → 16 frames
        frm  = _fix_orientation(frame_buffer[i])
        crop = _get_crop(frm, yolo_model, device)
        rgb  = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        clip.append(rgb)
    return np.stack(clip)                         # (16, 224, 224, 3)


TARGET_FPS = 30   # model was trained on ~30fps videos

# ── Per-video evaluation ───────────────────────────────────────────────────────
def evaluate_video(video_path: Path, yolo_model, device: str,
                   endpoint: str, threshold: float) -> dict | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # FPS normalisation: subsample high-fps videos to ~TARGET_FPS (30fps).
    # The model was trained on 30fps clips. At 120fps, 32 frames = only 0.27s —
    # too short to capture a fall. Keep every Nth frame so 32 frames ≈ 1 second.
    keep_every = max(1, round(fps / TARGET_FPS))

    frame_buffer = []
    all_probs    = []
    raw_idx      = 0    # raw frame counter (before subsampling)
    n_frames     = 0    # effective frame counter (after subsampling)
    last_eval_at = 0
    t_start      = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_idx += 1
        if (raw_idx - 1) % keep_every != 0:   # skip to reach ~TARGET_FPS
            continue
        n_frames += 1
        frame_buffer.append(frame)
        if len(frame_buffer) > FALL_BUFFER:
            frame_buffer.pop(0)

        if len(frame_buffer) == FALL_BUFFER and \
                (n_frames - FALL_BUFFER) % FALL_EVAL_STRIDE == 0:
            clip = _build_clip(frame_buffer, yolo_model, device)
            p    = _call_fall_api(clip, endpoint)
            if p >= 0:
                all_probs.append(p)
            last_eval_at = n_frames
            print(".", end="", flush=True)

    cap.release()

    # Tail evaluation: catches falls that peak in the last few frames
    if len(frame_buffer) == FALL_BUFFER and n_frames > last_eval_at:
        clip = _build_clip(frame_buffer, yolo_model, device)
        p    = _call_fall_api(clip, endpoint)
        if p >= 0:
            all_probs.append(p)
        print("t", end="", flush=True)

    elapsed = time.time() - t_start

    # Use the mean of the TOP-2 highest probabilities as the detection score.
    # This is more robust than bare max: one fluky spike won't fire an alert,
    # but two windows that both see a fall (which real falls produce) will.
    sorted_probs = sorted(all_probs, reverse=True)
    top2_mean    = sum(sorted_probs[:2]) / min(2, len(sorted_probs)) if all_probs else 0.0

    eff_fps = fps / keep_every
    return {
        "detected":  top2_mean >= threshold,
        "top2_mean": top2_mean,
        "max_prob":  sorted_probs[0] if sorted_probs else 0.0,
        "all_probs": all_probs,
        "n_frames":  n_frames,
        "fps":       fps,
        "eff_fps":   eff_fps,
        "keep_every": keep_every,
        "res":       f"{width}x{height}",
        "elapsed_s": elapsed,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    endpoint  = os.getenv("KAGGLE_ENDPOINT", "")
    mode      = os.getenv("INFERENCE_MODE", "LOCAL").upper()

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    threshold = cfg["vision"]["fall_classifier"]["threshold"]
    det_cfg   = cfg["vision"]["person_detector"]

    print("=" * 72)
    print("FALL MODEL TEST — demo_dataset/fall_test/")
    print("=" * 72)
    print(f"  Inference mode : {mode}")
    print(f"  Kaggle endpoint: {endpoint or '(not set!)'}")
    print(f"  Fall threshold : {threshold}")
    print(f"  YOLO device    : {det_cfg.get('device', 'intel:cpu')}")
    print(f"  YOLO padding   : {TRAINING_PADDING}  (matches training)")
    print()

    if mode == "KAGGLE" and not endpoint:
        print("ERROR: INFERENCE_MODE=KAGGLE but KAGGLE_ENDPOINT not set in .env")
        sys.exit(1)

    # Load YOLO (only YOLO — no TF needed locally)
    from ultralytics import YOLO
    yolo_model = YOLO(det_cfg["model"])
    device     = det_cfg.get("device", "intel:cpu")
    print("YOLO ready.\n")

    # Collect all videos
    clips = []
    for vpath in sorted(FALL_DIR.iterdir()):
        if vpath.suffix.lower() in VIDEO_EXTS:
            clips.append((vpath, "fall"))
    for vpath in sorted(NOFALL_DIR.iterdir()):
        if vpath.suffix.lower() in VIDEO_EXTS:
            clips.append((vpath, "nofall"))

    if not clips:
        print(f"No video files found in {ROOT}")
        sys.exit(1)

    print(f"Found {sum(1 for _,l in clips if l=='fall')} fall clips, "
          f"{sum(1 for _,l in clips if l=='nofall')} no-fall clips.\n")

    results = []
    for vpath, label in clips:
        pad_name = vpath.name[:26] + ".." if len(vpath.name) > 28 else vpath.name
        print(f"  [{label.upper():<6}] {pad_name:<30} ", end="", flush=True)
        res = evaluate_video(vpath, yolo_model, device, endpoint, threshold)
        if res is None:
            print("FAILED TO OPEN")
            results.append((vpath.name, label, None))
            continue
        results.append((vpath.name, label, res))
        probs_str   = "  ".join(f"{p:.3f}" for p in res["all_probs"])
        detect_str  = "DETECTED" if res["detected"] else "not detected"
        fps_str     = (f"{res['fps']:.0f}fps→{res['eff_fps']:.0f}fps"
                       if res["keep_every"] > 1 else f"{res['fps']:.0f}fps")
        print(f"\n             top2={res['top2_mean']:.3f} (max={res['max_prob']:.3f})  [{probs_str}]"
              f"  {res['n_frames']}f@{fps_str} {res['res']}"
              f"  {res['elapsed_s']:.0f}s  →  {detect_str}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("RESULTS SUMMARY")
    print("=" * 72)

    fall_clips  = [(n, r) for n, l, r in results if l == "fall"  and r is not None]
    nofall_clips= [(n, r) for n, l, r in results if l == "nofall" and r is not None]

    print("\n  FALL clips (expect DETECTED):")
    fall_correct = 0
    for name, r in fall_clips:
        ok  = r["detected"]
        fall_correct += int(ok)
        tag = "CORRECT" if ok else "WRONG"
        print(f"    {name:<35}  top2={r['top2_mean']:.3f}  [{tag}]")
    print(f"  → {fall_correct}/{len(fall_clips)} detected")

    print("\n  NO-FALL clips (expect not detected):")
    nofall_correct = 0
    for name, r in nofall_clips:
        ok  = not r["detected"]
        nofall_correct += int(ok)
        tag = "CORRECT" if ok else "WRONG (false alarm)"
        print(f"    {name:<35}  top2={r['top2_mean']:.3f}  [{tag}]")
    print(f"  → {nofall_correct}/{len(nofall_clips)} clean")

    total   = len(fall_clips) + len(nofall_clips)
    correct = fall_correct + nofall_correct
    print()
    print(f"  OVERALL : {correct}/{total} correct")
    print("=" * 72)


if __name__ == "__main__":
    main()
