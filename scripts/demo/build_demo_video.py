"""
Demo Video Builder — Vital Guardian

Preprocesses all 11 validated demo clips, normalizes them to a consistent
format (1920x1080 @ 30fps with CLAHE enhancement), and stitches them into
a single demo_combined.mp4 with 3-second black gaps between clips.

The black gaps serve as natural pipeline resets:
  - Darkness detector fires → pipeline ignores frames
  - Seizure buffer (60 frames / 2s) fully flushes before next clip
  - Smoothers reset due to zero-probability frames during gap

Usage:
    python scripts/demo/build_demo_video.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import json
import numpy as np
from pathlib import Path

# ──────────────────────────────────────────────
# OUTPUT SETTINGS
# ──────────────────────────────────────────────
OUTPUT_DIR  = Path(r"d:\project\FYP\demo_output")
OUTPUT_PATH = OUTPUT_DIR / "demo_combined.mp4"
OUTPUT_FPS  = 30
OUTPUT_W    = 1920
OUTPUT_H    = 1080
GAP_SECONDS = 3          # Black gap duration between clips

# CLAHE enhancement parameters
CLAHE_CLIP  = 2.0        # Contrast limit (higher = more contrast boost)
CLAHE_GRID  = (8, 8)     # Tile grid size

# ──────────────────────────────────────────────
# CLIP SEQUENCE (ordered for demo narrative)
# ──────────────────────────────────────────────
# Narrative: Normal activity → Falls → S37 calm/seizure → S15 restless/seizure
CLIPS = [
    {
        "name":    "B_M_48",
        "label":   "No Fall",
        "patient": "B",
        "type":    "normal",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\normal\Harvard_no-fall\nf_raw_b_3\nf_raw_b_3\B_M_48.mp4"),
        "caption": "Normal Activity — No Alert Expected",
    },
    {
        "name":    "B_D_0231",
        "label":   "Fall",
        "patient": "B",
        "type":    "fall",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\falls\harvard_fall\f_raw_b_1\f_raw_b_1\B_D_0231.mp4"),
        "caption": "Fall Detected",
    },
    {
        "name":    "B_N_458",
        "label":   "Fall",
        "patient": "B",
        "type":    "fall",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\falls\harvard_fall\f_raw_b_2\f_raw_b_2\B_N_458.mp4"),
        "caption": "Fall Detected",
    },
    {
        "name":    "S37_0_39",
        "label":   "Normal",
        "patient": "S37",
        "type":    "normal",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S37_0_39.mp4"),
        "caption": "Patient S37 — Resting Normally",
    },
    {
        "name":    "S37_0_170",
        "label":   "Normal",
        "patient": "S37",
        "type":    "normal",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S37_0_170.mp4"),
        "caption": "Patient S37 — Still Calm",
    },
    {
        "name":    "S37_0_80",
        "label":   "Seizure",
        "patient": "S37",
        "type":    "seizure",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S37_0_80.mp4"),
        "caption": "Patient S37 — SEIZURE EPISODE",
    },
    {
        "name":    "S37_0_75",
        "label":   "Seizure",
        "patient": "S37",
        "type":    "seizure",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S37_0_75.mp4"),
        "caption": "Patient S37 — SEIZURE EPISODE",
    },
    {
        "name":    "S15_1_140",
        "label":   "Normal",
        "patient": "S15",
        "type":    "normal",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S15_1_140.mp4"),
        "caption": "Patient S15 — Restless But Normal",
    },
    {
        "name":    "S15_2_2",
        "label":   "Normal",
        "patient": "S15",
        "type":    "normal",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S15_2_2.mp4"),
        "caption": "Patient S15 — Active Movement, No Alert",
    },
    {
        "name":    "S15_3_88",
        "label":   "Seizure",
        "patient": "S15",
        "type":    "seizure",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S15_3_88.mp4"),
        "caption": "Patient S15 — SEIZURE EPISODE (High-Motion Patient!)",
    },
    {
        "name":    "S15_3_89",
        "label":   "Seizure",
        "patient": "S15",
        "type":    "seizure",
        "path":    Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S15_3_89.mp4"),
        "caption": "Patient S15 — SEIZURE EPISODE",
    },
]


# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def apply_clahe(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE to the L-channel of LAB colorspace for natural enhancement."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Resize to 1080p and apply CLAHE."""
    h, w = frame.shape[:2]
    if w != OUTPUT_W or h != OUTPUT_H:
        frame = cv2.resize(frame, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LANCZOS4)
    frame = apply_clahe(frame)
    return frame


def make_gap_frame() -> np.ndarray:
    """Return a pure black 1080p frame for inter-clip gaps."""
    return np.zeros((OUTPUT_H, OUTPUT_W, 3), dtype=np.uint8)


def process_clip(writer, clip: dict, clip_num: int):
    """Read, preprocess, label, and write a single clip to the video writer."""
    path = clip["path"]
    if not path.exists():
        print(f"  [MISSING] {clip['name']} at {path}")
        return 0

    cap = cv2.VideoCapture(str(path))
    src_fps  = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    written  = 0

    # Frame-skip/duplicate ratio to convert source FPS → 30
    ratio = OUTPUT_FPS / src_fps

    frame_idx = 0.0
    src_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        src_frame_idx += 1

        # Determine how many output frames this source frame maps to
        next_frame_idx = frame_idx + ratio
        n_output = int(next_frame_idx) - int(frame_idx)
        frame_idx = next_frame_idx

        if n_output < 1:
            continue  # Skip this source frame (FPS downsampling)

        # Preprocess once (no label burn-in — dashboard handles labels)
        proc = preprocess_frame(frame)

        for _ in range(n_output):
            writer.write(proc)
            written += 1

    cap.release()
    return written


def write_gap(writer, gap_seconds: int = GAP_SECONDS):
    """Write black frames for the inter-clip gap."""
    n = gap_seconds * OUTPUT_FPS
    black = make_gap_frame()
    for _ in range(n):
        writer.write(black)
    return n


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, OUTPUT_FPS, (OUTPUT_W, OUTPUT_H))

    if not writer.isOpened():
        print(f"ERROR: Could not open video writer at {OUTPUT_PATH}")
        return

    BOUNDARIES_PATH = OUTPUT_DIR / "demo_clip_boundaries.json"

    print("=" * 70)
    print("VITAL GUARDIAN — DEMO VIDEO BUILDER")
    print("=" * 70)
    print(f"  Output  : {OUTPUT_PATH}")
    print(f"  Format  : {OUTPUT_W}x{OUTPUT_H} @ {OUTPUT_FPS}fps")
    print(f"  Clips   : {len(CLIPS)}")
    print(f"  Gap     : {GAP_SECONDS}s black between clips")
    print(f"  CLAHE   : clip={CLAHE_CLIP}, grid={CLAHE_GRID}")
    print()

    total_frames = 0
    boundaries = []

    for i, clip in enumerate(CLIPS):
        clip_num = i + 1
        print(f"  [{clip_num:02d}/{len(CLIPS)}] {clip['name']:<15} ({clip['type']:<7}) ... ", end="", flush=True)

        clip_start = total_frames
        written = process_clip(writer, clip, clip_num)
        total_frames += written
        clip_end = total_frames - 1
        print(f"{written} frames written")

        # Record boundary for this clip (1-indexed frames)
        boundaries.append({
            "clip_num":    clip_num,
            "name":        clip["name"],
            "patient":     clip["patient"],
            "type":        clip["type"],
            "label":       clip["label"],
            "caption":     clip["caption"],
            "start_frame": clip_start,
            "end_frame":   clip_end,
        })

        # Write inter-clip gap (except after the last clip)
        if i < len(CLIPS) - 1:
            gap_frames = write_gap(writer)
            total_frames += gap_frames
            print(f"         {'':15} (gap)       ... {gap_frames} black frames")

    writer.release()

    # Write boundaries JSON sidecar
    with open(BOUNDARIES_PATH, 'w') as f:
        json.dump(boundaries, f, indent=2)
    print(f"  Boundaries saved: {BOUNDARIES_PATH}")

    duration = total_frames / OUTPUT_FPS
    print()
    print("=" * 70)
    print(f"  Done! Total: {total_frames} frames ({duration:.1f}s / {duration/60:.1f}min)")
    print(f"  Saved to: {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
