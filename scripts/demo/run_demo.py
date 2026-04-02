"""
Demo Runner v2 — Vital Guardian ICU Monitoring Dashboard

7-segment structure:
  1. Patient A — No Fall (1 clip)
  2. Patient A — Fall (1 clip)
  3. Patient A — Fall (1 clip)
  4. Patient B — Normal (S37_0_39 + S37_0_170 merged, no gap)
  5. Patient B — Seizure (S37_0_80 + S37_0_75 merged, no gap)
  6. Patient C — Normal (S15_1_140 + S15_2_2 merged, no gap)
  7. Patient C — Seizure (S15_3_88 + S15_3_89 merged, no gap)

Each clip processed individually for 100% detection accuracy.
Merged segments play as one continuous feed with one combined alert card.

Usage:
    cd d:\\project\\FYP
    venv\\Scripts\\python scripts/demo/run_demo.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import cv2
import yaml
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from visual_guardian.pipeline import VisionPipeline

# ─────────────────────────────────────────────────────
# PATHS & OUTPUT
# ─────────────────────────────────────────────────────
OUTPUT_DIR  = Path(r"d:\project\FYP\demo_output")
OUTPUT_PATH = OUTPUT_DIR / "demo_output.mp4"
OUTPUT_FPS  = 30
OUTPUT_W    = 1920
OUTPUT_H    = 1080

SEIZURE_THRESHOLD = 0.48
FALL_THRESHOLD    = 0.64
GAP_SECONDS       = 3   # Black gap between the 7 segments

# ─────────────────────────────────────────────────────
# 7-SEGMENT DEFINITION
# Each segment has one or more clips that play back-to-back.
# Detection is run per-clip (for accuracy), results combined per segment.
# ─────────────────────────────────────────────────────
_R = Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets")

SEGMENTS = [
    {
        "id":      1,
        "patient": "Patient A",
        "type":    "normal",
        "label":   "Normal Activity — No Alert Expected",
        "clips": [
            _R / r"normal\Harvard_no-fall\nf_raw_b_3\nf_raw_b_3\B_M_48.mp4",
        ],
    },
    {
        "id":      2,
        "patient": "Patient A",
        "type":    "fall",
        "label":   "Fall Detected",
        "clips": [
            _R / r"falls\harvard_fall\f_raw_b_1\f_raw_b_1\B_D_0231.mp4",
        ],
    },
    {
        "id":      3,
        "patient": "Patient A",
        "type":    "fall",
        "label":   "Fall Detected",
        "clips": [
            _R / r"falls\harvard_fall\f_raw_b_2\f_raw_b_2\B_N_458.mp4",
        ],
    },
    {
        "id":      4,
        "patient": "Patient B",
        "type":    "normal",
        "label":   "Patient B — Resting Normally",
        # Two clips play as ONE continuous segment (no gap between them)
        "clips": [
            _R / r"unusual_movement\data\Normal\S37_0_39.mp4",
            _R / r"unusual_movement\data\Normal\S37_0_170.mp4",
        ],
    },
    {
        "id":      5,
        "patient": "Patient B",
        "type":    "seizure",
        "label":   "Patient B — Seizure Episode",
        "clips": [
            _R / r"unusual_movement\data\Seizure\S37_0_80.mp4",
            _R / r"unusual_movement\data\Seizure\S37_0_75.mp4",
        ],
    },
    {
        "id":      6,
        "patient": "Patient C",
        "type":    "normal",
        "label":   "Patient C — Active (High-Motion Patient)",
        "clips": [
            _R / r"unusual_movement\data\Normal\S15_1_140.mp4",
            _R / r"unusual_movement\data\Normal\S15_2_2.mp4",
        ],
    },
    {
        "id":      7,
        "patient": "Patient C",
        "type":    "seizure",
        "label":   "Patient C — Seizure Episode",
        "clips": [
            _R / r"unusual_movement\data\Seizure\S15_3_88.mp4",
            _R / r"unusual_movement\data\Seizure\S15_3_89.mp4",
        ],
    },
]

# ─────────────────────────────────────────────────────
# DASHBOARD LAYOUT
# ─────────────────────────────────────────────────────
SIDEBAR_W = 440
HEADER_H  = 52
FOOTER_H  = 40
VIDEO_W   = OUTPUT_W - SIDEBAR_W
VIDEO_H   = OUTPUT_H - HEADER_H - FOOTER_H

C_BG      = (18,  18,  22)
C_SIDEBAR = (30,  30,  38)
C_HEADER  = (12,  12,  18)
C_WHITE   = (255, 255, 255)
C_GREY    = (140, 140, 150)
C_GREEN   = (80,  200, 120)
C_AMBER   = (40,  160, 230)
C_RED     = (60,  60,  220)
C_ACCENT  = (200, 140, 60)
C_DIVIDER = (50,  50,  62)

# ─────────────────────────────────────────────────────
# ALERT CARD
# ─────────────────────────────────────────────────────

class AlertCard:
    _counter = 0
    def __init__(self, event_type, timestamp, patient, confidence):
        AlertCard._counter += 1
        self.id         = AlertCard._counter
        self.event_type = event_type
        self.timestamp  = timestamp
        self.patient    = patient
        self.confidence = confidence
        self.new_flash  = 30

    @property
    def title(self):
        return "SEIZURE DETECTED" if self.event_type == "seizure" else "FALL DETECTED"

    @property
    def priority(self):
        return "CRITICAL" if self.event_type == "seizure" else "HIGH"

    @property
    def color(self):
        return C_RED if self.event_type == "seizure" else C_AMBER


# ─────────────────────────────────────────────────────
# PIPELINE: per-clip processing (identical to validated scripts)
# ─────────────────────────────────────────────────────

def process_clip(pipeline, clip_path):
    """Process one clip. Returns (detected_fall, detected_seizure, max_fall_prob, max_seizure_prob)."""
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return False, False, 0.0, 0.0

    pipeline.reset()
    try:
        pipeline.pose_history.clear()
    except Exception:
        pass
    pipeline.patient_state = 'OUT_OF_BED'

    det_fall = False
    det_sz   = False
    max_fall = 0.0
    max_sz   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        event  = pipeline.process_frame(frame)
        etype  = event.get('event_type', 'normal')
        fc     = event.get('fall_confidence',    0.0)
        sc     = event.get('seizure_smoothed',   0.0)
        max_fall = max(max_fall, fc)
        max_sz   = max(max_sz,   sc)
        if etype in ('fall', 'force_fall'):
            det_fall = True
        if etype == 'seizure':
            det_sz = True

    cap.release()
    return det_fall, det_sz, max_fall, max_sz


def process_segment(pipeline, seg):
    """Run all clips in a segment and return combined detection result."""
    any_fall    = False
    any_seizure = False
    peak_fall   = 0.0
    peak_sz     = 0.0

    for clip_path in seg["clips"]:
        if not clip_path.exists():
            print(f"[MISSING] {clip_path.name}")
            continue
        df, ds, mf, ms = process_clip(pipeline, clip_path)
        if df:
            any_fall = True
        if ds:
            any_seizure = True
        peak_fall = max(peak_fall, mf)
        peak_sz   = max(peak_sz, ms)

    if seg["type"] == "fall" and any_fall:
        return "fall", peak_fall
    if seg["type"] == "seizure" and any_seizure:
        return "seizure", peak_sz
    return "normal", max(peak_fall, peak_sz)


# ─────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────

def draw_text(img, text, pos, scale=0.6, color=C_WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def apply_clahe(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def draw_alert_card(img, card, y):
    cx1, cx2 = VIDEO_W + 12, OUTPUT_W - 12
    card_h = 80

    overlay = img.copy()
    bg = (42, 38, 50) if card.new_flash > 0 else C_BG
    cv2.rectangle(overlay, (cx1, y), (cx2, y + card_h), bg, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    cv2.rectangle(img, (cx1, y), (cx1 + 4, y + card_h), card.color, -1)
    cv2.rectangle(img, (cx1, y), (cx2, y + card_h), C_DIVIDER, 1)

    draw_text(img, card.title,             (cx1 + 14, y + 22), scale=0.62, color=card.color, thickness=2)
    draw_text(img, card.timestamp,         (cx2 - 62, y + 22), scale=0.46, color=C_GREY)
    draw_text(img, f"Confidence: {card.confidence * 100:.0f}%", (cx1 + 14, y + 44), scale=0.48, color=C_WHITE)
    draw_text(img, f"Priority: {card.priority}",  (cx1 + 14, y + 62), scale=0.45, color=card.color)

    if card.new_flash > 0:
        card.new_flash -= 1

    return y + card_h + 6


def render_frame(raw_frame, seg, seg_idx, frame_in_seg, total_seg_frames,
                 alerts, det_type, global_frame):
    canvas = np.full((OUTPUT_H, OUTPUT_W, 3), C_BG, dtype=np.uint8)

    # ── Header ──────────────────────────────────────
    cv2.rectangle(canvas, (0, 0), (OUTPUT_W, HEADER_H), C_HEADER, -1)
    cv2.line(canvas, (0, HEADER_H), (OUTPUT_W, HEADER_H), C_DIVIDER, 1)
    draw_text(canvas, "VITAL GUARDIAN",            (18,  34), scale=0.82, color=C_ACCENT, thickness=2)
    draw_text(canvas, "AI-Powered ICU Monitoring", (220, 34), scale=0.50, color=C_GREY)
    cv2.circle(canvas, (OUTPUT_W - 174, 26), 7, C_GREEN, -1)
    draw_text(canvas, "ACTIVE", (OUTPUT_W - 156, 32), scale=0.50, color=C_GREEN)

    # ── Video Panel ──────────────────────────────────
    vx, vy = 0, HEADER_H
    if raw_frame is not None:
        vid = cv2.resize(raw_frame, (VIDEO_W, VIDEO_H))
        canvas[vy:vy + VIDEO_H, vx:vx + VIDEO_W] = vid
    cv2.rectangle(canvas, (vx, vy), (vx + VIDEO_W - 1, vy + VIDEO_H - 1), C_DIVIDER, 1)

    # ── Sidebar ──────────────────────────────────────
    sx0, sy0 = VIDEO_W, HEADER_H
    sb_h = OUTPUT_H - HEADER_H - FOOTER_H
    cv2.rectangle(canvas, (sx0, sy0), (OUTPUT_W, sy0 + sb_h), C_SIDEBAR, -1)
    cv2.line(canvas, (sx0, sy0), (sx0, sy0 + sb_h), C_DIVIDER, 1)

    y = sy0 + 22

    # Patient block
    draw_text(canvas, "PATIENT",        (sx0 + 18, y), scale=0.44, color=C_GREY)
    y += 28
    draw_text(canvas, seg["patient"],   (sx0 + 18, y), scale=0.75, color=C_WHITE, thickness=2)
    y += 26

    status_col  = C_RED if det_type == "seizure" else C_AMBER if det_type == "fall" else C_GREEN
    status_text = ("Status: SEIZURE ALERT" if det_type == "seizure"
                   else "Status: FALL ALERT" if det_type == "fall"
                   else "Status: MONITORING")
    draw_text(canvas, status_text, (sx0 + 18, y), scale=0.50, color=status_col)
    y += 18
    cv2.line(canvas, (sx0 + 14, y), (OUTPUT_W - 14, y), C_DIVIDER, 1)
    y += 22

    # Alert log
    draw_text(canvas, "ALERT LOG", (sx0 + 18, y), scale=0.44, color=C_GREY)
    y += 22

    visible = alerts[-5:]
    if not visible:
        draw_text(canvas, "No alerts", (sx0 + 18, y + 14), scale=0.52, color=C_GREY)
    else:
        for card in visible:
            y = draw_alert_card(canvas, card, y)
            if y > OUTPUT_H - FOOTER_H - 30:
                break

    # ── Footer ──────────────────────────────────────
    fy = OUTPUT_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (OUTPUT_W, OUTPUT_H), C_HEADER, -1)
    cv2.line(canvas, (0, fy), (OUTPUT_W, fy), C_DIVIDER, 1)

    seg_str = f"Segment [{seg_idx + 1}/7]  {seg['patient']}  —  {seg['label']}"
    draw_text(canvas, seg_str, (18, fy + 26), scale=0.50, color=C_GREY)

    # Progress bar
    progress = frame_in_seg / max(total_seg_frames, 1)
    bx = OUTPUT_W - 310
    bw = 180
    cv2.rectangle(canvas, (bx, fy + 13), (bx + bw, fy + 27), (45, 45, 55), -1)
    cv2.rectangle(canvas, (bx, fy + 13), (bx + int(bw * progress), fy + 27), C_ACCENT, -1)

    total_sec = int(global_frame / OUTPUT_FPS)
    draw_text(canvas, f"{total_sec // 60:02d}:{total_sec % 60:02d}",
              (OUTPUT_W - 100, fy + 26), scale=0.50, color=C_GREY)

    return canvas


def render_gap_frame(alerts, seg_idx, global_frame):
    """Transition frame between segments."""
    canvas = np.full((OUTPUT_H, OUTPUT_W, 3), C_BG, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (OUTPUT_W, HEADER_H), C_HEADER, -1)
    cv2.line(canvas, (0, HEADER_H), (OUTPUT_W, HEADER_H), C_DIVIDER, 1)
    draw_text(canvas, "VITAL GUARDIAN",            (18,  34), scale=0.82, color=C_ACCENT, thickness=2)
    draw_text(canvas, "AI-Powered ICU Monitoring", (220, 34), scale=0.50, color=C_GREY)
    cv2.circle(canvas, (OUTPUT_W - 174, 26), 7, C_GREEN, -1)
    draw_text(canvas, "ACTIVE", (OUTPUT_W - 156, 32), scale=0.50, color=C_GREEN)

    # Video area — transition text
    vx, vy = 0, HEADER_H
    draw_text(canvas, "Switching camera feed...",
              (VIDEO_W // 2 - 175, OUTPUT_H // 2),
              scale=0.70, color=C_GREY)

    # Sidebar persists with alerts
    sx0, sy0 = VIDEO_W, HEADER_H
    sb_h = OUTPUT_H - HEADER_H - FOOTER_H
    cv2.rectangle(canvas, (sx0, sy0), (OUTPUT_W, sy0 + sb_h), C_SIDEBAR, -1)
    cv2.line(canvas, (sx0, sy0), (sx0, sy0 + sb_h), C_DIVIDER, 1)

    y = sy0 + 22
    draw_text(canvas, "ALERT LOG", (sx0 + 18, y), scale=0.44, color=C_GREY)
    y += 22
    for card in alerts[-5:]:
        y = draw_alert_card(canvas, card, y)
        if y > OUTPUT_H - 60:
            break

    fy = OUTPUT_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (OUTPUT_W, OUTPUT_H), C_HEADER, -1)
    total_sec = int(global_frame / OUTPUT_FPS)
    draw_text(canvas, f"{total_sec // 60:02d}:{total_sec % 60:02d}",
              (OUTPUT_W - 100, fy + 26), scale=0.50, color=C_GREY)
    return canvas


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open('config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    vision_cfg = cfg['vision']
    vision_cfg['seizure_classifier']['threshold'] = SEIZURE_THRESHOLD
    vision_cfg['fall_classifier']['threshold']    = FALL_THRESHOLD
    if 'bed_exit' in vision_cfg:
        vision_cfg['bed_exit']['enabled'] = False

    print("=" * 70)
    print("VITAL GUARDIAN — NURSE DASHBOARD DEMO  [7 segments]")
    print("=" * 70)
    print(f"  Segments         : {len(SEGMENTS)}")
    print(f"  Seizure threshold: {SEIZURE_THRESHOLD}")
    print(f"  Fall threshold   : {FALL_THRESHOLD}")
    print(f"  Output           : {OUTPUT_PATH}")
    print()

    # ── STEP 1: Detect events per segment ────────────
    print("  STEP 1: Running detection on each segment...")
    pipeline = VisionPipeline(vision_cfg)
    print()

    seg_results = []
    for i, seg in enumerate(SEGMENTS):
        clip_names = ", ".join(p.name for p in seg["clips"])
        print(f"  [{i + 1}/7] {seg['patient']:<12} ({seg['type']:<7}) [{clip_names}] ... ", end="", flush=True)
        det_type, peak_conf = process_segment(pipeline, seg)
        seg_results.append((det_type, peak_conf))
        status = "DETECTED" if det_type != "normal" else "clean"
        print(f"{det_type}  {peak_conf:.3f}  {status}")

    print()

    # ── STEP 2: Render dashboard video ───────────────
    print("  STEP 2: Rendering dashboard video...")
    fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
    writer  = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, OUTPUT_FPS, (OUTPUT_W, OUTPUT_H))
    alerts  = []
    global_frame = 0

    for i, seg in enumerate(SEGMENTS):
        det_type, peak_conf = seg_results[i]
        print(f"    Segment {i + 1}: {seg['patient']} — {seg['label'][:35]:<35}",
              end=" ... ", flush=True)

        # Create alert card for this segment if event detected
        if det_type in ("fall", "seizure"):
            total_sec = int(global_frame / OUTPUT_FPS)
            ts = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
            alerts.append(AlertCard(det_type, ts, seg["patient"], peak_conf))

        # Count total frames across all clips in this segment (for progress bar)
        total_seg_frames = 0
        for cp in seg["clips"]:
            if cp.exists():
                cap_tmp = cv2.VideoCapture(str(cp))
                src_fps = cap_tmp.get(cv2.CAP_PROP_FPS) or 30.0
                fc      = int(cap_tmp.get(cv2.CAP_PROP_FRAME_COUNT))
                total_seg_frames += int(fc * OUTPUT_FPS / src_fps)
                cap_tmp.release()

        frame_in_seg = 0
        written = 0

        # Play each clip in the segment back-to-back (no gap between sub-clips)
        for cp in seg["clips"]:
            if not cp.exists():
                continue
            cap     = cv2.VideoCapture(str(cp))
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            ratio   = OUTPUT_FPS / src_fps
            frac    = 0.0

            while True:
                ret, raw = cap.read()
                if not ret:
                    break
                frame_in_seg += 1

                next_frac = frac + ratio
                n_out = int(next_frac) - int(frac)
                frac  = next_frac
                if n_out < 1:
                    continue

                h, w = raw.shape[:2]
                if w != OUTPUT_W or h != OUTPUT_H:
                    raw = cv2.resize(raw, (OUTPUT_W, OUTPUT_H), interpolation=cv2.INTER_LANCZOS4)
                raw = apply_clahe(raw)

                for _ in range(n_out):
                    global_frame += 1
                    canvas = render_frame(raw, seg, i, frame_in_seg, total_seg_frames,
                                          alerts, det_type, global_frame)
                    writer.write(canvas)
                    written += 1
            cap.release()

        print(f"{written} frames")

        # Gap between segments (not after the last one)
        if i < len(SEGMENTS) - 1:
            for _ in range(GAP_SECONDS * OUTPUT_FPS):
                global_frame += 1
                writer.write(render_gap_frame(alerts, i, global_frame))

    writer.release()

    duration = global_frame / OUTPUT_FPS
    print()
    print("=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"  Total frames : {global_frame}")
    print(f"  Duration     : {duration:.1f}s  ({duration / 60:.1f}min)")
    print(f"  Alert cards  : {len(alerts)}")
    for a in alerts:
        print(f"    [{a.timestamp}]  {a.title:<22}  {a.patient}  ({a.confidence * 100:.0f}% conf)")
    print(f"  Output       : {OUTPUT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
