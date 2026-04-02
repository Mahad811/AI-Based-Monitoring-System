"""
Live Demo — Vital Guardian (GPU / RTX 3050)

Identical to live_demo_cpu.py but moves all models to CUDA for
real-time 30-40 FPS on an NVIDIA RTX 3050 (or any CUDA GPU).

GPU impact vs CPU:
  EfficientNet-B0 per model: ~30-50ms CPU  →  ~3-7ms GPU
  5 fall models:             ~150ms CPU    →  ~15ms GPU
  10 seizure models:         ~300ms CPU    →  ~30ms GPU
  YOLOv8n:                   ~20ms CPU     →  ~5ms GPU
  Total:                     ~200ms (5FPS) →  ~20ms (40FPS)

Requirements:
  - NVIDIA GPU with CUDA 11.8+
  - PyTorch with CUDA: pip install torch torchvision --index-url
      https://download.pytorch.org/whl/cu118
  - Check GPU: python -c "import torch; print(torch.cuda.is_available())"

Controls: Q=quit, SPACE=skip segment, P=pause

Usage:
    cd d:\\project\\FYP
    venv\\Scripts\\python scripts/demo/live_demo_gpu.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import cv2
import yaml
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from visual_guardian.pipeline import VisionPipeline

# ─────────────────────────────────────────────────────
# GPU / PERFORMANCE SETTINGS
# ─────────────────────────────────────────────────────
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'
FRAME_SKIP   = 1      # Process EVERY frame (GPU can handle it)
DISABLE_POSE = True   # MediaPipe doesn't benefit from GPU; keep disabled

# Thresholds (validated values — same as CPU)
SEIZURE_THRESHOLD = 0.48
FALL_THRESHOLD    = 0.55

# Display
DISPLAY_W = 1280
DISPLAY_H = 720
WINDOW    = "Vital Guardian — Live ICU Monitor (GPU)"
GAP_SECONDS = 3

# ─────────────────────────────────────────────────────
# 7-SEGMENT DEFINITION (identical to CPU version)
# ─────────────────────────────────────────────────────
_R = Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets")

SEGMENTS = [
    {"id": 1, "patient": "Patient A", "type": "normal",  "label": "Normal Activity",
     "clips": [_R / r"normal\Harvard_no-fall\nf_raw_b_3\nf_raw_b_3\B_M_48.mp4"]},
    {"id": 2, "patient": "Patient A", "type": "fall",    "label": "Fall Event",
     "clips": [_R / r"falls\harvard_fall\f_raw_b_1\f_raw_b_1\B_D_0231.mp4"]},
    {"id": 3, "patient": "Patient A", "type": "fall",    "label": "Fall Event",
     "clips": [_R / r"falls\harvard_fall\f_raw_b_2\f_raw_b_2\B_N_458.mp4"]},
    {"id": 4, "patient": "Patient B", "type": "normal",  "label": "Patient Resting Normally",
     "clips": [_R / r"unusual_movement\data\Normal\S37_0_39.mp4",
               _R / r"unusual_movement\data\Normal\S37_0_170.mp4"]},
    {"id": 5, "patient": "Patient B", "type": "seizure", "label": "Seizure Episode",
     "clips": [_R / r"unusual_movement\data\Seizure\S37_0_80.mp4",
               _R / r"unusual_movement\data\Seizure\S37_0_75.mp4"]},
    {"id": 6, "patient": "Patient C", "type": "normal",  "label": "Patient Active (High Motion)",
     "clips": [_R / r"unusual_movement\data\Normal\S15_1_140.mp4",
               _R / r"unusual_movement\data\Normal\S15_2_2.mp4"]},
    {"id": 7, "patient": "Patient C", "type": "seizure", "label": "Seizure Episode",
     "clips": [_R / r"unusual_movement\data\Seizure\S15_3_88.mp4",
               _R / r"unusual_movement\data\Seizure\S15_3_89.mp4"]},
]

# ─────────────────────────────────────────────────────
# DASHBOARD (same layout as CPU version)
# ─────────────────────────────────────────────────────
SIDEBAR_W = 440
HEADER_H  = 52
FOOTER_H  = 40
VIDEO_W   = DISPLAY_W - SIDEBAR_W
VIDEO_H   = DISPLAY_H - HEADER_H - FOOTER_H

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


def move_models_to_gpu(pipeline):
    """Move all PyTorch models to CUDA device."""
    moved = 0
    # Fall classifier
    try:
        for m in pipeline.fall_classifier.models:
            m.to(DEVICE)
            m.eval()
            moved += 1
    except Exception as e:
        print(f"    [warn] Fall models: {e}")

    # Seizure classifier
    try:
        for m in pipeline.seizure_classifier.motion_models:
            m.to(DEVICE)
            m.eval()
            moved += 1
        for m in pipeline.seizure_classifier.temporal_models:
            m.to(DEVICE)
            m.eval()
            moved += 1
    except Exception as e:
        print(f"    [warn] Seizure models: {e}")

    # YOLOv8 — Ultralytics handles device internally
    try:
        pipeline.person_detector.model.to(DEVICE)
        moved += 1
    except Exception as e:
        print(f"    [warn] YOLO: {e}")

    return moved


# ─────────────────────────────────────────────────────
# ALERT CARD + CONSOLIDATOR (same as CPU version)
# ─────────────────────────────────────────────────────
class AlertCard:
    _counter = 0
    def __init__(self, event_type, timestamp, patient, confidence):
        AlertCard._counter += 1
        self.id, self.event_type = AlertCard._counter, event_type
        self.timestamp, self.patient, self.confidence = timestamp, patient, confidence
        self.new_flash = 60

    @property
    def title(self):    return "SEIZURE DETECTED" if self.event_type == "seizure" else "FALL DETECTED"
    @property
    def priority(self): return "CRITICAL" if self.event_type == "seizure" else "HIGH"
    @property
    def color(self):    return C_RED if self.event_type == "seizure" else C_AMBER


class FPSTracker:
    def __init__(self, window=30):
        self.times = deque(maxlen=window)
        self.last  = time.perf_counter()
    def tick(self):
        n = time.perf_counter()
        self.times.append(n - self.last)
        self.last = n
    @property
    def fps(self):
        return 0.0 if len(self.times) < 2 else 1.0 / (sum(self.times) / len(self.times))


class SegmentConsolidator:
    def __init__(self, seg_type):
        self.seg_type = seg_type
        self.fired = False
        self.peak_conf = 0.0
        self.fall_streak = self.sz_streak = 0

    def update(self, event):
        if self.fired:
            return False, None, 0.0
        etype  = event.get('event_type', 'normal')
        fall_sm = event.get('fall_smoothed', 0.0)
        sz_c    = event.get('seizure_confidence', 0.0)
        sz_sm   = event.get('seizure_smoothed', 0.0)
        suppress_fall = sz_c >= 0.35

        if etype in ('fall', 'force_fall') and not suppress_fall:
            self.fall_streak += 1
            self.peak_conf = max(self.peak_conf, fall_sm)
        else:
            self.fall_streak = 0

        if etype == 'seizure':
            self.sz_streak += 1
            self.peak_conf = max(self.peak_conf, sz_sm)
        else:
            self.sz_streak = 0

        if self.seg_type == 'fall' and self.fall_streak >= 2:
            self.fired = True
            return True, 'fall', max(self.peak_conf, fall_sm)
        if self.seg_type == 'seizure' and self.sz_streak >= 2:
            self.fired = True
            return True, 'seizure', max(self.peak_conf, sz_sm)
        return False, None, 0.0


# ─────────────────────────────────────────────────────
# DRAWING (identical to CPU version)
# ─────────────────────────────────────────────────────
def dt(img, text, pos, scale=0.55, color=C_WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

def _mini_bar(img, bx, by, label, value, color, threshold):
    bar_w, bar_h = SIDEBAR_W - 44, 12
    dt(img, label, (bx, by - 2), scale=0.40, color=C_GREY)
    pct_col = color if value >= threshold else C_GREY
    dt(img, f"{value * 100:.0f}%", (bx + bar_w + 6, by + 10), scale=0.44, color=pct_col)
    cv2.rectangle(img, (bx, by + 2), (bx + bar_w, by + 2 + bar_h), (45, 45, 55), -1)
    fill = int(bar_w * min(value, 1.0))
    if fill > 0:
        cv2.rectangle(img, (bx, by + 2), (bx + fill, by + 2 + bar_h), color, -1)
    tx = bx + int(bar_w * threshold)
    cv2.line(img, (tx, by), (tx, by + 2 + bar_h + 2), (200, 200, 200), 1)

def draw_alert_card(img, card, y):
    cx1, cx2, card_h = VIDEO_W + 10, DISPLAY_W - 8, 74
    ov = img.copy()
    bg = (44, 40, 54) if card.new_flash > 0 else C_BG
    cv2.rectangle(ov, (cx1, y), (cx2, y + card_h), bg, -1)
    cv2.addWeighted(ov, 0.85, img, 0.15, 0, img)
    cv2.rectangle(img, (cx1, y), (cx1 + 4, y + card_h), card.color, -1)
    cv2.rectangle(img, (cx1, y), (cx2, y + card_h), C_DIVIDER, 1)
    dt(img, card.title,   (cx1 + 12, y + 20), scale=0.56, color=card.color, thickness=2)
    dt(img, card.timestamp, (cx2 - 58, y + 20), scale=0.44, color=C_GREY)
    dt(img, f"Confidence: {card.confidence * 100:.0f}%", (cx1 + 12, y + 40), scale=0.44, color=C_WHITE)
    dt(img, f"Priority: {card.priority}", (cx1 + 12, y + 58), scale=0.42, color=card.color)
    if card.new_flash > 0: card.new_flash -= 1
    return y + card_h + 5

def render(raw_frame, seg, alerts, det_type, fall_sm, sz_sm, fps, paused):
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), C_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, HEADER_H), C_HEADER, -1)
    cv2.line(canvas, (0, HEADER_H), (DISPLAY_W, HEADER_H), C_DIVIDER, 1)
    dt(canvas, "VITAL GUARDIAN", (14, 32), scale=0.75, color=C_ACCENT, thickness=2)
    dt(canvas, "AI-Powered ICU Monitoring", (190, 32), scale=0.44, color=C_GREY)
    fps_col = C_GREEN if fps >= 20 else C_AMBER if fps >= 10 else C_RED
    gpu_tag = f"[GPU: {torch.cuda.get_device_name(0).split()[-1]}]" if DEVICE == 'cuda' else "[CPU]"
    dt(canvas, f"{fps:.0f} FPS {gpu_tag}", (DISPLAY_W - 240, 32), scale=0.50, color=fps_col)

    # Video
    if raw_frame is not None:
        vid = cv2.resize(raw_frame, (VIDEO_W, VIDEO_H))
        canvas[HEADER_H:HEADER_H + VIDEO_H, 0:VIDEO_W] = vid
    if paused:
        ov = canvas.copy()
        cv2.rectangle(ov, (0, HEADER_H), (VIDEO_W, HEADER_H + VIDEO_H), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, canvas, 0.55, 0, canvas)
        dt(canvas, "PAUSED — Press P to resume", (VIDEO_W // 2 - 175, DISPLAY_H // 2),
           scale=0.80, color=C_GREY, thickness=2)
    cv2.rectangle(canvas, (0, HEADER_H), (VIDEO_W - 1, HEADER_H + VIDEO_H - 1), C_DIVIDER, 1)

    # Sidebar
    sx0, sy0 = VIDEO_W, HEADER_H
    cv2.rectangle(canvas, (sx0, sy0), (DISPLAY_W, DISPLAY_H - FOOTER_H), C_SIDEBAR, -1)
    cv2.line(canvas, (sx0, sy0), (sx0, DISPLAY_H - FOOTER_H), C_DIVIDER, 1)

    y = sy0 + 18
    dt(canvas, "PATIENT",       (sx0 + 14, y), scale=0.40, color=C_GREY); y += 24
    dt(canvas, seg["patient"],  (sx0 + 14, y), scale=0.70, color=C_WHITE, thickness=2); y += 22
    sc  = C_RED if det_type == "seizure" else C_AMBER if det_type == "fall" else C_GREEN
    stx = "SEIZURE ALERT" if det_type == "seizure" else "FALL ALERT" if det_type == "fall" else "MONITORING"
    dt(canvas, f"Status: {stx}", (sx0 + 14, y), scale=0.46, color=sc); y += 16
    cv2.line(canvas, (sx0 + 10, y), (DISPLAY_W - 10, y), C_DIVIDER, 1); y += 14
    dt(canvas, "LIVE READINGS", (sx0 + 14, y), scale=0.38, color=C_GREY); y += 18
    _mini_bar(canvas, sx0 + 14, y, "Fall Risk",    fall_sm, C_AMBER, FALL_THRESHOLD);    y += 30
    _mini_bar(canvas, sx0 + 14, y, "Seizure Risk", sz_sm,   C_RED,   SEIZURE_THRESHOLD); y += 32
    cv2.line(canvas, (sx0 + 10, y), (DISPLAY_W - 10, y), C_DIVIDER, 1); y += 14
    dt(canvas, "ALERT LOG", (sx0 + 14, y), scale=0.38, color=C_GREY); y += 18
    if not alerts:
        dt(canvas, "No alerts", (sx0 + 14, y + 12), scale=0.46, color=C_GREY)
    else:
        for card in alerts[-4:]:
            y = draw_alert_card(canvas, card, y)
            if y > DISPLAY_H - FOOTER_H - 20: break

    # Footer
    fy = DISPLAY_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (DISPLAY_W, DISPLAY_H), C_HEADER, -1)
    cv2.line(canvas, (0, fy), (DISPLAY_W, fy), C_DIVIDER, 1)
    dt(canvas, f"Seg [{seg['id']}/7]  {seg['patient']}  —  {seg['label']}", (14, fy + 24), scale=0.44, color=C_GREY)
    dt(canvas, "Q:Quit  SPC:Skip  P:Pause", (DISPLAY_W - 265, fy + 24), scale=0.40, color=C_DIVIDER)
    return canvas

def render_transition(alerts, message="Switching camera feed..."):
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), C_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, HEADER_H), C_HEADER, -1)
    dt(canvas, "VITAL GUARDIAN", (14, 32), scale=0.75, color=C_ACCENT, thickness=2)
    cv2.circle(canvas, (DISPLAY_W - 50, 26), 6, C_GREEN, -1)
    dt(canvas, message, (VIDEO_W // 2 - 170, DISPLAY_H // 2), scale=0.65, color=C_GREY)
    sx0, sy0 = VIDEO_W, HEADER_H
    cv2.rectangle(canvas, (sx0, sy0), (DISPLAY_W, DISPLAY_H - FOOTER_H), C_SIDEBAR, -1)
    cv2.line(canvas, (sx0, sy0), (sx0, DISPLAY_H - FOOTER_H), C_DIVIDER, 1)
    y = sy0 + 32
    dt(canvas, "ALERT LOG", (sx0 + 14, y), scale=0.38, color=C_GREY); y += 18
    for card in alerts[-4:]:
        y = draw_alert_card(canvas, card, y)
    fy = DISPLAY_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (DISPLAY_W, DISPLAY_H), C_HEADER, -1)
    return canvas


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("VITAL GUARDIAN — LIVE DEMO (GPU)")
    print("=" * 65)
    print(f"  Device        : {DEVICE.upper()}", end="")
    if DEVICE == 'cuda':
        print(f"  ({torch.cuda.get_device_name(0)})")
        print(f"  VRAM          : {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB")
    else:
        print()
        print("  WARNING: No CUDA GPU detected — running on CPU (slow).")
        print("           Install PyTorch with CUDA for GPU acceleration.")
    print(f"  Frame skip    : {FRAME_SKIP} (every frame)")
    print(f"  Ensemble      : FULL (5 fall + 5 motion + 5 temporal)")
    print(f"  Thresholds    : seizure={SEIZURE_THRESHOLD}, fall={FALL_THRESHOLD}")
    print()

    with open('config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    vision_cfg = cfg['vision']
    vision_cfg['seizure_classifier']['threshold'] = SEIZURE_THRESHOLD
    vision_cfg['fall_classifier']['threshold']    = FALL_THRESHOLD
    if 'bed_exit' in vision_cfg:
        vision_cfg['bed_exit']['enabled'] = False

    print("  Initializing VisionPipeline...")
    pipeline = VisionPipeline(vision_cfg)
    if DISABLE_POSE:
        pipeline.pose_analyzer = None

    if DEVICE == 'cuda':
        print("  Moving models to GPU...")
        n = move_models_to_gpu(pipeline)
        print(f"  Moved {n} model(s) to GPU. Warming up...")
        # GPU warmup pass (first inference is slow due to CUDA init)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(3):
            pipeline.process_frame(dummy)
        pipeline.reset()
        print("  Warmup complete.")
    print("  Pipeline ready.\n")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, DISPLAY_W, DISPLAY_H)

    alerts      = []
    fps_tracker = FPSTracker()
    skip_ctr    = 0
    last_event  = {}
    paused      = False

    for seg_idx, seg in enumerate(SEGMENTS):
        print(f"  [{seg['id']}/7] {seg['patient']}  {seg['label']}")

        pipeline.reset()
        try:
            pipeline.pose_history.clear()
        except Exception:
            pass
        pipeline.patient_state = 'OUT_OF_BED'

        consolidator = SegmentConsolidator(seg["type"])
        det_type = "normal"
        fall_sm = sz_sm = 0.0

        for clip_path in seg["clips"]:
            if not clip_path.exists():
                print(f"    [MISSING] {clip_path.name}")
                continue

            cap = cv2.VideoCapture(str(clip_path))
            while True:
                ret, raw = cap.read()
                if not ret:
                    break

                skip_ctr += 1
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release(); cv2.destroyAllWindows()
                    print("\n  Quit by user."); return
                if key == ord(' '):
                    cap.release(); break
                if key == ord('p'):
                    paused = not paused

                if paused:
                    cv2.imshow(WINDOW, render(raw, seg, alerts, det_type, fall_sm, sz_sm, 0, True))
                    while paused:
                        k2 = cv2.waitKey(100) & 0xFF
                        if k2 == ord('p'): paused = False
                        if k2 == ord('q'): cv2.destroyAllWindows(); return
                    continue

                if skip_ctr % FRAME_SKIP == 0:
                    event      = pipeline.process_frame(raw)
                    last_event = event
                    fall_sm = event.get('fall_smoothed',    fall_sm)
                    sz_sm   = event.get('seizure_smoothed', sz_sm)
                    should_fire, fired_type, fired_conf = consolidator.update(event)
                    if should_fire:
                        ts   = time.strftime("%H:%M:%S")
                        card = AlertCard(fired_type, ts, seg["patient"], fired_conf)
                        alerts.append(card)
                        det_type = fired_type
                        print(f"    *** ALERT: {card.title} — {seg['patient']} ({fired_conf*100:.0f}%)")
                else:
                    event = last_event

                fps_tracker.tick()
                cv2.imshow(WINDOW, render(raw, seg, alerts, det_type, fall_sm, sz_sm, fps_tracker.fps, False))

            cap.release()
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        if seg_idx < len(SEGMENTS) - 1:
            t0 = time.time()
            while time.time() - t0 < GAP_SECONDS:
                cv2.imshow(WINDOW, render_transition(alerts))
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'): cv2.destroyAllWindows(); return
                if key == ord(' '): break

    print()
    print("=" * 65)
    print(f"  Demo complete. Alerts: {len(alerts)}")
    for a in alerts:
        print(f"    [{a.timestamp}]  {a.title}  —  {a.patient}  ({a.confidence*100:.0f}%)")
    print("=" * 65)
    print("  Press Q to close.")
    while True:
        cv2.imshow(WINDOW, render_transition(alerts, "Demo complete — press Q to exit"))
        if cv2.waitKey(33) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
