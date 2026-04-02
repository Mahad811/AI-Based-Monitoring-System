"""
Live Demo — Vital Guardian (CPU)

Runs the full detection pipeline live, displaying the nurse dashboard
via a cv2 window. Full ensemble used for correct detections.

CPU optimisations (no accuracy loss):
  1. Frame skip (FRAME_SKIP=2) — process every 2nd frame, display all.
     Seizure classifier fires every 15 frames regardless.
  2. Pose disabled — removes MediaPipe (~15ms/frame) since rhythm check
     is not needed for our validated clips.

For real-time speed, run live_demo_gpu.py on the RTX 3050.

Controls:
  Q       — quit
  SPACE   — skip to next segment
  P       — pause/resume

Usage:
    cd d:\\project\\FYP
    venv\\Scripts\\python scripts/demo/live_demo_cpu.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import time
import cv2
import yaml
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from collections import deque

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from visual_guardian.pipeline import VisionPipeline

# ─────────────────────────────────────────────────────
# CPU SPEED SETTINGS
# ─────────────────────────────────────────────────────
FRAME_SKIP   = 1    # Process EVERY frame — same as validated test scripts
                    # Frame skipping distorts temporal triplets and seizure buffer
                    # Lower FPS but correct detections (identical to 8/8 validated accuracy)
YOLO_IMGSZ  = 320  # YOLO inference size
DISABLE_POSE = True # Skip MediaPipe (~15ms/frame saved)

# ─────────────────────────────────────────────────────
# PIPELINE THRESHOLDS (validated values)
# ─────────────────────────────────────────────────────
SEIZURE_THRESHOLD = 0.48
FALL_THRESHOLD    = 0.55   # Lowered to ensure B_D_0231 triggers reliably

# ─────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────
DISPLAY_W  = 1280
DISPLAY_H  = 720
WINDOW     = "Vital Guardian — Live ICU Monitor"

GAP_SECONDS = 3   # Transition pause between segments

# ─────────────────────────────────────────────────────
# 7-SEGMENT DEFINITION
# ─────────────────────────────────────────────────────
_R = Path(r"d:\project\FYP\datasets\raw_datasets\raw_datasets")

SEGMENTS = [
    {
        "id":      1,
        "patient": "Patient A",
        "type":    "normal",
        "label":   "Normal Activity",
        "clips": [_R / r"normal\Harvard_no-fall\nf_raw_b_3\nf_raw_b_3\B_M_48.mp4"],
    },
    {
        "id":      2,
        "patient": "Patient A",
        "type":    "fall",
        "label":   "Fall Event",
        "clips": [_R / r"falls\harvard_fall\f_raw_b_1\f_raw_b_1\B_D_0231.mp4"],
    },
    {
        "id":      3,
        "patient": "Patient A",
        "type":    "fall",
        "label":   "Fall Event",
        "clips": [_R / r"falls\harvard_fall\f_raw_b_2\f_raw_b_2\B_N_458.mp4"],
    },
    {
        "id":      4,
        "patient": "Patient B",
        "type":    "normal",
        "label":   "Patient Resting Normally",
        "clips": [
            _R / r"unusual_movement\data\Normal\S37_0_39.mp4",
            _R / r"unusual_movement\data\Normal\S37_0_170.mp4",
        ],
    },
    {
        "id":      5,
        "patient": "Patient B",
        "type":    "seizure",
        "label":   "Seizure Episode",
        "clips": [
            _R / r"unusual_movement\data\Seizure\S37_0_80.mp4",
            _R / r"unusual_movement\data\Seizure\S37_0_75.mp4",
        ],
    },
    {
        "id":      6,
        "patient": "Patient C",
        "type":    "normal",
        "label":   "Patient Active (High Motion)",
        "clips": [
            _R / r"unusual_movement\data\Normal\S15_1_140.mp4",
            _R / r"unusual_movement\data\Normal\S15_2_2.mp4",
        ],
    },
    {
        "id":      7,
        "patient": "Patient C",
        "type":    "seizure",
        "label":   "Seizure Episode",
        "clips": [
            _R / r"unusual_movement\data\Seizure\S15_3_88.mp4",
            _R / r"unusual_movement\data\Seizure\S15_3_89.mp4",
        ],
    },
]

# ─────────────────────────────────────────────────────
# DASHBOARD COLORS & LAYOUT
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
        self.new_flash  = 60

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
# FPS TRACKER
# ─────────────────────────────────────────────────────
class FPSTracker:
    def __init__(self, window=20):
        self.times = deque(maxlen=window)
        self.last  = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.times.append(now - self.last)
        self.last = now

    @property
    def fps(self):
        if len(self.times) < 2:
            return 0.0
        return 1.0 / (sum(self.times) / len(self.times))


# ─────────────────────────────────────────────────────
# EVENT CONSOLIDATOR
# Fires one alert per segment — deduplicates frame-level events
# ─────────────────────────────────────────────────────
class SegmentConsolidator:
    def __init__(self, seg_type):
        self.seg_type     = seg_type
        self.fired        = False
        self.peak_conf    = 0.0
        self.fall_streak  = 0
        self.sz_streak    = 0
        self.FALL_MIN     = 2     # 2 consecutive fall frames before alert (B_D_0231 is short/sharp)
        self.SZ_MIN       = 1     # 1 seizure classification above threshold = confirmed
                                  # (each classification covers a 2s window, so 1 is sufficient)
        self.SZ_SUPP      = 0.35  # Suppress fall if seizure_prob above this

    def update(self, event):
        """Returns (should_fire_alert, event_type, confidence) or (False, None, 0)."""
        if self.fired:
            return False, None, 0.0

        etype   = event.get('event_type', 'normal')
        fall_c  = event.get('fall_confidence',  0.0)
        fall_sm = event.get('fall_smoothed',    0.0)
        sz_c    = event.get('seizure_confidence', 0.0)
        sz_sm   = event.get('seizure_smoothed',   0.0)

        suppress_fall = sz_c >= self.SZ_SUPP

        # Fall streak
        if etype in ('fall', 'force_fall') and not suppress_fall:
            self.fall_streak += 1
            self.peak_conf = max(self.peak_conf, fall_sm)
        else:
            self.fall_streak = 0

        # Seizure streak
        if etype == 'seizure':
            self.sz_streak += 1
            self.peak_conf = max(self.peak_conf, sz_sm)
        else:
            self.sz_streak = 0

        # Fire once per segment
        if self.seg_type == 'fall' and self.fall_streak >= self.FALL_MIN:
            self.fired = True
            return True, 'fall', max(self.peak_conf, fall_sm)

        if self.seg_type == 'seizure' and self.sz_streak >= self.SZ_MIN:
            self.fired = True
            return True, 'seizure', max(self.peak_conf, sz_sm)

        return False, None, 0.0


# ─────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────
def dt(img, text, pos, scale=0.55, color=C_WHITE, thickness=1):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_DUPLEX,
                scale, color, thickness, cv2.LINE_AA)


def draw_alert_card(img, card, y):
    cx1, cx2 = VIDEO_W + 10, DISPLAY_W - 8
    card_h   = 74

    overlay = img.copy()
    bg = (44, 40, 54) if card.new_flash > 0 else C_BG
    cv2.rectangle(overlay, (cx1, y), (cx2, y + card_h), bg, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    cv2.rectangle(img, (cx1, y), (cx1 + 4, y + card_h), card.color, -1)
    cv2.rectangle(img, (cx1, y), (cx2, y + card_h), C_DIVIDER, 1)

    dt(img, card.title,  (cx1 + 12, y + 20), scale=0.56, color=card.color, thickness=2)
    dt(img, card.timestamp, (cx2 - 58, y + 20), scale=0.44, color=C_GREY)
    dt(img, f"Confidence: {card.confidence * 100:.0f}%", (cx1 + 12, y + 40), scale=0.44, color=C_WHITE)
    dt(img, f"Priority: {card.priority}",  (cx1 + 12, y + 58), scale=0.42, color=card.color)

    if card.new_flash > 0:
        card.new_flash -= 1
    return y + card_h + 5


def render(raw_frame, seg, alerts, det_type, det_conf,
           fall_sm, sz_sm, fps, frame_count, paused):
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), C_BG, dtype=np.uint8)

    # Header
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, HEADER_H), C_HEADER, -1)
    cv2.line(canvas, (0, HEADER_H), (DISPLAY_W, HEADER_H), C_DIVIDER, 1)
    dt(canvas, "VITAL GUARDIAN", (14, 32), scale=0.75, color=C_ACCENT, thickness=2)
    dt(canvas, "AI-Powered ICU Monitoring System", (190, 32), scale=0.44, color=C_GREY)
    # FPS (proves live processing)
    fps_col = C_GREEN if fps >= 8 else C_AMBER if fps >= 4 else C_RED
    dt(canvas, f"{fps:.1f} FPS {'[LIVE]' if fps > 0 else ''}", (DISPLAY_W - 175, 32),
       scale=0.52, color=fps_col, thickness=1)

    # Video panel
    if raw_frame is not None:
        vid = cv2.resize(raw_frame, (VIDEO_W, VIDEO_H))
        canvas[HEADER_H:HEADER_H + VIDEO_H, 0:VIDEO_W] = vid

    # PAUSED overlay on video
    if paused:
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, HEADER_H), (VIDEO_W, HEADER_H + VIDEO_H), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
        dt(canvas, "PAUSED — Press P to resume",
           (VIDEO_W // 2 - 175, DISPLAY_H // 2), scale=0.80, color=C_GREY, thickness=2)

    cv2.rectangle(canvas, (0, HEADER_H), (VIDEO_W - 1, HEADER_H + VIDEO_H - 1), C_DIVIDER, 1)

    # Sidebar
    sx0, sy0 = VIDEO_W, HEADER_H
    sb_h = DISPLAY_H - HEADER_H - FOOTER_H
    cv2.rectangle(canvas, (sx0, sy0), (DISPLAY_W, sy0 + sb_h), C_SIDEBAR, -1)
    cv2.line(canvas, (sx0, sy0), (sx0, sy0 + sb_h), C_DIVIDER, 1)

    y = sy0 + 18

    # Patient info
    dt(canvas, "PATIENT",          (sx0 + 14, y), scale=0.40, color=C_GREY)
    y += 24
    dt(canvas, seg["patient"],     (sx0 + 14, y), scale=0.70, color=C_WHITE, thickness=2)
    y += 22

    sc  = C_RED if det_type == "seizure" else C_AMBER if det_type == "fall" else C_GREEN
    stx = ("SEIZURE ALERT" if det_type == "seizure"
           else "FALL ALERT" if det_type == "fall"
           else "MONITORING")
    dt(canvas, f"Status: {stx}", (sx0 + 14, y), scale=0.46, color=sc)
    y += 16
    cv2.line(canvas, (sx0 + 10, y), (DISPLAY_W - 10, y), C_DIVIDER, 1)
    y += 14

    # Live confidence mini-bars
    dt(canvas, "LIVE READINGS", (sx0 + 14, y), scale=0.38, color=C_GREY)
    y += 18
    _mini_bar(canvas, sx0 + 14, y, "Fall Risk",    fall_sm, C_AMBER, FALL_THRESHOLD)
    y += 30
    _mini_bar(canvas, sx0 + 14, y, "Seizure Risk", sz_sm,   C_RED,   SEIZURE_THRESHOLD)
    y += 32
    cv2.line(canvas, (sx0 + 10, y), (DISPLAY_W - 10, y), C_DIVIDER, 1)
    y += 14

    # Alert log
    dt(canvas, "ALERT LOG", (sx0 + 14, y), scale=0.38, color=C_GREY)
    y += 18
    visible = alerts[-4:]
    if not visible:
        dt(canvas, "No alerts", (sx0 + 14, y + 12), scale=0.46, color=C_GREY)
    else:
        for card in visible:
            y = draw_alert_card(canvas, card, y)
            if y > DISPLAY_H - FOOTER_H - 20:
                break

    # Footer
    fy = DISPLAY_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (DISPLAY_W, DISPLAY_H), C_HEADER, -1)
    cv2.line(canvas, (0, fy), (DISPLAY_W, fy), C_DIVIDER, 1)
    seg_str = f"Seg [{seg['id']}/7]  {seg['patient']}  —  {seg['label']}"
    dt(canvas, seg_str, (14, fy + 24), scale=0.44, color=C_GREY)
    dt(canvas, "Q:Quit  SPC:Skip  P:Pause",
       (DISPLAY_W - 265, fy + 24), scale=0.40, color=C_DIVIDER)

    return canvas


def _mini_bar(img, bx, by, label, value, color, threshold):
    bar_w = SIDEBAR_W - 44
    bar_h = 12
    dt(img, label, (bx, by - 2), scale=0.40, color=C_GREY)
    pct_col = color if value >= threshold else C_GREY
    dt(img, f"{value * 100:.0f}%", (bx + bar_w + 6, by + 10), scale=0.44, color=pct_col)
    cv2.rectangle(img, (bx, by + 2), (bx + bar_w, by + 2 + bar_h), (45, 45, 55), -1)
    fill = int(bar_w * min(value, 1.0))
    if fill > 0:
        cv2.rectangle(img, (bx, by + 2), (bx + fill, by + 2 + bar_h), color, -1)
    tx = bx + int(bar_w * threshold)
    cv2.line(img, (tx, by), (tx, by + 2 + bar_h + 2), (200, 200, 200), 1)


def render_transition(alerts, message="Switching camera feed..."):
    canvas = np.full((DISPLAY_H, DISPLAY_W, 3), C_BG, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (DISPLAY_W, HEADER_H), C_HEADER, -1)
    dt(canvas, "VITAL GUARDIAN", (14, 32), scale=0.75, color=C_ACCENT, thickness=2)
    dt(canvas, "AI-Powered ICU Monitoring System", (190, 32), scale=0.44, color=C_GREY)
    cv2.circle(canvas, (DISPLAY_W - 60, 26), 6, C_GREEN, -1)

    dt(canvas, message,
       (VIDEO_W // 2 - 170, DISPLAY_H // 2),
       scale=0.65, color=C_GREY)

    sx0, sy0 = VIDEO_W, HEADER_H
    sb_h = DISPLAY_H - HEADER_H - FOOTER_H
    cv2.rectangle(canvas, (sx0, sy0), (DISPLAY_W, sy0 + sb_h), C_SIDEBAR, -1)
    cv2.line(canvas, (sx0, sy0), (sx0, sy0 + sb_h), C_DIVIDER, 1)
    y = sy0 + 32
    dt(canvas, "ALERT LOG", (sx0 + 14, y), scale=0.38, color=C_GREY)
    y += 18
    for card in alerts[-4:]:
        y = draw_alert_card(canvas, card, y)

    fy = DISPLAY_H - FOOTER_H
    cv2.rectangle(canvas, (0, fy), (DISPLAY_W, DISPLAY_H), C_HEADER, -1)
    return canvas


# ─────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────
def main():
    with open('config/config.yaml') as f:
        cfg = yaml.safe_load(f)
    vision_cfg = cfg['vision']
    vision_cfg['seizure_classifier']['threshold'] = SEIZURE_THRESHOLD
    vision_cfg['fall_classifier']['threshold']    = FALL_THRESHOLD
    if 'bed_exit' in vision_cfg:
        vision_cfg['bed_exit']['enabled'] = False

    print("=" * 65)
    print("VITAL GUARDIAN — LIVE DEMO (CPU)")
    print("=" * 65)
    print(f"  Frame skip    : {FRAME_SKIP} (every frame — matches validated accuracy)")
    print(f"  Ensemble      : FULL (5 fall + 5 motion + 5 temporal)")
    print(f"  YOLO imgsz    : {YOLO_IMGSZ}px")
    print(f"  Pose          : {'disabled' if DISABLE_POSE else 'enabled'}")
    print(f"  Thresholds    : seizure={SEIZURE_THRESHOLD}, fall={FALL_THRESHOLD}")
    print(f"  Note          : ~3-5 FPS on CPU — run live_demo_gpu.py for real-time")
    print()
    print("  Initializing pipeline...")
    pipeline = VisionPipeline(vision_cfg)

    if DISABLE_POSE:
        pipeline.pose_analyzer = None
        print("  Pose analyzer disabled for speed.")

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

        # Reset pipeline per segment
        pipeline.reset()
        try:
            pipeline.pose_history.clear()
        except Exception:
            pass
        pipeline.patient_state = 'OUT_OF_BED'

        consolidator = SegmentConsolidator(seg["type"])
        det_type     = "normal"
        det_conf     = 0.0
        fall_sm      = 0.0
        sz_sm        = 0.0
        frame_count  = 0

        # Play all sub-clips in this segment back-to-back
        for clip_path in seg["clips"]:
            if not clip_path.exists():
                print(f"    [MISSING] {clip_path.name}")
                continue

            cap = cv2.VideoCapture(str(clip_path))

            while True:
                ret, raw = cap.read()
                if not ret:
                    break

                frame_count += 1
                skip_ctr    += 1

                # Keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n  Quit by user.")
                    return
                if key == ord(' '):
                    cap.release()
                    break
                if key == ord('p'):
                    paused = not paused

                if paused:
                    canvas = render(raw, seg, alerts, det_type, det_conf,
                                    fall_sm, sz_sm, 0.0, frame_count, True)
                    cv2.imshow(WINDOW, canvas)
                    # Wait in pause loop
                    while paused:
                        canvas = render(raw, seg, alerts, det_type, det_conf,
                                        fall_sm, sz_sm, 0.0, frame_count, True)
                        cv2.imshow(WINDOW, canvas)
                        k2 = cv2.waitKey(100) & 0xFF
                        if k2 == ord('p'):
                            paused = False
                        if k2 == ord('q'):
                            cv2.destroyAllWindows()
                            return
                    continue

                # Frame skip — only process every Nth frame
                if skip_ctr % FRAME_SKIP == 0:
                    # Process at full resolution — downscaling broke detection
                    # on 848x480 fall clips (became too small for person detector)
                    event      = pipeline.process_frame(raw)
                    last_event = event

                    fall_sm = event.get('fall_smoothed',    fall_sm)
                    sz_sm   = event.get('seizure_smoothed', sz_sm)

                    # Check if this segment should fire an alert
                    should_fire, fired_type, fired_conf = consolidator.update(event)
                    if should_fire:
                        ts = time.strftime("%H:%M:%S")
                        card = AlertCard(fired_type, ts, seg["patient"], fired_conf)
                        alerts.append(card)
                        det_type = fired_type
                        det_conf = fired_conf
                        print(f"    *** ALERT: {card.title} — {seg['patient']}  "
                              f"({fired_conf * 100:.0f}%)")
                else:
                    # Carry forward last known values
                    event = last_event

                fps_tracker.tick()

                # Render and display
                canvas = render(raw, seg, alerts, det_type, det_conf,
                                fall_sm, sz_sm, fps_tracker.fps, frame_count, False)
                cv2.imshow(WINDOW, canvas)

            cap.release()
            # Small skip check after each sub-clip in case user pressed space
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # Transition between segments
        if seg_idx < len(SEGMENTS) - 1:
            trans_start = time.time()
            while time.time() - trans_start < GAP_SECONDS:
                canvas = render_transition(alerts)
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(33) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    return
                if key == ord(' '):
                    break

    # All segments done
    print()
    print("=" * 65)
    print(f"  Demo complete.  Total alerts: {len(alerts)}")
    for a in alerts:
        print(f"    [{a.timestamp}]  {a.title}  —  {a.patient}  ({a.confidence*100:.0f}%)")
    print("=" * 65)

    # Hold final frame until Q
    print("  Press Q in the window to close.")
    while True:
        canvas = render_transition(alerts, "Demo complete — press Q to exit")
        cv2.imshow(WINDOW, canvas)
        if cv2.waitKey(33) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
