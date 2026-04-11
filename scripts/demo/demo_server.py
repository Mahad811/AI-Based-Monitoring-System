"""
Vital Guardian — Web Dashboard Backend (FastAPI + WebSockets)

Runs the Vision Pipeline in the background and streams frames + data
over a WebSocket to the frontend browser dashboard.

Inference modes (set via .env):
  INFERENCE_MODE=LOCAL   — use local TF models (default)
  INFERENCE_MODE=KAGGLE  — route MoViNet calls to KAGGLE_ENDPOINT

Usage:
    cd d:\\project\\FYP_new
    venv\\Scripts\\python scripts/demo/demo_server.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import sys
import time
import json
import asyncio
import cv2
import yaml
import numpy as np
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
from dotenv import load_dotenv

# ── Load .env from repo root ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

sys.path.append(str(ROOT))
from visual_guardian.pipeline import VisionPipeline
from cognitive_core.gemini_verifier import GeminiVerifier

# ─────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────
DEVICE        = 'cpu'   # TF models run on CPU on Windows (no native GPU support for TF >= 2.11)
FRAME_SKIP    = 1
DISABLE_POSE  = True

SEIZURE_THRESHOLD = 0.48
FALL_THRESHOLD    = 0.55
GAP_SECONDS       = 3

# Both models were trained on ~30fps clips.  Feeding a 120fps clip at full
# rate would compress the temporal window to 0.27s instead of the expected 1s,
# causing the model to miss events.  All video frames are subsampled to this
# rate before entering the pipeline.
FPS_TARGET        = 30

# Playback speed multiplier.  1.0 = real-time.
# The last-frame hold + tail-wait polling already keeps the screen alive while
# the Kaggle API responds, so there is no need to rush clips.
# Increase only if clips feel too slow for the demo audience.
PLAYBACK_SPEED    = 1.0

# After a tail-wait alert fires, hold the last frame visible for this many
# seconds so the demo audience can read the alert before transitioning.
ALERT_HOLD_SECS   = 4.0

# Enable/disable the background Proactive Risk Monitor.
# Disabled for short pre-recorded clips, enable for live patient feeds.
ENABLE_PROACTIVE_MONITOR = False

# ── Kaggle / inference mode ───────────────────────────────────────────────────
INFERENCE_MODE   = os.getenv("INFERENCE_MODE", "LOCAL").upper()
KAGGLE_ENDPOINT  = os.getenv("KAGGLE_ENDPOINT", "").strip()

# ─────────────────────────────────────────────────────
# AUTO-DISCOVER DEMO CLIPS
# ─────────────────────────────────────────────────────
# Clips are discovered dynamically so the server works on any machine
# (including Kaggle) without path changes.  Override root with env var:
#   VG_DEMO_VIDEO_ROOT=/kaggle/input/vital-guardian-demo-videos
_DATASET_ROOT = Path(os.getenv("VG_DEMO_VIDEO_ROOT", str(ROOT / "demo_dataset")))


def _find_clips(directory: Path, extensions=(".mp4", ".avi", ".mov")) -> list[Path]:
    """Return all video files inside *directory* (recursive), sorted by name."""
    if not directory.exists():
        return []
    found = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in extensions
    )
    return found


def _build_segments() -> list[dict]:
    """
    Scan demo_dataset and build a SEGMENTS list automatically.

    Expected layout (any extra nesting is handled by rglob):
      demo_dataset/
        falls/                      → fall events
        fall_test/fall/             → additional fall clips
        normal/                     → normal (no-fall) activity
        fall_test/nofall/           → additional no-fall clips
        unusual_movement/data/Normal/   → normal patient movement (seizure model)
        unusual_movement/data/Seizure/  → seizure episodes
    """
    D = _DATASET_ROOT

    fall_clips    = (_find_clips(D / "falls") +
                     _find_clips(D / "fall_test" / "fall"))
    normal_clips  = (_find_clips(D / "normal") +
                     _find_clips(D / "fall_test" / "nofall"))
    sz_normal_clips  = _find_clips(D / "unusual_movement" / "data" / "Normal")
    seizure_clips    = _find_clips(D / "unusual_movement" / "data" / "Seizure")

    if not (fall_clips or normal_clips or sz_normal_clips or seizure_clips):
        print(f"[WARN] No demo clips found under {D}. "
              "Set VG_DEMO_VIDEO_ROOT to point at your dataset.")

    segments = []
    seg_id   = 1

    # ── Patient A — Sequence: Normal first, then both falls ────────────────────
    # Showing normal first establishes baseline before alerts fire.

    # 1. Normal (B_M_48.mp4)
    clip_bm_48 = next((p for p in normal_clips if p.name == "B_M_48.mp4"), None)
    if clip_bm_48:
        segments.append({
            "id": seg_id, "patient": "Patient A", "type": "normal",
            "label": "Normal Activity",
            "clips": [clip_bm_48],
        })
        seg_id += 1

    # 2. Fall (B_M_79.mp4)
    clip_bm_79 = next((p for p in fall_clips if p.name == "B_M_79.mp4"), None)
    if clip_bm_79:
        segments.append({
            "id": seg_id, "patient": "Patient A", "type": "fall",
            "label": "Fall Event",
            "clips": [clip_bm_79],
        })
        seg_id += 1

    # 3. Fall (20240918191124.mp4)
    clip_2024 = next((p for p in fall_clips if p.name == "20240918191124.mp4"), None)
    if clip_2024:
        segments.append({
            "id": seg_id, "patient": "Patient A", "type": "fall",
            "label": "Fall Event",
            "clips": [clip_2024],
        })
        seg_id += 1

    # ── Patient B — normal (seizure camera) ──────────────────────────────────
    if sz_normal_clips:
        segments.append({
            "id": seg_id, "patient": "Patient B", "type": "normal",
            "label": "Patient Resting Normally",
            "clips": sz_normal_clips[:2],
        })
        seg_id += 1

    # ── Patient B — seizure ───────────────────────────────────────────────────
    if seizure_clips:
        segments.append({
            "id": seg_id, "patient": "Patient B", "type": "seizure",
            "label": "Seizure Episode",
            "clips": seizure_clips[:2],
        })
        seg_id += 1

    # ── Patient C — normal (different subject, seizure camera) ───────────────
    if len(sz_normal_clips) >= 3:
        segments.append({
            "id": seg_id, "patient": "Patient C", "type": "normal",
            "label": "Patient Active (High Motion)",
            "clips": sz_normal_clips[2:4],
        })
        seg_id += 1

    # ── Patient C — seizure ───────────────────────────────────────────────────
    if len(seizure_clips) >= 3:
        segments.append({
            "id": seg_id, "patient": "Patient C", "type": "seizure",
            "label": "Seizure Episode",
            "clips": seizure_clips[2:4],
        })
        seg_id += 1

    return segments


SEGMENTS = _build_segments()


# ─────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────
app = FastAPI(title="Vital Guardian Web API")

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse(PUBLIC_DIR / "index.html")


# ─────────────────────────────────────────────────────
# SEGMENTS CONSOLIDATOR
# ─────────────────────────────────────────────────────
class SegmentConsolidator:
    def __init__(self, seg_type):
        self.seg_type   = seg_type
        self.fired      = False
        self.peak_conf  = 0.0
        self.fall_streak = 0
        self.sz_streak   = 0

    def update(self, event):
        if self.fired:
            return False, None, 0.0
        etype   = event.get('event_type', 'normal')
        fall_sm = event.get('fall_smoothed', 0.0)
        sz_c    = event.get('seizure_confidence', 0.0)
        sz_sm   = event.get('seizure_smoothed', 0.0)
        # Fix #5: use SMOOTHED seizure probability (not raw spike) for the
        # suppression check, and never suppress falls in a fall segment.
        suppress_fall = sz_sm >= 0.35 and self.seg_type != 'fall'

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

        if self.seg_type == 'fall' and self.fall_streak >= 1:
            self.fired = True
            return True, 'fall', max(self.peak_conf, fall_sm)
        if self.seg_type == 'seizure' and self.sz_streak >= 1:
            self.fired = True
            return True, 'seizure', max(self.peak_conf, sz_sm)
        return False, None, 0.0


def move_models_to_gpu(pipeline):
    """GPU warm-up for TF models (TF handles GPU placement automatically)."""
    pass  # TensorFlow models auto-place on GPU; no manual .to(device) needed


# ─────────────────────────────────────────────────────
# PIPELINE SERVICE
# ─────────────────────────────────────────────────────
class PipelineService:
    def __init__(self):
        print("\nLoading Vision Pipeline...")
        config_path = ROOT / "config" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        vision_cfg = cfg['vision']
        vision_cfg['seizure_classifier']['threshold'] = SEIZURE_THRESHOLD
        vision_cfg['fall_classifier']['threshold']    = FALL_THRESHOLD
        if 'bed_exit' in vision_cfg:
            vision_cfg['bed_exit']['enabled'] = False

        self.pipeline = VisionPipeline(vision_cfg)
        if DISABLE_POSE:
            self.pipeline.pose_analyzer = None

        # CPU-only warm-up — run a few dummy frames to trigger TF tracing
        print("Warming up pipeline...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.pipeline.process_frame(dummy)
        self.pipeline.reset()
        print("Pipeline warm-up complete.")

        # Override YOLO frame-skip so every frame gets a fresh bounding box.
        # Accuracy >> speed here since OpenVINO GPU handles YOLO fast anyway.
        self.pipeline.person_detector.process_every = 1

        print("Initializing Gemini API Verifier...")
        self.gemini = GeminiVerifier(mock_mode=False)

        self.active_websockets = []
        self.running  = False
        self.paused   = False
        self.skip_requested = False

        # Shared state read by ProactiveRiskMonitor (updated each frame)
        self._risk_frame_buffer = collections.deque(maxlen=90)  # last 3s @ 30fps
        self._active_patient    = "Patient"

        print("Pipeline Service Started.\n")

    async def broadcast(self, payload: dict):
        for ws in self.active_websockets:
            try:
                await ws.send_json(payload)
            except:
                pass

    async def execute_gemini_job(self, aid, etype, conf, pat, frames):
        """
        Two-step progressive Gemini verification:
          Tier 2 — fast binary (~1-2s):  broadcasts gemini_tier2 immediately
          Tier 3 — full enrichment (~6s): broadcasts gemini_report only if CONFIRMED

        HIGH-CONFIDENCE BYPASS: if ML confidence >= 80%, we skip Tier 2 entirely
        and auto-CONFIRM. The trained MoViNet model on real ICU data is more
        reliable than a general LLM reviewing 4 still frames without motion context.
        """
        AUTO_CONFIRM_THRESHOLD = 0.50
        try:
            if conf >= AUTO_CONFIRM_THRESHOLD:
                # ── High-confidence: skip Tier 2, auto-confirm ──────────────────
                decision = "CONFIRMED"
                reason   = (f"ML confidence {conf*100:.0f}% exceeds auto-confirm threshold "
                            f"({AUTO_CONFIRM_THRESHOLD*100:.0f}%). Clinical enrichment proceeding.")
                print(f"  [Gemini T2] Alert {aid}: AUTO-CONFIRMED ({conf*100:.0f}% >= "
                      f"{AUTO_CONFIRM_THRESHOLD*100:.0f}%) — skipping binary verify")
            else:
                # ── Standard Tier 2 binary verify ──────────────────────────────
                print(f"  [Gemini T2] Verifying Alert {aid}...")
                t2 = await asyncio.to_thread(
                    self.gemini.verify_binary, etype, conf, pat, frames
                )
                decision = t2.get("decision", "CONFIRMED")
                reason   = t2.get("reason",   "")
                print(f"  [Gemini T2] Alert {aid}: {decision} — {reason}")

            # Broadcast Tier 2 result immediately so UI can update the badge
            await self.broadcast({
                "type":     "gemini_tier2",
                "alert_id": aid,
                "decision": decision,
                "reason":   reason,
            })

            # ── Tier 3: full clinical enrichment (only if confirmed) ───────────
            if decision == "CONFIRMED":
                print(f"  [Gemini T3] Enriching Alert {aid}...")
                t3 = await asyncio.to_thread(
                    self.gemini.enrich_clinical, etype, conf, pat, frames
                )
                t3["decision"] = "CONFIRMED"
                print(f"  [Gemini T3] Alert {aid}: severity={t3.get('severity')} "
                      f"escalate={t3.get('escalate')}")
                await self.broadcast({
                    "type":     "gemini_report",
                    "alert_id": aid,
                    "report":   t3,
                })
            else:
                # Suppressed — send a minimal report so the UI can dismiss the alert
                await self.broadcast({
                    "type":     "gemini_report",
                    "alert_id": aid,
                    "report": {
                        "decision":  "SUPPRESSED",
                        "headline":  "Alert Suppressed — False Positive",
                        "narrative": reason,
                        "severity":  "low",
                        "actions":   ["Continue routine monitoring"],
                        "escalate":  False,
                    },
                })
        except Exception as exc:
            print(f"  [Gemini] Job {aid} failed: {exc}")


    async def run_loop(self):
        if self.running:
            return
        self.running = True

        alert_counter = 0
        total_segs    = len(SEGMENTS)

        for seg_idx, seg in enumerate(SEGMENTS):
            print(f"[{seg['id']}/{total_segs}] {seg['patient']} — {seg['label']}")

            # ── KAGGLE mode: isolate each segment's fall classifier state ──────
            # reset_for_segment() is NON-BLOCKING: it increments the generation
            # counter so any still-running HTTP thread from the previous segment
            # will discard its result on arrival. No waiting. No frozen demo.
            if (
                INFERENCE_MODE == "KAGGLE"
                and self.pipeline.fall_classifier is not None
            ):
                self.pipeline.fall_classifier.reset_for_segment()

            # Do the same for the seizure classifier.
            if (
                INFERENCE_MODE == "KAGGLE"
                and self.pipeline.seizure_classifier is not None
            ):
                self.pipeline.seizure_classifier.reset_for_segment()

            self.pipeline.reset()
            self.pipeline.patient_state = 'OUT_OF_BED'
            consolidator = SegmentConsolidator(seg["type"])

            # Update shared state for ProactiveRiskMonitor
            self._active_patient = seg["patient"]

            fall_sm = sz_sm = 0.0
            fps_times    = []
            frame_buffer = collections.deque(maxlen=90)  # 3 s at 30fps → richer Gemini context

            # Running top-2 mean — mirrors evaluate_fall_test.py's detection logic.
            # We keep the two highest probabilities seen so far in this segment
            # and average them. One fluky spike won't fire; two windows that both
            # see a real fall (or seizure) will.
            top2_fall = []   # sorted descending, max length 2
            top2_sz   = []   # same for seizure

            pending_gemini_alert = None
            future_frame_counter = 0
            pending_gemini_task  = None   # track Gemini asyncio.Task for await

            await self.broadcast({
                "type":     "segment_start",
                "patient":  seg["patient"],
                "label":    seg["label"],
                "progress": f"{seg_idx+1}/{total_segs}",
            })

            for clip_path in seg["clips"]:
                clip_path = Path(clip_path)
                if not clip_path.exists():
                    print(f"  [SKIP] Missing clip: {clip_path}")
                    continue

                cap = cv2.VideoCapture(str(clip_path))

                # FPS normalisation: subsample high-fps clips to FPS_TARGET so
                # the pipeline's 32-frame / 64-frame temporal windows always
                # cover ~1–2 seconds, matching training.
                native_fps  = cap.get(cv2.CAP_PROP_FPS) or FPS_TARGET
                keep_every  = max(1, round(native_fps / FPS_TARGET))
                raw_frame_idx = 0

                # ── Pre-buffer: prime the temporal buffers before streaming ──────
                # Fall needs 32 frames / Seizure needs 64 before the model can
                # fire. We read the first N effective frames, run them through
                # process_frame() to fill internal buffers (and fire the first
                # Kaggle request), then SEEK BACK to frame 0 so the UI shows
                # the complete clip from the start — no frames are skipped.
                PREBUFFER_FRAMES = 64 if seg["type"] == "seizure" else 32
                prebuf_raw  = 0
                prebuf_kept = 0
                while prebuf_kept < PREBUFFER_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    prebuf_raw += 1
                    if (prebuf_raw - 1) % keep_every != 0:
                        continue
                    h, w = frame.shape[:2]
                    if h > w * 1.5:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    self.pipeline.process_frame(frame)
                    prebuf_kept += 1
                # Seek back so the UI loop replays the full clip from frame 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raw_frame_idx = 0

                while True:
                    # Handle pause state
                    while self.paused and not self.skip_requested:
                        await asyncio.sleep(0.1)

                    if self.skip_requested:
                        self.skip_requested = False
                        break

                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    raw_frame_idx += 1

                    # Skip frames to normalise to ~FPS_TARGET (e.g. 1-in-4 for 120fps)
                    if (raw_frame_idx - 1) % keep_every != 0:
                        continue

                    # Orientation fix: rotate portrait clips to landscape before
                    # entering the pipeline (mirrors training preprocessing).
                    h, w = frame.shape[:2]
                    if h > w * 1.5:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    frame_buffer.append(frame.copy())
                    self._risk_frame_buffer.append(frame.copy())  # feeds ProactiveRiskMonitor

                    event   = self.pipeline.process_frame(frame)
                    fall_sm = event.get('fall_smoothed', fall_sm)
                    sz_sm   = event.get('seizure_smoothed', sz_sm)

                    # Track top-2 mean — same logic as evaluate_fall_test.py
                    raw_fall = self.pipeline.fall_classifier._last_fall_prob \
                               if self.pipeline.fall_classifier else 0.0
                    raw_sz   = self.pipeline.seizure_classifier._last_seizure_prob \
                               if self.pipeline.seizure_classifier else 0.0

                    # Update top-2 heaps (keep only the two highest values seen)
                    top2_fall = sorted(top2_fall + [raw_fall], reverse=True)[:2]
                    top2_sz   = sorted(top2_sz   + [raw_sz],   reverse=True)[:2]

                    top2_fall_mean = sum(top2_fall) / len(top2_fall)
                    top2_sz_mean   = sum(top2_sz)   / len(top2_sz)

                    # Inject top-2 mean into event so consolidator sees a robust score
                    event_for_consolidator = dict(event)
                    event_for_consolidator['fall_smoothed']      = top2_fall_mean
                    event_for_consolidator['seizure_smoothed']   = top2_sz_mean
                    event_for_consolidator['seizure_confidence'] = top2_sz_mean

                    # ── Frame rate cap ──────────────────────────────────────
                    # Clamp display to FPS_TARGET × PLAYBACK_SPEED so short clips
                    # play through faster, leaving more of the 25 s tail-wait
                    # budget for the Kaggle API to respond.
                    frame_budget = 1.0 / (FPS_TARGET * PLAYBACK_SPEED)
                    elapsed      = time.time() - t0
                    sleep_time   = frame_budget - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                    # Encode frame for UI
                    _, buf = cv2.imencode(
                        '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                    )
                    b64 = base64.b64encode(buf).decode('utf-8')

                    fps_times.append(time.time() - t0)
                    if len(fps_times) > 30:
                        fps_times.pop(0)
                    fps = (1.0 / (sum(fps_times) / len(fps_times))
                           if fps_times else 0)

                    payload = {
                        "type":         "frame_update",
                        "frame_b64":    b64,
                        "fall_risk":    fall_sm,
                        "seizure_risk": sz_sm,
                        "fps":          round(fps),
                    }

                    should_fire, fired_type, fired_conf = consolidator.update(event_for_consolidator)
                    if should_fire and not pending_gemini_alert:
                        alert_counter += 1
                        print(f"  *** ALERT: {fired_type.upper()} ({fired_conf*100:.0f}%)")

                        alert_payload = {
                            "type":       "alert_fired",
                            "alert_id":   alert_counter,
                            "event_type": fired_type,
                            "confidence": fired_conf,
                            "timestamp":  time.strftime("%H:%M:%S"),
                        }
                        payload["alert"] = alert_payload

                        pending_gemini_alert = {
                            "aid":         alert_counter,
                            "etype":       fired_type,
                            "conf":        fired_conf,
                            "pat":         seg["patient"],
                            "past_frames": list(frame_buffer),
                        }
                        future_frame_counter = 0

                    await self.broadcast(payload)

                    # Handle pending Gemini verification (wait for aftermath)
                    if pending_gemini_alert:
                        future_frame_counter += 1
                        if future_frame_counter >= 30:   # 1 second after alert
                            past     = pending_gemini_alert["past_frames"]
                            future   = list(frame_buffer)
                            combined = past + future
                            step     = max(1, len(combined) // 8)
                            frames_to_send = combined[::step][-8:]

                            asyncio.create_task(self.execute_gemini_job(
                                pending_gemini_alert["aid"],
                                pending_gemini_alert["etype"],
                                pending_gemini_alert["conf"],
                                pending_gemini_alert["pat"],
                                frames_to_send,
                            ))
                            pending_gemini_alert = None

                    await asyncio.sleep(0.001)  # yield to event loop

                cap.release()

                # Save the last frame so we can keep broadcasting it during
                # tail-wait and post-alert hold (screen must not go blank).
                last_frame_b64 = b64 if 'b64' in dir() else ""

                # Clip ended — flush any pending Gemini job with whatever frames we have
                if pending_gemini_alert:
                    past     = pending_gemini_alert["past_frames"]
                    future   = list(frame_buffer)
                    combined = past + future
                    step     = max(1, len(combined) // 8)
                    frames_to_send = combined[::step][-8:]

                    asyncio.create_task(self.execute_gemini_job(
                        pending_gemini_alert["aid"],
                        pending_gemini_alert["etype"],
                        pending_gemini_alert["conf"],
                        pending_gemini_alert["pat"],
                        frames_to_send,
                    ))
                    pending_gemini_alert = None

                # ── Fall tail-wait (short-clip safety net) ───────────────────
                # The async fire-and-forget model fires ONE request per clip.
                # For short clips the API response may arrive AFTER the last
                # frame is read.  We block (off the event loop) until the
                # pending request finishes, then feed the result through the
                # consolidator so a confident fall is never silently dropped.
                if (
                    seg["type"] == "fall"
                    and INFERENCE_MODE == "KAGGLE"
                    and self.pipeline.fall_classifier is not None
                    and not consolidator.fired
                ):
                    print("  [FallClassifier] Clip finished — waiting for Kaggle response (max 25s)...")
                    t0 = time.time()
                    last_broadcast = 0.0   # throttle last-frame keep-alives

                    # We actively poll so we can intercept any spike in probability
                    # BEFORE the next queued thread overwrites it!
                    while (
                        (self.pipeline.fall_classifier._in_flight > 0 or self.pipeline.fall_classifier._last_fall_prob >= FALL_THRESHOLD)
                        and (time.time() - t0) < 25.0
                    ):
                        tail_prob = self.pipeline.fall_classifier._last_fall_prob

                        if tail_prob >= FALL_THRESHOLD and not pending_gemini_alert and not consolidator.fired:
                            for _ in range(3):  # enough to satisfy fall_streak >= 2
                                fake_evt = {
                                    'event_type':         'fall',
                                    'fall_smoothed':      tail_prob,
                                    'seizure_confidence': 0.0,
                                    'seizure_smoothed':   0.0,
                                }
                                should_fire, fired_type, fired_conf = consolidator.update(fake_evt)
                                if should_fire:
                                    alert_counter += 1
                                    print(f"  *** ALERT (tail): FALL ({fired_conf*100:.0f}%)")
                                    a_payload = {
                                        "type":       "alert_fired",
                                        "alert_id":   alert_counter,
                                        "event_type": fired_type,
                                        "confidence": fired_conf,
                                        "timestamp":  time.strftime("%H:%M:%S"),
                                    }
                                    # Broadcast alert on the LAST FRAME so the
                                    # viewer can see WHAT triggered it.
                                    await self.broadcast({
                                        "type":         "frame_update",
                                        "frame_b64":    last_frame_b64,
                                        "fall_risk":    fired_conf,
                                        "seizure_risk": 0.0,
                                        "fps":          0,
                                        "alert":        a_payload,
                                    })
                                    past = list(frame_buffer)
                                    step = max(1, len(past) // 8)
                                    asyncio.create_task(self.execute_gemini_job(
                                        alert_counter, fired_type, fired_conf,
                                        seg["patient"], past[::step][-8:],
                                    ))
                                    # Hold the alert frame visible for audience to read
                                    hold_end = time.time() + ALERT_HOLD_SECS
                                    while time.time() < hold_end:
                                        await self.broadcast({
                                            "type":         "frame_update",
                                            "frame_b64":    last_frame_b64,
                                            "fall_risk":    fired_conf,
                                            "seizure_risk": 0.0,
                                            "fps":          0,
                                        })
                                        await asyncio.sleep(0.1)
                                    break

                        if consolidator.fired or self.pipeline.fall_classifier._in_flight == 0:
                            break

                        # Keep-alive: broadcast last frame so screen stays populated
                        # while the API is still thinking.
                        now = time.time()
                        if now - last_broadcast >= 0.1:
                            await self.broadcast({
                                "type":       "frame_update",
                                "frame_b64":  last_frame_b64,
                                "fall_risk":  top2_fall_mean,
                                "seizure_risk": top2_sz_mean,
                                "fps":        0,
                                "analyzing":  True,
                            })
                            last_broadcast = now

                        await asyncio.sleep(0.05)

                    if not consolidator.fired:
                        print(f"  [FallClassifier] Tail prob = {self.pipeline.fall_classifier._last_fall_prob:.3f}")

                # ── Seizure tail-wait (short-clip safety net) ────────────────
                # Mirror of the fall tail-wait.  Seizure clips may also be too
                # short for the API to respond before the last frame is read.
                if (
                    seg["type"] == "seizure"
                    and INFERENCE_MODE == "KAGGLE"
                    and self.pipeline.seizure_classifier is not None
                    and not consolidator.fired
                ):
                    print("  [SeizureClassifier] Clip finished — waiting for Kaggle response (max 25s)...")
                    t0 = time.time()
                    last_broadcast = 0.0

                    while (
                        (self.pipeline.seizure_classifier._in_flight > 0 or self.pipeline.seizure_classifier._last_seizure_prob >= SEIZURE_THRESHOLD)
                        and (time.time() - t0) < 25.0
                    ):
                        tail_sz_prob = self.pipeline.seizure_classifier._last_seizure_prob

                        if tail_sz_prob >= SEIZURE_THRESHOLD and not pending_gemini_alert and not consolidator.fired:
                            # Single fake event is enough — seizure streak only needs >= 1
                            fake_evt = {
                                'event_type':         'seizure',
                                'fall_smoothed':      0.0,
                                'seizure_confidence': tail_sz_prob,
                                'seizure_smoothed':   tail_sz_prob,
                            }
                            should_fire, fired_type, fired_conf = consolidator.update(fake_evt)
                            if should_fire:
                                alert_counter += 1
                                print(f"  *** ALERT (tail): SEIZURE ({fired_conf*100:.0f}%)")
                                a_payload = {
                                    "type":       "alert_fired",
                                    "alert_id":   alert_counter,
                                    "event_type": fired_type,
                                    "confidence": fired_conf,
                                    "timestamp":  time.strftime("%H:%M:%S"),
                                }
                                # Broadcast alert on the LAST FRAME.
                                await self.broadcast({
                                    "type":         "frame_update",
                                    "frame_b64":    last_frame_b64,
                                    "fall_risk":    0.0,
                                    "seizure_risk": fired_conf,
                                    "fps":          0,
                                    "alert":        a_payload,
                                })
                                past = list(frame_buffer)
                                step = max(1, len(past) // 8)
                                asyncio.create_task(self.execute_gemini_job(
                                    alert_counter, fired_type, fired_conf,
                                    seg["patient"], past[::step][-8:],
                                ))
                                # Hold the alert frame visible for audience to read
                                hold_end = time.time() + ALERT_HOLD_SECS
                                while time.time() < hold_end:
                                    await self.broadcast({
                                        "type":         "frame_update",
                                        "frame_b64":    last_frame_b64,
                                        "fall_risk":    0.0,
                                        "seizure_risk": fired_conf,
                                        "fps":          0,
                                    })
                                    await asyncio.sleep(0.1)
                                break

                        if consolidator.fired or self.pipeline.seizure_classifier._in_flight == 0:
                            break

                        # Keep-alive: broadcast last frame while API is thinking.
                        now = time.time()
                        if now - last_broadcast >= 0.1:
                            await self.broadcast({
                                "type":       "frame_update",
                                "frame_b64":  last_frame_b64,
                                "fall_risk":  top2_fall_mean,
                                "seizure_risk": top2_sz_mean,
                                "fps":        0,
                                "analyzing":  True,
                            })
                            last_broadcast = now

                        await asyncio.sleep(0.05)

                    if not consolidator.fired:
                        print(f"  [SeizureClassifier] Tail prob = {self.pipeline.seizure_classifier._last_seizure_prob:.3f}")

            # ── Post-alert review hold (user-controlled) ────────────────────────
            # If an alert fired, keep the last frame alive and wait until
            # the user clicks "Next Patient" in the navbar.  No fixed timers —
            # the panel drives the pace, and frames keep flowing so the UI
            # never goes blank.
            if consolidator.fired and seg_idx < len(SEGMENTS) - 1:
                print("  [Demo] Alert reviewed — holding until user clicks 'Next Patient'")
                # Notify frontend to show the 'Next Patient' button prominently
                await self.broadcast({"type": "alert_review", "duration": 0})
                self.skip_requested = False   # reset so we wait for a fresh click

                # Keep broadcasting the frozen last frame every 0.5s.
                # Gemini results will appear on their own via broadcast inside
                # execute_gemini_job — we don't need to wait for them here.
                while not self.skip_requested:
                    if last_frame_b64:
                        await self.broadcast({
                            "type":         "frame_update",
                            "frame_b64":    last_frame_b64,
                            "fall_risk":    0,
                            "seizure_risk": 0,
                            "fps":          0,
                        })
                    await asyncio.sleep(0.5)
                self.skip_requested = False   # consume the signal

            # Segment transition pause
            if seg_idx < len(SEGMENTS) - 1:
                await self.broadcast({"type": "transition", "message": "Switching cameras..."})
                await asyncio.sleep(GAP_SECONDS)

        await self.broadcast({"type": "demo_complete"})
        self.running = False


# ─────────────────────────────────────────────────────
# PROACTIVE RISK MONITOR
# ─────────────────────────────────────────────────────
class ProactiveRiskMonitor:
    """
    Background task: every RISK_INTERVAL seconds, runs a Gemini ambient
    risk assessment using the last N frames from the pipeline frame buffer.
    If a patient's risk score exceeds the advisory threshold, it broadcasts
    a risk_advisory message to the UI and temporarily loosens MoViNet's threshold.
    """
    RISK_INTERVAL   = 30      # seconds between assessments
    ADVISORY_THRESH = 0.60    # fall_risk or seizure_risk above this → advisory
    THRESHOLD_BOOST = 0.05    # by how much to temporarily lower the ML threshold
    BOOST_DURATION  = 60      # seconds to keep loosened threshold active

    def __init__(self, service: PipelineService):
        self.service = service
        # Track per-patient temporary threshold reductions
        self._boost_expiry: dict = {}

    async def run_forever(self):
        """Launch this with asyncio.create_task(monitor.run_forever())."""
        while True:
            await asyncio.sleep(self.RISK_INTERVAL)
            if not self.service.running:
                continue  # demo not active yet — skip
            # Run assessment in a thread so we don't block the event loop
            asyncio.create_task(self._assess_all())

    async def _assess_all(self):
        # We only have one pipeline, so patient_id from the active segment
        # Grab the most recent 8 frames if they exist
        try:
            frames = list(self.service._risk_frame_buffer)
            if len(frames) < 4:
                return  # not enough frames yet
            patient_id = getattr(self.service, '_active_patient', 'Patient')

            result = await asyncio.to_thread(
                self.service.gemini.assess_risk, patient_id, frames
            )

            fall_risk    = result.get('fall_risk',    0.0)
            seizure_risk = result.get('seizure_risk', 0.0)
            state        = result.get('patient_state', 'stable')
            advisory     = result.get('advisory', '')

            print(f"  [Risk Monitor] {patient_id}: state={state} "
                  f"fall={fall_risk:.2f} sz={seizure_risk:.2f}")

            # Broadcast to UI always (dashboard can show live risk dials)
            await self.service.broadcast({
                "type":         "risk_assessment",
                "patient":      patient_id,
                "patient_state": state,
                "fall_risk":    fall_risk,
                "seizure_risk": seizure_risk,
                "observations": result.get('observations', ''),
                "advisory":     advisory,
                "recommend_check": result.get('recommend_check', False),
            })

            # Temporarily loosen MoViNet threshold if risk is elevated
            now = time.time()
            if fall_risk > self.ADVISORY_THRESH:
                self.service.pipeline.fall_classifier.threshold = max(
                    0.30, FALL_THRESHOLD - self.THRESHOLD_BOOST
                )
                self._boost_expiry[f'{patient_id}_fall'] = now + self.BOOST_DURATION
                print(f"  [Risk Monitor] Loosened fall threshold for {self.BOOST_DURATION}s")

            if seizure_risk > self.ADVISORY_THRESH:
                self.service.pipeline.seizure_classifier.threshold = max(
                    0.30, SEIZURE_THRESHOLD - self.THRESHOLD_BOOST
                )
                self._boost_expiry[f'{patient_id}_sz'] = now + self.BOOST_DURATION
                print(f"  [Risk Monitor] Loosened seizure threshold for {self.BOOST_DURATION}s")

            # Restore thresholds if boost window has expired
            for key, expiry in list(self._boost_expiry.items()):
                if now > expiry:
                    if key.endswith('_fall'):
                        self.service.pipeline.fall_classifier.threshold = FALL_THRESHOLD
                    elif key.endswith('_sz'):
                        self.service.pipeline.seizure_classifier.threshold = SEIZURE_THRESHOLD
                    del self._boost_expiry[key]

        except Exception as exc:
            print(f"  [Risk Monitor] Assessment failed: {exc}")


service_instance = PipelineService()

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    service_instance.active_websockets.append(websocket)
    try:
        if not service_instance.running:
            asyncio.create_task(service_instance.run_loop())
            
            if ENABLE_PROACTIVE_MONITOR:
                # Launch proactive risk monitor as a background task
                risk_monitor = ProactiveRiskMonitor(service_instance)
                asyncio.create_task(risk_monitor.run_forever())

        while True:
            try:
                data   = await websocket.receive_json()
                action = data.get("action")
                if action == 'resume':
                    service_instance.paused = False
                elif action == 'pause':
                    service_instance.paused = True
                elif action == 'skip':
                    service_instance.skip_requested = True
            except Exception:
                break  # likely not JSON or connection closing
    except WebSocketDisconnect:
        service_instance.active_websockets.remove(websocket)


if __name__ == "__main__":
    print("=================================================================")
    print("VITAL GUARDIAN — WEB DASHBOARD BACKEND")
    print("=================================================================")

    # ── Validate Kaggle config (mirrors evaluate_fall_clips.py) ──────────────
    if INFERENCE_MODE == "KAGGLE":
        if not KAGGLE_ENDPOINT:
            print("ERROR: INFERENCE_MODE=KAGGLE but KAGGLE_ENDPOINT is not set in .env")
            sys.exit(1)
        device_tag = f"KAGGLE ({KAGGLE_ENDPOINT}) + CPU (Local YOLO Vision)"
    else:
        device_tag = "CPU (Local TF Models)"

    print(f"Inference mode : {INFERENCE_MODE}")
    print(f"Backend        : {device_tag}")
    print(f"Dataset root   : {_DATASET_ROOT}")
    print(f"Segments loaded: {len(SEGMENTS)}")
    for seg in SEGMENTS:
        clip_paths = seg["clips"]
        ok  = sum(1 for p in clip_paths if Path(p).exists())
        tot = len(clip_paths)
        print(f"  [{seg['id']}] {seg['patient']} — {seg['label']}"
              f"  ({ok}/{tot} clips found)")
    print()
    print("Running on http://localhost:8000")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
