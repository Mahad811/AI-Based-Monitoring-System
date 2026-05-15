import os
import time
import cv2
import base64
import requests
import threading
import numpy as np
from collections import deque
from pathlib import Path
from .movinet_loader import load_movinet

SEIZURE_CLIP_FRAMES = 32
SEIZURE_BUFFER      = 64   # 64 raw → stride-2 → 32 model frames
MAX_CONCURRENT      = 1    # keep at 1 — free ngrok tunnels drop SSL under parallel load
MAX_RETRIES         = 3    # retry failed requests with exponential backoff
RETRY_BASE_S        = 0.5  # first retry after 0.5s, then 1s, then 2s


class SeizureClassifier:
    def __init__(self, model_path, window_frames=64, target_size=224, device='auto'):
        self.target_size  = target_size
        self.frame_buffer = deque(maxlen=SEIZURE_BUFFER)

        self.mode       = os.getenv("INFERENCE_MODE", "LOCAL").upper()
        self.kaggle_url = os.getenv("KAGGLE_ENDPOINT", "")

        # ── Async inference state (KAGGLE mode only) ─────────────────────────
        self._last_seizure_prob = 0.0
        self._infer_lock        = threading.Lock()
        self._in_flight         = 0   # number of HTTP requests currently in-flight

        # Generation counter — incremented on every segment reset so that
        # results from the previous segment are silently discarded.
        self._generation  = 0

        if self.mode == "KAGGLE":
            print("🚀 Initialize SeizureClassifier in KAGGLE Mode (async fire-and-forget)")
            if not self.kaggle_url:
                print("⚠️  KAGGLE_ENDPOINT not set in .env! Inference will fail.")
            self.model = None
        else:
            model_path = str(model_path)
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Seizure model not found: {model_path}")
            print(f"Loading MoViNet-A2 Seizure Classifier from {model_path} ...")
            self.model = load_movinet(model_path, clip_frames=SEIZURE_CLIP_FRAMES)
            print("✓ Seizure MoViNet-A2 loaded successfully")

    # ── Encoding ──────────────────────────────────────────────────────────────
    def _encode_clip_to_b64(self, clip_array: np.ndarray):
        frames_b64 = []
        for i in range(clip_array.shape[0]):
            img_uint8 = (clip_array[i] * 255.0).astype(np.uint8)
            img_bgr   = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            _, buf    = cv2.imencode('.jpg', img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            frames_b64.append(base64.b64encode(buf).decode('utf-8'))
        return frames_b64

    # ── Background HTTP worker ────────────────────────────────────────────────
    def _fire_kaggle_request(self, clip: np.ndarray, gen: int):
        """
        Runs in a daemon thread. Retries up to MAX_RETRIES times with
        exponential backoff so transient SSL/EOF tunnel errors are recovered
        automatically without spamming the console.
        """
        url        = f"{self.kaggle_url.rstrip('/')}/predict/seizure"
        frames_b64 = self._encode_clip_to_b64(clip)
        last_err   = None
        succeeded  = False

        try:
            for attempt in range(MAX_RETRIES):
                with self._infer_lock:
                    if self._generation != gen:
                        return  # segment changed — discard silently

                try:
                    resp = requests.post(
                        url,
                        json={"frames_b64": frames_b64},
                        timeout=45.0
                    )
                    if resp.status_code == 200:
                        prob = float(resp.json().get("seizure_prob", 0.0))
                        with self._infer_lock:
                            if self._generation == gen and prob > self._last_seizure_prob:
                                self._last_seizure_prob = prob
                        print(f"  [SeizureClassifier] seizure_prob={prob:.3f}")
                        succeeded = True
                        return
                    else:
                        last_err = f"HTTP {resp.status_code}"
                except Exception as e:
                    last_err = str(e)

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_S * (2 ** attempt))

            if not succeeded:
                print(f"[SeizureClassifier] Failed after {MAX_RETRIES} attempts: "
                      f"{(last_err or '')[:100]}")
        finally:
            with self._infer_lock:
                self._in_flight = max(0, self._in_flight - 1)

    # ── Background LOCAL worker (fire-and-forget, mirrors KAGGLE pattern) ──────
    def _fire_local_request(self, x: np.ndarray, gen: int):
        """Runs GPU inference in a daemon thread. Never blocks the frame loop."""
        try:
            x_input      = np.expand_dims(x, axis=0)
            seizure_prob = float(self.model.predict_on_batch(x_input).flatten()[0])
        except Exception as e:
            print(f"  [SeizureClassifier] LOCAL inference error: {e}")
            seizure_prob = 0.0
        finally:
            with self._infer_lock:
                if gen == self._generation:   # discard stale results from old segments
                    self._last_seizure_prob = seizure_prob
                self._in_flight = max(0, self._in_flight - 1)

    # ── Segment lifecycle ─────────────────────────────────────────────────────
    def reset_for_segment(self) -> None:
        """
        NON-BLOCKING. Advances generation so stale in-flight results are discarded.
        Resets cached probability so the new segment starts clean.
        """
        with self._infer_lock:
            self._generation       += 1
            self._last_seizure_prob = 0.0
            self._in_flight         = 0

    def wait_for_pending(self, timeout: float = 25.0) -> None:
        """Spin-wait until all in-flight requests finish or timeout elapses."""
        t0 = time.time()
        while self._in_flight > 0 and (time.time() - t0) < timeout:
            time.sleep(0.05)

    # ── Frame feed ────────────────────────────────────────────────────────────
    def update(self, frame):
        self.frame_buffer.append(frame.copy())

    def is_ready(self):
        return len(self.frame_buffer) == SEIZURE_BUFFER

    def reset(self):
        self.frame_buffer.clear()

    # ── Inference ─────────────────────────────────────────────────────────────
    def classify(self, detection=None):
        """
        Returns dict with seizure_prob / normal_prob / class / confidence  OR  None.

        KAGGLE mode:
          - If no request in-flight: fire async request, return last known prob immediately.
          - If request in-flight: store this clip in the single-slot queue (overwriting
            any previously queued clip) so the background thread picks it up next.
            Return last known prob immediately. Frame loop is NEVER blocked.
        """
        if not self.is_ready():
            return None

        frames = list(self.frame_buffer)

        clip = []
        for i in range(0, SEIZURE_BUFFER, 2):   # stride-2: 64→32
            frm = frames[i]
            if detection is not None:
                x1, y1, x2, y2 = detection['bbox']
                x1, y1 = max(0, x1), max(0, y1)
                if x2 > x1 and y2 > y1:
                    crop = frm[y1:y2, x1:x2]
                    frm  = cv2.resize(crop, (self.target_size, self.target_size)) if crop.size > 0 \
                           else cv2.resize(frm, (self.target_size, self.target_size))
                else:
                    frm = cv2.resize(frm, (self.target_size, self.target_size))
            else:
                frm = cv2.resize(frm, (self.target_size, self.target_size))
            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clip.append(rgb)

        x = np.stack(clip)   # (32, H, W, 3)

        if self.mode == "KAGGLE" and self.kaggle_url:
            gen = self._generation
            with self._infer_lock:
                can_fire = self._in_flight < MAX_CONCURRENT
                if can_fire:
                    self._in_flight += 1

            if can_fire:
                t = threading.Thread(
                    target=self._fire_kaggle_request,
                    args=(x.copy(), gen),
                    daemon=True
                )
                t.start()

            with self._infer_lock:
                seizure_prob = self._last_seizure_prob
        else:
            # LOCAL mode: fire-and-forget in background thread (mirrors KAGGLE pattern)
            # Never blocks — returns last known probability immediately.
            gen = self._generation
            with self._infer_lock:
                can_fire = self._in_flight < MAX_CONCURRENT
                if can_fire:
                    self._in_flight += 1

            if can_fire:
                t = threading.Thread(
                    target=self._fire_local_request,
                    args=(x.copy(), gen),
                    daemon=True
                )
                t.start()

            with self._infer_lock:
                seizure_prob = self._last_seizure_prob

        normal_prob = 1.0 - seizure_prob
        return {
            'class':        'seizure' if seizure_prob >= 0.30 else 'normal',
            'confidence':   max(seizure_prob, normal_prob),
            'seizure_prob': seizure_prob,
            'normal_prob':  normal_prob,
        }
