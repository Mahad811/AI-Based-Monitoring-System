import os
import time
import numpy as np
import cv2
import base64
import requests
import threading
from pathlib import Path
from .movinet_loader import load_movinet

FALL_CLIP_FRAMES  = 16
MAX_CONCURRENT    = 1   # keep at 1 — free ngrok tunnels drop SSL under parallel load
MAX_RETRIES       = 3   # retry failed requests with exponential backoff
RETRY_BASE_S      = 0.5 # first retry after 0.5s, then 1s, then 2s


class FallClassifier:
    def __init__(self, model_path, device='auto', use_ensemble=False):
        self.mode       = os.getenv("INFERENCE_MODE", "LOCAL").upper()
        self.kaggle_url = os.getenv("KAGGLE_ENDPOINT", "")

        # ── Async inference state (KAGGLE mode only) ─────────────────────────
        self._last_fall_prob = 0.0
        self._infer_lock     = threading.Lock()
        self._in_flight      = 0   # number of HTTP requests currently in-flight

        # Generation counter — incremented on every segment boundary so that
        # results from the previous segment are silently discarded.
        self._generation = 0

        if self.mode == "KAGGLE":
            print("🚀 Initialize FallClassifier in KAGGLE Mode (async fire-and-forget)")
            if not self.kaggle_url:
                print("⚠️  KAGGLE_ENDPOINT not set in .env! Inference will fail.")
            self.model = None
        else:
            print("Loading MoViNet-A2 Fall Classifier locally...")
            model_path = str(model_path)
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Fall model not found: {model_path}")
            self.model = load_movinet(model_path, clip_frames=FALL_CLIP_FRAMES)
            print("✓ Fall MoViNet-A2 loaded locally successfully")

    # ── Encoding ──────────────────────────────────────────────────────────────
    def _encode_clip_to_b64(self, clip: np.ndarray):
        frames_b64 = []
        for i in range(clip.shape[0]):
            img_uint8 = (clip[i] * 255.0).astype(np.uint8)
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
        url        = f"{self.kaggle_url.rstrip('/')}/predict/fall"
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
                        prob = float(resp.json().get("fall_prob", 0.0))
                        with self._infer_lock:
                            if self._generation == gen and prob > self._last_fall_prob:
                                self._last_fall_prob = prob
                        print(f"  [FallClassifier] fall_prob={prob:.3f}")
                        succeeded = True
                        return
                    else:
                        last_err = f"HTTP {resp.status_code}"
                except Exception as e:
                    last_err = str(e)

                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_BASE_S * (2 ** attempt))

            if not succeeded:
                print(f"[FallClassifier] Failed after {MAX_RETRIES} attempts: "
                      f"{(last_err or '')[:100]}")
        finally:
            with self._infer_lock:
                self._in_flight = max(0, self._in_flight - 1)

    # ── Background LOCAL worker (fire-and-forget, mirrors KAGGLE pattern) ──────
    def _fire_local_request(self, clip: np.ndarray, gen: int):
        """Runs GPU inference in a daemon thread. Never blocks the frame loop."""
        try:
            x         = np.expand_dims(clip, axis=0)
            fall_prob = float(self.model.predict_on_batch(x).flatten()[0])
        except Exception as e:
            print(f"  [FallClassifier] LOCAL inference error: {e}")
            fall_prob = 0.0
        finally:
            with self._infer_lock:
                if gen == self._generation:   # discard stale results from old segments
                    self._last_fall_prob = fall_prob
                self._in_flight = max(0, self._in_flight - 1)

    # ── Segment lifecycle ─────────────────────────────────────────────────────
    def reset_for_segment(self) -> None:
        """
        NON-BLOCKING. Advances generation so stale in-flight results are discarded.
        Resets cached probability so the new segment starts clean.
        """
        with self._infer_lock:
            self._generation    += 1
            self._last_fall_prob = 0.0
            self._in_flight      = 0

    def wait_for_pending(self, timeout: float = 25.0) -> None:
        """Spin-wait until all in-flight requests finish or timeout elapses."""
        t0 = time.time()
        while self._in_flight > 0 and (time.time() - t0) < timeout:
            time.sleep(0.05)

    # ── Public interface ──────────────────────────────────────────────────────
    def classify(self, clip):
        """
        Args:
            clip: ndarray (16, 224, 224, 3) float32 [0,1] RGB
        Returns:
            dict with fall_prob / normal_prob / class / confidence   OR   None

        KAGGLE mode: fires a new background thread if fewer than MAX_CONCURRENT
        requests are already in-flight, then immediately returns the last known
        result. Multiple overlapping requests mean the fastest response wins and
        the highest probability is always surfaced — dramatically reducing alert
        latency compared to the original single-request gate.
        """
        if clip is None:
            return None

        if self.mode == "KAGGLE" and self.kaggle_url:
            gen = self._generation
            with self._infer_lock:
                can_fire = self._in_flight < MAX_CONCURRENT
                if can_fire:
                    self._in_flight += 1

            if can_fire:
                t = threading.Thread(
                    target=self._fire_kaggle_request,
                    args=(clip.copy(), gen),
                    daemon=True
                )
                t.start()

            with self._infer_lock:
                fall_prob = self._last_fall_prob
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
                    args=(clip.copy(), gen),
                    daemon=True
                )
                t.start()

            with self._infer_lock:
                fall_prob = self._last_fall_prob

        normal_prob = 1.0 - fall_prob
        return {
            'class':       'fall' if fall_prob >= 0.55 else 'normal',
            'confidence':  max(fall_prob, normal_prob),
            'fall_prob':   fall_prob,
            'normal_prob': normal_prob,
        }
