import os
import time
import numpy as np
import cv2
import base64
import requests
import threading
from pathlib import Path
from .movinet_loader import load_movinet

FALL_CLIP_FRAMES = 16


class FallClassifier:
    def __init__(self, model_path, device='auto', use_ensemble=False):
        self.mode       = os.getenv("INFERENCE_MODE", "LOCAL").upper()
        self.kaggle_url = os.getenv("KAGGLE_ENDPOINT", "")

        # ── Async inference state (KAGGLE mode only) ─────────────────────────
        self._last_fall_prob = 0.0
        self._infer_lock     = threading.Lock()
        self._pending        = False   # True while an HTTP call is in-flight

        # Generation counter — incremented on every segment boundary.
        # Each background thread captures the generation at launch time; if the
        # generation has advanced by the time the response arrives the result is
        # silently discarded.  This makes reset_for_segment() instantaneous
        # (no blocking wait) while still preventing cross-segment contamination.
        self._generation = 0
        # Single-slot queue: when _pending=True, classify() stores the most
        # recent clip here instead of dropping it.  The background thread drains
        # it immediately after finishing so the best available window is always
        # evaluated, not just the first one.
        self._queued_clip = None
        self._queued_gen  = 0

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
        Runs in a daemon thread.
        *gen* is the segment generation captured at launch time.
        If self._generation has advanced by the time we respond, the result is
        discarded so it cannot contaminate the current segment.
        """
        try:
            frames_b64 = self._encode_clip_to_b64(clip)
            resp = requests.post(
                f"{self.kaggle_url.rstrip('/')}/predict/fall",
                json={"frames_b64": frames_b64},
                timeout=45.0
            )
            if resp.status_code == 200:
                prob = float(resp.json().get("fall_prob", 0.0))
                with self._infer_lock:
                    if self._generation == gen:      # still the same segment?
                        self._last_fall_prob = prob
                        print(f"  [FallClassifier] fall_prob={prob:.3f} (gen {gen})")
                    else:
                        print(f"  [FallClassifier] Discarding stale result "
                              f"prob={prob:.3f} (gen {gen} vs current {self._generation})")
            else:
                print(f"[FallClassifier] Kaggle API error {resp.status_code}: "
                      f"{resp.text[:120]}")
        except Exception as e:
            print(f"[FallClassifier] Kaggle request failed: {e}")
        finally:
            # ── Drain the queue (collect values inside lock, start thread outside) ──
            next_clip = None
            next_gen  = 0
            with self._infer_lock:
                if self._generation == gen:       # still the current segment?
                    self._pending = False
                    if self._queued_clip is not None:
                        next_clip         = self._queued_clip
                        next_gen          = self._queued_gen
                        self._queued_clip = None
                        self._queued_gen  = 0
                        self._pending     = True   # re-latch before releasing lock

            if next_clip is not None:
                t = threading.Thread(
                    target=self._fire_kaggle_request,
                    args=(next_clip, next_gen),
                    daemon=True
                )
                t.start()

    # ── Segment lifecycle ─────────────────────────────────────────────────────
    def reset_for_segment(self) -> None:
        """
        NON-BLOCKING. Call at the START of every new segment (no asyncio.to_thread needed).

        Advances the generation counter so any in-flight HTTP request from the
        previous segment will be silently discarded when it eventually arrives.
        Immediately zeroes _last_fall_prob, _pending and the queued-clip slot so
        this segment starts with a completely clean state without waiting at all.
        """
        with self._infer_lock:
            self._generation    += 1
            self._last_fall_prob = 0.0
            self._queued_clip    = None
            self._queued_gen     = 0
        self._pending = False   # safe: old thread checks generation before using it

    def wait_for_pending(self, timeout: float = 25.0) -> None:
        """
        Spin-wait until the in-flight request for the CURRENT generation finishes
        or *timeout* seconds elapse.  Call via asyncio.to_thread() from the demo loop.
        25 s is enough for Kaggle to respond and short enough that it doesn't freeze
        a 5-second-clip demo.
        """
        t0 = time.time()
        while self._pending and (time.time() - t0) < timeout:
            time.sleep(0.05)

    # ── Public interface ──────────────────────────────────────────────────────
    def classify(self, clip):
        """
        Args:
            clip: ndarray (16, 224, 224, 3) float32 [0,1] RGB
        Returns:
            dict with fall_prob / normal_prob / class / confidence   OR   None
        """
        if clip is None:
            return None

        if self.mode == "KAGGLE" and self.kaggle_url:
            gen = self._generation          # capture current generation
            if not self._pending:
                self._pending = True
                t = threading.Thread(
                    target=self._fire_kaggle_request,
                    args=(clip.copy(), gen),
                    daemon=True
                )
                t.start()
            else:
                # Queue the most-recent clip instead of dropping it silently.
                # The background thread will fire it as soon as it finishes.
                with self._infer_lock:
                    self._queued_clip = clip.copy()
                    self._queued_gen  = gen
            # Return the last known result immediately — never block the frame loop
            with self._infer_lock:
                fall_prob = self._last_fall_prob
        else:
            x         = np.expand_dims(clip, axis=0)
            fall_prob = float(self.model.predict_on_batch(x).flatten()[0])

        normal_prob = 1.0 - fall_prob
        return {
            'class':       'fall' if fall_prob >= 0.55 else 'normal',
            'confidence':  max(fall_prob, normal_prob),
            'fall_prob':   fall_prob,
            'normal_prob': normal_prob,
        }
