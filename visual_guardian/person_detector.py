"""
Person Detector Module

Wraps YOLOv11n pretrained on COCO for person detection.
Supports both CUDA GPU (PyTorch) and CPU inference.

Device handling:
  - YOLO_FORCE_CPU=true env var: always use CPU regardless of config.
    Set this in Docker to avoid the cuDNN broken-context hang that occurs
    when TF's GPU init corrupts the shared CUDA state.
  - device="0" or "cuda:0": attempt CUDA GPU with explicit device per-call.
    If CUDA fails, falls back to CPU with explicit device='cpu' to prevent
    Ultralytics from auto-selecting the broken GPU again.
  - All CPU paths pass device='cpu' explicitly. This is critical: without it,
    Ultralytics auto-detects GPU (torch.cuda.is_available()=True even when
    cuDNN is broken) and hangs on CUDNN_STATUS_NOT_INITIALIZED.
"""

import os
import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """
    Detects persons in video frames using YOLO pretrained on COCO.
    Returns bounding boxes with optional padding for use in temporal encoders.
    """

    def __init__(self, model_path='yolo11n_openvino_model', confidence=0.5,
                 process_every=3, device='cpu'):
        """
        Args:
            model_path: Path to YOLO .pt model or OpenVINO folder
            confidence: Minimum confidence threshold for detections
            process_every: Run YOLO inference every N frames. Caches bbox in between.
            device: CUDA device index ("0") or "cpu"
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.process_every = process_every
        self._frame_count = 0
        self._last_detection = None

        # YOLO_FORCE_CPU=true → always use CPU (Docker workaround for broken cuDNN)
        _force_cpu = os.getenv("YOLO_FORCE_CPU", "false").lower() == "true"

        _dev_str = str(device).strip()
        self._is_cuda = (not _force_cpu) and (_dev_str.isdigit() or _dev_str.startswith("cuda"))
        self.device = f"cuda:{_dev_str}" if _dev_str.isdigit() else _dev_str

        if _force_cpu:
            print(f"[PersonDetector] Loaded '{model_path}' → CPU (YOLO_FORCE_CPU=true)")
        elif self._is_cuda:
            print(f"[PersonDetector] Loaded '{model_path}' → CUDA GPU ({self.device})")
        else:
            print(f"[PersonDetector] Loaded '{model_path}' → CPU")

    def _infer(self, frame):
        """Run YOLO inference with explicit device. Falls back to CPU if CUDA fails."""
        if self._is_cuda:
            try:
                return self.model(frame, verbose=False, classes=[0], device=self.device)
            except Exception as e:
                print(f"[PersonDetector] CUDA failed ({type(e).__name__}), switching to CPU permanently.")
                self._is_cuda = False
                # Explicit device='cpu' to prevent Ultralytics auto-GPU-select
        return self.model(frame, verbose=False, classes=[0], device='cpu')

    def detect(self, frame, padding=0.0):
        """
        Detect person in frame and return bounding box with optional padding.
        Returns dict with 'bbox', 'confidence', 'center', or None if no person.
        """
        self._frame_count += 1

        # Frame skip optimization: reuse cached bbox between YOLO calls
        if self._frame_count % self.process_every != 1 and self._last_detection is not None:
            return self._last_detection

        results = self._infer(frame)

        if len(results) == 0 or len(results[0].boxes) == 0:
            self._last_detection = None
            return None

        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()

        valid_indices = np.where(confidences >= self.confidence)[0]
        if len(valid_indices) == 0:
            return None

        best_idx = valid_indices[np.argmax(confidences[valid_indices])]
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(confidences[best_idx])

        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        pad_w = w * padding
        pad_h = h * padding

        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(frame.shape[1], x2 + pad_w)
        y2_padded = min(frame.shape[0], y2 + pad_h)

        bbox_padded = (int(x1_padded), int(y1_padded), int(x2_padded), int(y2_padded))
        center = (int((x1_padded + x2_padded) / 2), int((y1_padded + y2_padded) / 2))

        result = {
            'bbox':       bbox_padded,
            'confidence': confidence,
            'center':     center,
        }
        self._last_detection = result
        return result

    def detect_batch(self, frames, padding=0.0):
        """Detect persons in multiple frames."""
        return [self.detect(frame, padding) for frame in frames]

    def reset(self):
        """Reset cached detection state at every segment boundary."""
        self._frame_count    = 0
        self._last_detection = None
