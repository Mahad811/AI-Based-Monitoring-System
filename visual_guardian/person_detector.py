"""
Person Detector Module

Wraps YOLOv11n pretrained on COCO for person detection.
Supports OpenVINO-exported models for Intel iGPU acceleration.
This is a shared utility used by both fall and seizure classifiers.
"""

import numpy as np
from ultralytics import YOLO


class PersonDetector:
    """
    Detects persons in video frames using YOLO pretrained on COCO.
    Returns bounding boxes with optional padding for use in temporal encoders.
    """
    
    def __init__(self, model_path='yolo11n_openvino_model', confidence=0.5, process_every=3, device='intel:cpu'):
        """
        Args:
            model_path: Path to YOLO model or OpenVINO folder (e.g. 'yolo11n_openvino_model')
            confidence: Minimum confidence threshold for detections
            process_every: Run heavy YOLO inference every N frames. Caches box. (Optimize CPU)
            device: OpenVINO device string — 'intel:gpu' for Intel iGPU, 'intel:cpu' for CPU
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.process_every = process_every
        self.device = device
        self._frame_count = 0
        self._last_detection = None
        # Simple runtime check so you can see what backend/device is being used
        try:
            print(f"[PersonDetector] Loaded model='{model_path}' with device='{self.device}'")
        except Exception:
            # Avoid crashing on environments where stdout is not available
            pass
    
    def detect(self, frame, padding=0.0):
        """
        Detect person in frame and return bounding box with optional padding
        
        Args:
            frame: numpy array (H, W, 3) in BGR format
            padding: Bbox expansion factor (e.g., 0.2 = 20% padding)
            
        Returns:
            dict with keys:
                - 'bbox': (x1, y1, x2, y2) with padding, clamped to frame bounds
                - 'confidence': detection confidence (0-1)
                - 'center': (cx, cy) center point of bbox
            Returns None if no person detected
        """
        self._frame_count += 1
        
        # ── Frame Skip Optimization ──
        # Patient in bed doesn't move much in 3 frames (100ms). Reuse cached box!
        if self._frame_count % self.process_every != 1 and self._last_detection is not None:
            return self._last_detection

        # Run heavy detection
        results = self.model(frame, verbose=False, classes=[0], device=self.device)
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            self._last_detection = None
            return None
        
        # Get highest confidence detection
        boxes = results[0].boxes
        confidences = boxes.conf.cpu().numpy()
        
        # Filter by confidence threshold
        valid_indices = np.where(confidences >= self.confidence)[0]
        if len(valid_indices) == 0:
            return None
        
        best_idx = valid_indices[np.argmax(confidences[valid_indices])]
        bbox = boxes.xyxy[best_idx].cpu().numpy()
        confidence = float(confidences[best_idx])
        
        # Apply padding
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        
        pad_w = w * padding
        pad_h = h * padding
        
        # Expand and clamp to frame bounds
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(frame.shape[1], x2 + pad_w)
        y2_padded = min(frame.shape[0], y2 + pad_h)
        
        bbox_padded = (int(x1_padded), int(y1_padded), int(x2_padded), int(y2_padded))
        
        # Calculate center
        center = (int((x1_padded + x2_padded) / 2), int((y1_padded + y2_padded) / 2))
        
        result = {
            'bbox': bbox_padded,
            'confidence': confidence,
            'center': center
        }
        self._last_detection = result
        return result
    
    def detect_batch(self, frames, padding=0.0):
        """
        Detect persons in multiple frames
        
        Args:
            frames: list of numpy arrays (H, W, 3)
            padding: Bbox expansion factor
            
        Returns:
            list of detection dicts (or None for frames with no detection)
        """
        return [self.detect(frame, padding) for frame in frames]

    def reset(self):
        """
        Reset cached detection state. Call at every segment boundary so the
        stale bounding box from the previous clip is never reused as the first
        detection in the new clip.
        """
        self._frame_count    = 0
        self._last_detection = None
