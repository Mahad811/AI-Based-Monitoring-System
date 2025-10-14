"""
Temporal Encoder Module (V2: Temporal RGB Triplets)

Maintains a rolling buffer of frames and encodes 3 consecutive frames
into a temporal RGB image for fall detection using RGB stacking.
"""

import cv2
import numpy as np
from collections import deque


class TemporalEncoder:
    """
    Encodes temporal context using 3 consecutive frames (V2: RGB stacking):
    - Red channel = grayscale(frame[t-1])  # Past frame
    - Green channel = grayscale(frame[t])   # Current frame (appearance)
    - Blue channel = grayscale(frame[t+1]) # Future frame
    
    Uses consistent bbox cropping: detects person in middle frame (t),
    then crops all 3 frames using the same bbox to preserve relative motion.
    """
    
    def __init__(self, buffer_size=3, frame_size=224):
        """
        Args:
            buffer_size: Number of frames to buffer (default: 3 = ~0.1s @ 30fps)
            frame_size: Output image size (default: 224x224)
        """
        self.buffer_size = buffer_size
        self.frame_size = frame_size
        self.frame_buffer = deque(maxlen=buffer_size)
    
    def update(self, frame):
        """
        Add a frame to the rolling buffer
        
        Args:
            frame: numpy array (H, W, 3) in BGR format
        """
        self.frame_buffer.append(frame.copy())
    
    def encode(self, person_detector, padding=0.2):
        """
        Encode current buffer into temporal RGB image (V2: RGB stacking)
        
        Args:
            person_detector: PersonDetector instance (shared utility)
            padding: Bbox padding factor (default: 0.2 = 20%)
            
        Returns:
            temporal_rgb: (224, 224, 3) numpy array, or None if:
                - Buffer not full (warmup period)
                - No person detected in middle frame
                - Cropping failed
        """
        # Check if buffer is full
        if len(self.frame_buffer) < self.buffer_size:
            return None
        
        # Get frames as list (should be 3 frames: [t-1, t, t+1])
        frames = list(self.frame_buffer)
        
        if len(frames) != 3:
            return None
        
        # Detect person in MIDDLE frame (frame_t)
        middle_frame = frames[1]  # Index 1 = middle frame
        
        detection = person_detector.detect(middle_frame, padding=padding)
        
        if detection is None:
            return None
        
        bbox = detection['bbox']
        x1, y1, x2, y2 = bbox
        
        # Validate bbox
        if x2 <= x1 or y2 <= y1:
            return None
        
        try:
            # Crop all 3 frames using the SAME bbox (preserves motion within bbox)
            cropped_frames = []
            for frame in frames:
                cropped = frame[y1:y2, x1:x2]
                if cropped.size == 0:
                    return None
                # Convert to grayscale
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                cropped_frames.append(gray)
            
            # V2 Encoding: RGB stacking
            # R channel = frame[t-1] (past)
            # G channel = frame[t] (current - appearance)
            # B channel = frame[t+1] (future)
            R = cropped_frames[0]  # t-1
            G = cropped_frames[1]  # t (current frame - appearance)
            B = cropped_frames[2]  # t+1
            
            # Stack as RGB
            temporal_rgb = np.stack([R, G, B], axis=-1)
            
            # Resize to target size
            temporal_rgb = cv2.resize(temporal_rgb, (self.frame_size, self.frame_size))
            
            return temporal_rgb
            
        except Exception as e:
            # Handle any cropping/resizing errors
            return None
    
    def reset(self):
        """Clear the frame buffer"""
        self.frame_buffer.clear()
    
    def is_ready(self):
        """Check if buffer is full and ready to encode"""
        return len(self.frame_buffer) == self.buffer_size
