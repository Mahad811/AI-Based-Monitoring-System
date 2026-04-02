"""
Temporal Encoder (MoViNet V2)

Maintains a 32-frame rolling buffer of raw BGR frames.
Outputs 16-frame stride-2 clips in RGB [0,1] ready for Fall MoViNet-A2.
"""

import cv2
import numpy as np
from collections import deque


class TemporalEncoder:
    BUFFER = 32   # 32 raw frames -> stride-2 -> 16 model frames

    def __init__(self, buffer_size=32, frame_size=224):
        self.frame_size = frame_size
        self.frame_buffer = deque(maxlen=self.BUFFER)

    def update(self, frame):
        self.frame_buffer.append(frame.copy())

    def encode(self, detection=None):
        """
        Returns ndarray (16, frame_size, frame_size, 3) float32 in [0,1] RGB,
        or None if buffer not full / no person detected.
        """
        if len(self.frame_buffer) < self.BUFFER:
            return None

        frames = list(self.frame_buffer)

        clip = []
        for i in range(0, self.BUFFER, 2):   # stride-2: indices 0,2,4,...,30 -> 16 frames
            frm = frames[i]
            if detection is not None:
                x1, y1, x2, y2 = detection['bbox']
                x1, y1 = max(0, x1), max(0, y1)
                if x2 > x1 and y2 > y1:
                    crop = frm[y1:y2, x1:x2]
                    if crop.size > 0:
                        frm = cv2.resize(crop, (self.frame_size, self.frame_size))
                    else:
                        frm = cv2.resize(frm, (self.frame_size, self.frame_size))
                else:
                    frm = cv2.resize(frm, (self.frame_size, self.frame_size))
            else:
                frm = cv2.resize(frm, (self.frame_size, self.frame_size))

            rgb = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            clip.append(rgb)

        return np.stack(clip)   # (16, H, W, 3)

    def reset(self):
        self.frame_buffer.clear()

    def is_ready(self):
        return len(self.frame_buffer) == self.BUFFER
