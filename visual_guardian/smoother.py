"""
Sliding Window Smoother Module

Smooths predictions over time using a sliding window to reduce false positives
from transient misclassifications.
"""

from collections import deque
import numpy as np


class SlidingWindowSmoother:
    """
    Maintains a sliding window of recent predictions and returns smoothed probabilities.
    Helps stabilize detections by averaging over time.
    """
    
    def __init__(self, window_size=10):
        """
        Args:
            window_size: Number of recent predictions to average (default: 10 frames)
        """
        self.window_size = window_size
        self.probability_buffer = deque(maxlen=window_size)
    
    def update(self, probability):
        """
        Add a new probability to the sliding window
        
        Args:
            probability: float in [0, 1] representing P(positive class)
            
        Returns:
            smoothed_probability: float, average over sliding window
        """
        self.probability_buffer.append(probability)
        return self.get_smoothed()
    
    def get_smoothed(self):
        """
        Get current smoothed probability
        
        Returns:
            float: average probability over window, or 0.0 if buffer is empty
        """
        if len(self.probability_buffer) == 0:
            return 0.0
        return float(np.mean(self.probability_buffer))
    
    def is_alert(self, threshold=0.5):
        """
        Check if smoothed probability exceeds threshold (alert condition)
        
        Args:
            threshold: Alert threshold (default: 0.5)
            
        Returns:
            bool: True if smoothed probability >= threshold
        """
        return self.get_smoothed() >= threshold
    
    def reset(self):
        """Clear the probability buffer"""
        self.probability_buffer.clear()
    
    def is_warmed_up(self):
        """Check if buffer is full (warmed up)"""
        return len(self.probability_buffer) == self.window_size
