"""
Visual Guardian - Vision Module for Patient Monitoring

Components:
- PersonDetector: Shared person detection using YOLOv8n
- TemporalEncoder: Encodes 3 consecutive frames into temporal RGB (V2: RGB stacking)
- FallClassifier: EfficientNet-B0 for fall detection
- SeizureClassifier: EfficientNet-B0 for seizure detection with motion summaries
- SlidingWindowSmoother: Temporal smoothing of predictions
- VisionPipeline: Main orchestrator integrating all components
"""

from .person_detector import PersonDetector
from .temporal_encoder import TemporalEncoder
from .fall_classifier import FallClassifier
from .seizure_classifier import SeizureClassifier
from .smoother import SlidingWindowSmoother
from .pipeline import VisionPipeline

__all__ = [
    'PersonDetector',
    'TemporalEncoder',
    'FallClassifier',
    'SeizureClassifier',
    'SlidingWindowSmoother',
    'VisionPipeline'
]

__version__ = '2.0.0'
