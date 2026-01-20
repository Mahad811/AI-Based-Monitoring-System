"""
Cognitive Core Module
"Reflex-then-Reason" AI engine for Vital Guardian.

Layer 1 — ReflexEngine:    Instant (<50ms) deterministic alerts
Layer 2 — ReasoningEngine: Gemini 3 Flash multimodal decision brain
"""

from .models import (
    AlertLevel,
    AudioEvent,
    CognitiveCoreAlert,
    GeminiDecision,
    IncidentReport,
    ReflexAlert,
    VisionEvent,
)
from .reflex_engine import ReflexEngine
from .reasoning_engine import ReasoningEngine
from .audio_simulator import AudioSimulator
from .core import CognitiveCore

__all__ = [
    # Orchestrator
    "CognitiveCore",
    # Engines
    "ReflexEngine",
    "ReasoningEngine",
    "AudioSimulator",
    # Data models
    "AlertLevel",
    "GeminiDecision",
    "VisionEvent",
    "AudioEvent",
    "ReflexAlert",
    "IncidentReport",
    "CognitiveCoreAlert",
]
