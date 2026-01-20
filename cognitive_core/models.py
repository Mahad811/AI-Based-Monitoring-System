"""
Cognitive Core — Shared Data Models
All dataclasses, enums, and Pydantic schemas used across the Cognitive Core.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AlertLevel(str, enum.Enum):
    """Alert severity levels, ordered from lowest to highest."""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def is_at_least(self, other: "AlertLevel") -> bool:
        _ORDER = ["INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        return _ORDER.index(self.value) >= _ORDER.index(other.value)


class GeminiDecision(str, enum.Enum):
    """Gemini's final ruling on the reflex alert."""
    CONFIRM   = "CONFIRM"    # Reflex was right — validate the alert
    ESCALATE  = "ESCALATE"   # Situation is MORE serious than reflex detected
    SUPPRESS  = "SUPPRESS"   # False alarm — override the reflex, no incident
    OVERRIDE  = "OVERRIDE"   # Change the alert level (up or down)


# ---------------------------------------------------------------------------
# Input Event Types
# ---------------------------------------------------------------------------

@dataclass
class VisionEvent:
    """Structured event from the VisionPipeline."""
    event_type: str            # 'fall', 'seizure', 'normal', 'in_bed', 'restlessness', etc.
    fall_confidence: float = 0.0
    seizure_confidence: float = 0.0
    fall_smoothed: float = 0.0
    seizure_smoothed: float = 0.0
    person_bbox: Optional[List[int]] = None
    state: str = "UNKNOWN"     # 'IN_BED', 'OUT_OF_BED', 'FALLEN', etc.
    debug_info: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @classmethod
    def from_pipeline_event(cls, event: dict) -> "VisionEvent":
        """Build a VisionEvent from the raw dict returned by VisionPipeline.process_frame()."""
        return cls(
            event_type=event.get("event_type", "normal"),
            fall_confidence=event.get("fall_confidence", 0.0),
            seizure_confidence=event.get("seizure_confidence", 0.0),
            fall_smoothed=event.get("fall_smoothed", 0.0),
            seizure_smoothed=event.get("seizure_smoothed", 0.0),
            person_bbox=event.get("person_bbox"),
            state=event.get("state", "UNKNOWN"),
            debug_info=event.get("debug_info", ""),
            timestamp=event.get("timestamp", datetime.now().isoformat()),
        )

    def is_significant(self) -> bool:
        """Return True if this event warrants cognitive processing."""
        return self.event_type in ("fall", "force_fall", "seizure", "restlessness",
                                   "bed_exit", "missing_patient")

    def __str__(self) -> str:
        conf = self.fall_confidence if "fall" in self.event_type else self.seizure_confidence
        return f"VisionEvent({self.event_type}, conf={conf:.2f}, state={self.state})"


@dataclass
class AudioEvent:
    """Structured event from AudioSimulator or a real audio module."""
    event_type: str            # 'distress', 'keyword', 'silence'
    sound_type: Optional[str] = None   # 'thud', 'scream', 'groan', 'gasp'
    keyword: Optional[str] = None      # 'help', 'nurse', 'madad', etc.
    language: Optional[str] = None     # 'english', 'urdu'
    confidence: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_distress(self) -> bool:
        return self.event_type == "distress"

    def is_keyword(self) -> bool:
        return self.event_type == "keyword"

    def __str__(self) -> str:
        if self.sound_type:
            return f"AudioEvent(distress={self.sound_type}, conf={self.confidence:.2f})"
        if self.keyword:
            return f"AudioEvent(keyword='{self.keyword}', lang={self.language})"
        return f"AudioEvent({self.event_type})"


# ---------------------------------------------------------------------------
# Reflex Engine Output
# ---------------------------------------------------------------------------

@dataclass
class ReflexAlert:
    """Preliminary alert produced by the ReflexEngine, <50ms, no API."""
    level: AlertLevel
    confidence: float
    message: str
    corroborated: bool = False          # True if BOTH vision + audio fired
    vision_event: Optional[VisionEvent] = None
    audio_event: Optional[AudioEvent] = None
    trigger_reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def is_actionable(self) -> bool:
        """Return True if this alert should trigger Gemini reasoning."""
        return self.level.is_at_least(AlertLevel.MEDIUM)

    def __str__(self) -> str:
        cb = "✓" if self.corroborated else "?"
        return f"[{cb}] ReflexAlert({self.level.value}, conf={self.confidence:.2f}): {self.message}"


# ---------------------------------------------------------------------------
# Gemini Reasoning Engine Output — Pydantic schema for structured output
# ---------------------------------------------------------------------------

class IncidentReport(BaseModel):
    """
    Structured JSON output returned by Gemini 3 Flash.
    Pydantic ensures the response is fully type-safe and validated.
    """
    reasoning: str = Field(
        description="Gemini's internal reasoning: what it saw in the frame and data "
                    "that led to its decision. IMPORTANT: Output your full thoughts here FIRST."
    )
    final_decision: GeminiDecision = Field(
        description="Gemini's ruling: CONFIRM/ESCALATE/SUPPRESS/OVERRIDE the reflex alert. ONLY USE THESE EXACT WORDS."
    )
    severity: AlertLevel = Field(
        description="Final alert severity after Gemini's analysis."
    )
    headline: str = Field(
        description="One-line summary of the incident (max 80 chars)."
    )
    narrative: str = Field(
        description="Clinical description of what was observed (50-120 words). "
                    "Factual, never diagnostic."
    )
    recommended_actions: List[str] = Field(
        description="Ordered list of 2-4 immediate actions for nursing staff."
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Gemini's confidence in its final_decision (0.0–1.0)."
    )

    def is_suppressed(self) -> bool:
        return self.final_decision == GeminiDecision.SUPPRESS

    def display_level_color(self) -> str:
        """Returns ANSI color code for terminal display."""
        colors = {
            AlertLevel.INFO:     "\033[37m",   # White
            AlertLevel.LOW:      "\033[32m",   # Green
            AlertLevel.MEDIUM:   "\033[33m",   # Yellow
            AlertLevel.HIGH:     "\033[91m",   # Orange/Red
            AlertLevel.CRITICAL: "\033[31m",   # Bright Red
        }
        return colors.get(self.severity, "\033[0m")


# ---------------------------------------------------------------------------
# Complete Alert (Reflex + Gemini combined)
# ---------------------------------------------------------------------------

@dataclass
class CognitiveCoreAlert:
    """
    The complete alert object held by CognitiveCore.
    Initially populated by the ReflexEngine, then upgraded by Gemini.
    """
    reflex: ReflexAlert
    report: Optional[IncidentReport] = None   # Populated async by Gemini
    gemini_pending: bool = False
    gemini_error: Optional[str] = None

    @property
    def final_level(self) -> AlertLevel:
        """Returns Gemini's verdict if available, else reflex level."""
        r = self.report
        if r is not None and not r.is_suppressed():
            return r.severity
        return self.reflex.level

    @property
    def is_resolved(self) -> bool:
        """True if Gemini has finalized the decision."""
        return (self.report is not None) or (self.gemini_error is not None)
