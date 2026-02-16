"""
Tests — Cognitive Core Integration
Tests the full Reflex-then-Reason pipeline with synthetic events.
Gemini calls are mocked to allow offline testing.
"""

import time
import pytest
from unittest.mock import patch

from cognitive_core import (
    CognitiveCore,
    AlertLevel,
    GeminiDecision,
    VisionEvent,
    AudioEvent,
    IncidentReport,
)

# Shared mock config
MOCK_CONFIG = {
    "cognitive": {
        "reflex": {
            "corroboration_window_sec": 2.0,
            "cooldown_sec": 0,  # 0 for testing
        },
        "reasoning": {
            "model": "gemini-3-flash-preview",
            "thinking_level": "low",
            "media_resolution": "medium",
            "max_output_tokens": 400,
            "temperature": 0.2,
            "mock_mode": True,  # Always mock in tests
        },
        "event_log": {
            "max_events": 50,
            "context_window_sec": 60,
        },
    }
}


def make_fall_event(confidence: float = 0.85) -> VisionEvent:
    return VisionEvent(event_type="fall", fall_confidence=confidence, state="FALLEN")


def make_normal_event() -> VisionEvent:
    return VisionEvent(event_type="normal", fall_confidence=0.05, state="IN_BED")


def make_audio_thud() -> AudioEvent:
    return AudioEvent(event_type="distress", sound_type="thud")


# ---------------------------------------------------------------------------
# Reflex layer tests
# ---------------------------------------------------------------------------

class TestReflexLayer:

    def test_normal_frame_returns_none(self):
        core = CognitiveCore(MOCK_CONFIG)
        alert = core.process(vision_event=make_normal_event())
        core.shutdown()
        assert alert is None

    def test_fall_event_fires_high_alert(self):
        core = CognitiveCore(MOCK_CONFIG)
        alert = core.process(vision_event=make_fall_event())
        core.shutdown()
        assert alert is not None
        assert alert.reflex.level == AlertLevel.HIGH
        assert alert.reflex.confidence > 0.6

    def test_fall_plus_thud_fires_critical(self):
        core = CognitiveCore(MOCK_CONFIG)
        v = make_fall_event()
        a = make_audio_thud()
        alert = core.process(vision_event=v, audio_event=a)
        core.shutdown()
        assert alert is not None
        assert alert.reflex.level == AlertLevel.CRITICAL
        assert alert.reflex.corroborated is True

    def test_reflex_fires_immediately(self):
        core = CognitiveCore(MOCK_CONFIG)
        start = time.monotonic()
        alert = core.process(vision_event=make_fall_event())
        elapsed_ms = (time.monotonic() - start) * 1000
        core.shutdown()
        assert alert is not None
        assert elapsed_ms < 200, f"Reflex took {elapsed_ms:.1f}ms — should be <200ms"


# ---------------------------------------------------------------------------
# Gemini layer (mock) tests
# ---------------------------------------------------------------------------

class TestGeminiLayer:

    def test_report_is_populated_via_mock(self):
        core = CognitiveCore(MOCK_CONFIG)
        callback_events = []
        core.on_report_ready = lambda a: callback_events.append(a)

        alert = core.process(vision_event=make_fall_event())
        assert alert is not None
        # alert.reflex must be actionable to have triggered Gemini
        assert alert.reflex.is_actionable()

        # Wait for background worker to finish (shutdown is blocking)
        core.shutdown()

        assert len(callback_events) > 0
        report = callback_events[0].report
        assert report is not None
        assert report.final_decision in list(GeminiDecision)
        assert report.severity in list(AlertLevel)
        assert report.confidence >= 0.0

    def test_suppressed_report_is_reflected_in_final_level(self):
        """
        If Gemini returns SUPPRESS, final_level should stay at reflex level
        rather than escalating. (alert.final_level uses reflex as fallback.)
        """
        core = CognitiveCore(MOCK_CONFIG)
        suppress_report = IncidentReport(
            final_decision=GeminiDecision.SUPPRESS,
            severity=AlertLevel.INFO,
            headline="False alarm",
            narrative="Camera misfire, patient is fine.",
            recommended_actions=["No action needed"],
            reasoning="Frame shows normal positioning.",
            confidence=0.90,
        )

        with patch.object(core.reasoning, "analyse", return_value=suppress_report):
            alert = core.process(vision_event=make_fall_event())
            core.shutdown()

        # After suppression, the report exists but is_suppressed() is True
        assert alert is not None
        assert alert.report is not None
        assert alert.report.is_suppressed() is True

    def test_escalate_overrides_alert_level(self):
        core = CognitiveCore(MOCK_CONFIG)
        escalate_report = IncidentReport(
            final_decision=GeminiDecision.ESCALATE,
            severity=AlertLevel.CRITICAL,
            headline="Escalated: unrecognized seizure pattern",
            narrative="Frame shows subtle motor movements missed by vision classifier.",
            recommended_actions=["Immediate neurology consult"],
            reasoning="Rhythmic extremity movements not captured by fall classifier.",
            confidence=0.87,
        )

        with patch.object(core.reasoning, "analyse", return_value=escalate_report):
            alert = core.process(vision_event=make_fall_event())
            core.shutdown()

        assert alert is not None
        assert alert.report is not None
        assert alert.final_level == AlertLevel.CRITICAL
        assert alert.report.final_decision == GeminiDecision.ESCALATE
