"""
Tests — Gemini API Connection
Verifies that the Gemini API key is valid, the model responds,
and the IncidentReport Pydantic schema parses correctly.

Run this test AFTER setting GEMINI_API_KEY in your .env file.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")


# Skip all tests if no API key provided
HAS_KEY = bool(os.getenv("GEMINI_API_KEY"))
skip_no_key = pytest.mark.skipif(
    not HAS_KEY,
    reason="GEMINI_API_KEY not set — run with a valid key to test live API",
)


from cognitive_core.models import AlertLevel, AudioEvent, VisionEvent, ReflexAlert
from cognitive_core.reasoning_engine import ReasoningEngine


@pytest.fixture()
def live_engine():
    cfg = {
        "model": "gemini-3-flash-preview",
        "thinking_level": "low",
        "media_resolution": "low",   # Low for faster test
        "max_output_tokens": 300,
        "temperature": 0.2,
        "mock_mode": False,
    }
    return ReasoningEngine(cfg)


def make_reflex_alert(level: AlertLevel) -> ReflexAlert:
    return ReflexAlert(
        level=level,
        confidence=0.85,
        message="Automated test alert",
        corroborated=False,
    )


@skip_no_key
def test_gemini_returns_incident_report(live_engine):
    """Full end-to-end: Gemini responds with a valid IncidentReport."""
    reflex = make_reflex_alert(AlertLevel.HIGH)
    vision = [VisionEvent(event_type="fall", fall_confidence=0.81, state="FALLEN")]
    audio  = [AudioEvent(event_type="distress", sound_type="thud")]

    report = live_engine.analyse(reflex, vision, audio, frame=None)

    assert report is not None
    assert report.final_decision is not None
    assert report.severity in list(AlertLevel)
    assert isinstance(report.headline, str) and len(report.headline) > 0
    assert isinstance(report.narrative, str) and len(report.narrative) > 0
    assert isinstance(report.recommended_actions, list)
    assert len(report.recommended_actions) >= 1
    assert 0.0 <= report.confidence <= 1.0

    print(f"\n  Decision : {report.final_decision.value}")
    print(f"  Severity : {report.severity.value}")
    print(f"  Headline : {report.headline}")
    print(f"  Confidence: {report.confidence:.0%}")


@skip_no_key
def test_gemini_structured_output_is_parseable(live_engine):
    """Verify that Pydantic parsing succeeds — no raw JSON errors."""
    reflex = make_reflex_alert(AlertLevel.MEDIUM)
    report = live_engine.analyse(reflex, [], [], frame=None)
    # This will raise if IncidentReport.model_validate_json() fails
    assert report is not None


def test_mock_mode_returns_report():
    """Mock mode must always return a valid IncidentReport — no API needed."""
    cfg = {
        "model": "gemini-3-flash-preview",
        "thinking_level": "low",
        "media_resolution": "medium",
        "max_output_tokens": 400,
        "temperature": 0.2,
        "mock_mode": True,
    }
    engine = ReasoningEngine(cfg)
    reflex = make_reflex_alert(AlertLevel.CRITICAL)
    report = engine.analyse(reflex, [], [], frame=None)
    assert report is not None
    assert report.headline
    assert len(report.recommended_actions) >= 1
