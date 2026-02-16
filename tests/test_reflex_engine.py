"""
Tests — Reflex Engine
Tests the scoring matrix, corroboration logic, and cooldown behavior.
No API calls needed — pure Python tests.
"""

import pytest
from cognitive_core.models import AlertLevel, AudioEvent, VisionEvent
from cognitive_core.reflex_engine import ReflexEngine


@pytest.fixture()
def engine():
    cfg = {"corroboration_window_sec": 2.0, "cooldown_sec": 0}  # 0 cooldown for tests
    e = ReflexEngine(cfg)
    return e


def make_vision(event_type: str, confidence: float = 0.85) -> VisionEvent:
    return VisionEvent(
        event_type=event_type,
        fall_confidence=confidence if "fall" in event_type else 0.0,
        seizure_confidence=confidence if "seizure" in event_type else 0.0,
    )


def make_audio_distress(sound_type: str) -> AudioEvent:
    return AudioEvent(event_type="distress", sound_type=sound_type)


def make_audio_keyword(keyword: str) -> AudioEvent:
    return AudioEvent(event_type="keyword", keyword=keyword)


# ---------------------------------------------------------------------------
# Core fusion matrix tests
# ---------------------------------------------------------------------------

class TestFusionMatrix:

    def test_fall_plus_thud_is_critical(self, engine):
        v = make_vision("fall")
        a = make_audio_distress("thud")
        alert = engine.evaluate(v, a)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert alert.corroborated is True
        assert alert.confidence >= 0.90

    def test_fall_plus_scream_is_critical(self, engine):
        v = make_vision("fall")
        a = make_audio_distress("scream")
        alert = engine.evaluate(v, a)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_fall_plus_help_keyword_is_critical(self, engine):
        v = make_vision("fall")
        a = make_audio_keyword("help")
        alert = engine.evaluate(v, a)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL
        assert alert.corroborated is True

    def test_fall_plus_madad_is_critical(self, engine):
        v = make_vision("fall")
        a = make_audio_keyword("madad")
        alert = engine.evaluate(v, a)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_seizure_plus_groan_is_critical(self, engine):
        v = make_vision("seizure")
        a = make_audio_distress("groan")
        alert = engine.evaluate(v, a)
        assert alert is not None
        assert alert.level == AlertLevel.CRITICAL

    def test_fall_alone_is_high(self, engine):
        v = make_vision("fall")
        alert = engine.evaluate(v, None)
        assert alert is not None
        assert alert.level == AlertLevel.HIGH
        assert alert.corroborated is False

    def test_seizure_alone_is_high(self, engine):
        v = make_vision("seizure")
        alert = engine.evaluate(v, None)
        assert alert is not None
        assert alert.level == AlertLevel.HIGH

    def test_restlessness_alone_is_medium(self, engine):
        v = make_vision("restlessness")
        alert = engine.evaluate(v, None)
        assert alert is not None
        assert alert.level == AlertLevel.MEDIUM

    def test_help_keyword_no_vision_is_medium(self, engine):
        a = make_audio_keyword("help")
        alert = engine.evaluate(None, a)
        assert alert is not None
        assert alert.level == AlertLevel.MEDIUM

    def test_normal_event_returns_none(self, engine):
        v = make_vision("normal")
        alert = engine.evaluate(v, None)
        assert alert is None

    def test_no_event_returns_none(self, engine):
        alert = engine.evaluate(None, None)
        assert alert is None


# ---------------------------------------------------------------------------
# Cooldown tests
# ---------------------------------------------------------------------------

class TestCooldown:

    def test_cooldown_prevents_duplicate_alert(self):
        cfg = {"corroboration_window_sec": 2.0, "cooldown_sec": 60}
        engine = ReflexEngine(cfg)
        v = make_vision("fall")
        a = make_audio_distress("thud")

        alert1 = engine.evaluate(v, a)
        assert alert1 is not None  # First one fires

        alert2 = engine.evaluate(v, a)
        assert alert2 is None  # Second blocked by cooldown

    def test_different_alert_types_have_separate_cooldowns(self):
        cfg = {"corroboration_window_sec": 2.0, "cooldown_sec": 60}
        engine = ReflexEngine(cfg)
        v_fall  = make_vision("fall")
        v_seiz  = make_vision("seizure")

        alert_fall   = engine.evaluate(v_fall, None)
        alert_seizure = engine.evaluate(v_seiz, None)
        # Both should fire — different keys
        assert alert_fall    is not None
        assert alert_seizure is not None

    def test_reset_clears_cooldown(self):
        cfg = {"corroboration_window_sec": 2.0, "cooldown_sec": 60}
        engine = ReflexEngine(cfg)
        v = make_vision("fall")

        engine.evaluate(v, None)
        engine.reset_cooldowns()
        alert = engine.evaluate(v, None)
        assert alert is not None  # Fires again after reset


# ---------------------------------------------------------------------------
# AlertLevel ordering tests
# ---------------------------------------------------------------------------

class TestAlertLevelOrdering:

    def test_critical_is_at_least_high(self):
        assert AlertLevel.CRITICAL.is_at_least(AlertLevel.HIGH)

    def test_high_is_at_least_medium(self):
        assert AlertLevel.HIGH.is_at_least(AlertLevel.MEDIUM)

    def test_low_is_not_at_least_medium(self):
        assert not AlertLevel.LOW.is_at_least(AlertLevel.MEDIUM)

    def test_medium_is_not_at_least_high(self):
        assert not AlertLevel.MEDIUM.is_at_least(AlertLevel.HIGH)

    def test_info_is_at_least_info(self):
        assert AlertLevel.INFO.is_at_least(AlertLevel.INFO)
