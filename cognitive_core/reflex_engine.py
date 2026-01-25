"""
Cognitive Core — Reflex Engine (Layer 1)
Deterministic, <50ms, zero API calls. Provides IMMEDIATE preliminary alerts.

This is the safety net — it fires instantly while Gemini thinks.
"""

from __future__ import annotations

import time
import threading
from typing import Optional

from .models import AlertLevel, AudioEvent, ReflexAlert, VisionEvent


class ReflexEngine:
    """
    Bayesian-inspired scoring matrix for instant (<50ms) alert generation.

    Decision logic:
    - If BOTH vision and audio fire corroborating events → immediate alert
    - If vision fires alone → wait corroboration_window_sec for audio
    - If audio alone fires for a keyword ("help") → MEDIUM alert

    All alerts respect a per-type cooldown to prevent spam.
    """

    # -----------------------------------------------------------------------
    # Scoring Matrix: (vision_event_type, audio_event_type) → (AlertLevel, confidence, message)
    # -----------------------------------------------------------------------
    FUSION_MATRIX = {
        # --- Critical: Both sensors confirm a dangerous event ---
        ("fall",    "thud"):   (AlertLevel.CRITICAL, 0.95, "Fall detected with impact sound — high confidence incident"),
        ("fall",    "scream"): (AlertLevel.CRITICAL, 0.95, "Fall detected with patient scream — immediate response required"),
        ("fall",    "groan"):  (AlertLevel.CRITICAL, 0.92, "Fall detected with vocal distress — possible injury"),
        ("fall",    "help"):   (AlertLevel.CRITICAL, 0.97, "Fall detected — patient calling for help"),
        ("fall",    "madad"):  (AlertLevel.CRITICAL, 0.97, "Fall detected — patient calling for help (Urdu)"),
        ("seizure", "groan"):  (AlertLevel.CRITICAL, 0.92, "Seizure detected with vocal distress"),
        ("seizure", "scream"): (AlertLevel.CRITICAL, 0.93, "Seizure detected with distress sounds"),
        ("seizure", "help"):   (AlertLevel.CRITICAL, 0.94, "Seizure detected — patient or bystander calling for help"),

        # --- High: Vision alone (no corroboration yet) or weaker audio ---
        ("fall",    None):     (AlertLevel.HIGH,     0.70, "Fall detected — awaiting audio corroboration"),
        ("fall",    "gasp"):   (AlertLevel.HIGH,     0.80, "Fall detected with patient gasp"),
        ("fall",    "silence"): (AlertLevel.HIGH,    0.80, "Fall detected — patient silent (possible unconscious)"),
        ("seizure", None):     (AlertLevel.HIGH,     0.75, "Seizure detected — awaiting audio corroboration"),
        ("seizure", "gasp"):   (AlertLevel.HIGH,     0.82, "Seizure detected with patient gasp"),
        ("force_fall", None):  (AlertLevel.HIGH,     0.90, "Safety net: patient detected on floor zone"),

        # --- Medium: Non-critical events, proactive alerts ---
        ("restlessness", None): (AlertLevel.MEDIUM,  0.55, "Unusual sleep restlessness — monitor patient"),
        (None,           "help"):  (AlertLevel.MEDIUM, 0.65, "Patient calling for help — no visual event"),
        (None,           "madad"): (AlertLevel.MEDIUM, 0.65, "Patient calling for help (Urdu) — no visual event"),
        (None,           "scream"): (AlertLevel.MEDIUM, 0.70, "Distress scream heard — visual check inconclusive"),
        ("missing_patient", None): (AlertLevel.MEDIUM, 0.80, "Patient not visible for extended period"),
    }

    def __init__(self, config: dict):
        """
        Args:
            config: The 'cognitive.reflex' section of config.yaml.
        """
        self.corroboration_window = config.get("corroboration_window_sec", 2.0)
        self.cooldown_sec = config.get("cooldown_sec", 30)

        # Cooldown tracking: {alert_key: last_fire_timestamp}
        self._cooldowns: dict[str, float] = {}
        self._lock = threading.Lock()

    def evaluate(
        self,
        vision: Optional[VisionEvent],
        audio: Optional[AudioEvent],
    ) -> Optional[ReflexAlert]:
        """
        Evaluate the current sensor state and return a ReflexAlert, or None.

        Args:
            vision: Most recent significant VisionEvent (or None).
            audio:  Most recent AudioEvent (or None).

        Returns:
            ReflexAlert if an alert should fire, else None.
        """
        v_type = self._vision_type(vision)
        a_type = self._audio_type(audio)

        result = self._lookup(v_type, a_type)
        if result is None:
            return None

        level, confidence, message = result
        corroborated = v_type is not None and a_type is not None

        alert = ReflexAlert(
            level=level,
            confidence=confidence,
            message=message,
            corroborated=corroborated,
            vision_event=vision,
            audio_event=audio,
            trigger_reasoning=f"vision={v_type}, audio={a_type}, corroborated={corroborated}",
        )

        alert_key = f"{v_type}_{a_type}"
        with self._lock:
            if self._in_cooldown(alert_key):
                return None
            self._cooldowns[alert_key] = time.monotonic()

        return alert

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _lookup(
        self,
        v_type: Optional[str],
        a_type: Optional[str],
    ) -> Optional[tuple]:
        """
        Try to find a match in FUSION_MATRIX.
        Try exact match first, then fallback to (v_type, None) and (None, a_type).
        """
        # 1. Exact match (both sensors)
        if v_type and a_type:
            key = (v_type, a_type)
            if key in self.FUSION_MATRIX:
                return self.FUSION_MATRIX[key]
            # Fall back to trying a 'silence' pairing for audio events we don't recognise
            # (but still want to treat as corroborated)

        # 2. Vision-only match
        if v_type:
            key = (v_type, None)
            if key in self.FUSION_MATRIX:
                return self.FUSION_MATRIX[key]

        # 3. Audio-only match
        if a_type:
            key = (None, a_type)
            if key in self.FUSION_MATRIX:
                return self.FUSION_MATRIX[key]

        return None

    @staticmethod
    def _vision_type(event: Optional[VisionEvent]) -> Optional[str]:
        if event is None:
            return None
        t = event.event_type
        # Normalise variants to canonical types
        if t in ("fall", "force_fall", "seizure", "restlessness", "missing_patient"):
            return t
        return None  # 'normal', 'in_bed', 'darkness' → not significant for reflex

    @staticmethod
    def _audio_type(event: Optional[AudioEvent]) -> Optional[str]:
        if event is None:
            return None
        if event.is_keyword():
            return event.keyword  # 'help', 'nurse', 'madad'
        if event.is_distress():
            return event.sound_type  # 'thud', 'scream', 'groan', 'gasp'
        return None

    def _in_cooldown(self, alert_key: str) -> bool:
        last = self._cooldowns.get(alert_key, 0.0)
        return (time.monotonic() - last) < self.cooldown_sec

    def reset_cooldowns(self):
        """Clear all cooldowns — useful for testing."""
        with self._lock:
            self._cooldowns.clear()
