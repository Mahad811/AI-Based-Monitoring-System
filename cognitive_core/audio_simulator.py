"""
Cognitive Core — Audio Simulator
Keyboard-driven AudioEvent injector for demos and testing.

During a live demo, pressing these keys injects realistic AudioEvents
into the CognitiveCore pipeline — no microphone required.

Key mappings:
    T  → Thud / Impact (distress)
    S  → Scream (distress)
    G  → Groan / Moan (distress)
    P  → Gasp (distress)
    H  → "Help" keyword (English)
    N  → "Nurse" keyword (English)
    M  → "Madad" keyword (Urdu — 'help')
    X  → Silence (clears current audio state)
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional

from .models import AudioEvent

log = logging.getLogger(__name__)


class AudioSimulator:
    """
    Non-blocking keyboard audio injector.

    Maintains the 'last injected event' so the CognitiveCore can poll it each frame.
    Events expire after `event_duration_sec` to simulate transient sounds.
    """

    # Key → (event_type, sound_type, keyword, language)
    KEY_MAP = {
        ord('t'): ("distress", "thud",   None,    None),
        ord('T'): ("distress", "thud",   None,    None),
        ord('s'): ("distress", "scream", None,    None),
        ord('S'): ("distress", "scream", None,    None),
        ord('g'): ("distress", "groan",  None,    None),
        ord('G'): ("distress", "groan",  None,    None),
        ord('p'): ("distress", "gasp",   None,    None),
        ord('P'): ("distress", "gasp",   None,    None),
        ord('h'): ("keyword",  None,     "help",  "english"),
        ord('H'): ("keyword",  None,     "help",  "english"),
        ord('n'): ("keyword",  None,     "nurse", "english"),
        ord('N'): ("keyword",  None,     "nurse", "english"),
        ord('m'): ("keyword",  None,     "madad", "urdu"),
        ord('M'): ("keyword",  None,     "madad", "urdu"),
        ord('x'): None,   # Silence / clear
        ord('X'): None,
    }

    LABELS = {
        ord('t'): "🔊 THUD",
        ord('s'): "🔊 SCREAM",
        ord('g'): "🔊 GROAN",
        ord('p'): "🔊 GASP",
        ord('h'): "🗣️  HELP (EN)",
        ord('n'): "🗣️  NURSE",
        ord('m'): "🗣️  MADAD (UR)",
        ord('x'): "🔇 SILENCE",
    }

    def __init__(self, event_duration_sec: float = 3.0):
        """
        Args:
            event_duration_sec: How long an injected event stays 'active' before
                                auto-expiring. Simulates a transient audio event.
        """
        self.event_duration_sec = event_duration_sec
        self._current_event: Optional[AudioEvent] = None
        self._inject_time: float = 0.0
        self._lock = threading.Lock()

    def handle_keypress(self, key: int) -> Optional[str]:
        """
        Call this with a raw OpenCV waitKey() result.

        Args:
            key: Integer key code from cv2.waitKey().

        Returns:
            Human-readable label string if a key was handled, else None.
        """
        key_lower = key | 0x20 if 65 <= key <= 90 else key  # normalise
        mapping = self.KEY_MAP.get(key)
        if mapping is None and key not in self.KEY_MAP:
            return None  # Not an audio key

        with self._lock:
            if mapping is None:
                # Silence / clear
                self._current_event = None
                label = "🔇 SILENCE"
            else:
                event_type, sound_type, keyword, language = mapping
                self._current_event = AudioEvent(
                    event_type=event_type,
                    sound_type=sound_type,
                    keyword=keyword,
                    language=language,
                    confidence=0.92,
                )
                self._inject_time = time.monotonic()
                label = self.LABELS.get(key, self.LABELS.get(key | 0x20, "🔊 AUDIO"))

        log.info(f"[AudioSimulator] Injected: {label}")
        return label

    def get_current_event(self) -> Optional[AudioEvent]:
        """
        Return the current active AudioEvent, or None if expired/silence.
        Call once per frame from the main loop.
        """
        with self._lock:
            if self._current_event is None:
                return None
            elapsed = time.monotonic() - self._inject_time
            if elapsed > self.event_duration_sec:
                self._current_event = None
                return None
            return self._current_event

    def reset(self):
        """Clear any injected event."""
        with self._lock:
            self._current_event = None

    @staticmethod
    def get_help_text() -> str:
        return (
            "Audio Keys: T=Thud  S=Scream  G=Groan  P=Gasp  "
            "H=Help  N=Nurse  M=Madad(Ur)  X=Silence"
        )
