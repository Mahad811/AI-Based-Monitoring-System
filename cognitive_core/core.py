"""
Cognitive Core — Orchestrator (CognitiveCore)
The main integration class. Wires together the ReflexEngine and ReasoningEngine
into the "Reflex-then-Reason" pipeline.

Usage:
    core = CognitiveCore(config)
    pending_alert = core.process(vision_event, audio_event, frame)
    # pending_alert.reflex is available IMMEDIATELY
    # pending_alert.report is filled in ~1-2s later (async Gemini call)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Deque, List, Optional

from .models import (
    AlertLevel,
    AudioEvent,
    CognitiveCoreAlert,
    IncidentReport,
    ReflexAlert,
    VisionEvent,
)
from .reflex_engine import ReflexEngine
from .reasoning_engine import ReasoningEngine

log = logging.getLogger(__name__)


class CognitiveCore:
    """
    Central intelligence of Vital Guardian.

    Processing pipeline per frame/event:
    1. Accept VisionEvent + optional AudioEvent + current frame
    2. ReflexEngine → instant ReflexAlert (<50ms)
    3. If alert ≥ MEDIUM → submit async Gemini reasoning job
    4. Return CognitiveCoreAlert immediately (report fills in later)

    Caller registers an on_report_ready callback to receive Gemini results:
        core.on_report_ready = lambda alert: dashboard.update(alert)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Full config dict (from config.yaml). Uses 'cognitive' section.
        """
        cog_cfg = config.get("cognitive", {})
        reflex_cfg = cog_cfg.get("reflex", {})
        reasoning_cfg = cog_cfg.get("reasoning", {})
        log_cfg = cog_cfg.get("event_log", {})

        self.reflex   = ReflexEngine(reflex_cfg)
        self.reasoning = ReasoningEngine(reasoning_cfg)

        # Event log ring-buffer
        self._max_events = log_cfg.get("max_events", 100)
        self._vision_log: Deque[VisionEvent] = deque(maxlen=self._max_events)
        self._audio_log:  Deque[AudioEvent]  = deque(maxlen=self._max_events)

        # Gemini worker thread pool (1 worker — serial, prevents API flood)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gemini")

        # Last active alert (for corroboration / display)
        self._pending_alert: Optional[CognitiveCoreAlert] = None
        self._lock = threading.Lock()

        # Callback: called when Gemini finishes (on background thread)
        self.on_report_ready: Optional[Callable[[CognitiveCoreAlert], None]] = None

        log.info("[CognitiveCore] Initialized (Reflex + Gemini 3 Flash)")

    # -----------------------------------------------------------------------
    # Main entry point — called every frame / event
    # -----------------------------------------------------------------------

    def process(
        self,
        vision_event: Optional[VisionEvent] = None,
        audio_event:  Optional[AudioEvent]  = None,
        frame = None,
    ) -> Optional[CognitiveCoreAlert]:
        """
        Process one vision+audio observation.

        Args:
            vision_event: Latest VisionEvent from the vision pipeline (or None).
            audio_event:  Latest AudioEvent from the audio module (or None).
            frame:        Current camera frame (BGR numpy array). Given to Gemini.

        Returns:
            CognitiveCoreAlert if the reflex fired, else None.
            The .report field will be None at first, filled in ~1s later via callback.
        """
        # 1. Log events
        if vision_event:
            self._vision_log.append(vision_event)
        if audio_event:
            self._audio_log.append(audio_event)

        # 2. Run reflex — instant
        sig_vision = vision_event if (vision_event and vision_event.is_significant()) else None
        reflex_alert = self.reflex.evaluate(sig_vision, audio_event)

        if reflex_alert is None:
            return None

        # 3. Build the preliminary CognitiveCoreAlert
        alert = CognitiveCoreAlert(reflex=reflex_alert)

        with self._lock:
            self._pending_alert = alert

        # 4. If worth deeper analysis → submit Gemini job (async, non-blocking)
        if reflex_alert.is_actionable():
            alert.gemini_pending = True
            vision_snapshot = list(self._vision_log)
            audio_snapshot  = list(self._audio_log)
            self._executor.submit(
                self._run_gemini,
                alert, reflex_alert,
                vision_snapshot, audio_snapshot,
                frame.copy() if frame is not None else None,
            )

        return alert

    # -----------------------------------------------------------------------
    # Background Gemini worker
    # -----------------------------------------------------------------------

    def _run_gemini(
        self,
        alert: CognitiveCoreAlert,
        reflex_alert: ReflexAlert,
        vision_events: List[VisionEvent],
        audio_events:  List[AudioEvent],
        frame:          Optional[np.ndarray],
    ):
        """Runs in background thread. Calls Gemini, fills in alert.report."""
        try:
            report: IncidentReport = self.reasoning.analyse(
                reflex_alert, vision_events, audio_events, frame
            )
            alert.report = report
            alert.gemini_pending = False

            log.info(
                f"[CognitiveCore] Gemini decision: {report.final_decision.value} "
                f"→ {report.severity.value} | {report.headline}"
            )

            # Fire callback on the background thread
            if self.on_report_ready:
                try:
                    self.on_report_ready(alert)
                except Exception as cb_err:
                    log.warning(f"[CognitiveCore] on_report_ready callback error: {cb_err}")

        except Exception as exc:
            log.error(f"[CognitiveCore] Gemini background job failed: {exc}")
            alert.gemini_error = str(exc)
            alert.gemini_pending = False

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def latest_alert(self) -> Optional[CognitiveCoreAlert]:
        with self._lock:
            return self._pending_alert

    def shutdown(self):
        """Gracefully shut down the background thread pool."""
        log.info("[CognitiveCore] Shutting down...")
        self._executor.shutdown(wait=True)
