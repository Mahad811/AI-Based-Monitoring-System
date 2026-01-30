"""
Cognitive Core — Reasoning Engine V2 (Layer 2)
The actual AI brain: Gemini 3 Flash with multimodal input and structured output.

Gemini receives:
  - The live video frame (JPEG image)
  - Structured event log (last N seconds)
  - The ReflexEngine's preliminary decision

Gemini makes the FINAL decision: CONFIRM / ESCALATE / SUPPRESS / OVERRIDE
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import List, Optional

from .models import (
    AlertLevel,
    AudioEvent,
    GeminiDecision,
    IncidentReport,
    ReflexAlert,
    VisionEvent,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock response — used when mock_mode=True or API is unavailable
# ---------------------------------------------------------------------------
_MOCK_RESPONSES = {
    AlertLevel.CRITICAL: IncidentReport(
        final_decision=GeminiDecision.CONFIRM,
        severity=AlertLevel.CRITICAL,
        headline="Fall Confirmed — Patient on Floor with Impact Sound",
        narrative=(
            "Patient transitioned from upright to horizontal position at high velocity "
            "consistent with an uncontrolled fall. Audio channel confirmed impact sound "
            "immediately following visual detection. Patient is unresponsive to environment. "
            "Immediate bedside assessment is required to rule out head trauma."
        ),
        recommended_actions=[
            "Respond to room immediately",
            "Do not move patient — assess for spinal injury",
            "Check consciousness and airway",
            "Call code if patient unresponsive",
        ],
        reasoning=(
            "Vision confidence 0.85+ for fall event, corroborated by audio impact sound. "
            "Frame shows patient supine on floor outside expected bed zone."
        ),
        confidence=0.94,
    ),
    AlertLevel.HIGH: IncidentReport(
        final_decision=GeminiDecision.CONFIRM,
        severity=AlertLevel.HIGH,
        headline="Possible Fall — Visual Detection, Awaiting Confirmation",
        narrative=(
            "Vision pipeline has flagged a significant position change consistent with a fall. "
            "No audio corroboration received at this time. Patient may have fallen silently "
            "or the impact was below the audio detection threshold. Visual check recommended."
        ),
        recommended_actions=[
            "Visual check via camera or in-person",
            "Verify patient location and status",
            "Document event for review",
        ],
        reasoning=(
            "Visual model confidence above threshold. No corroborating audio. "
            "Classifying as probable fall pending human confirmation."
        ),
        confidence=0.75,
    ),
    AlertLevel.MEDIUM: IncidentReport(
        final_decision=GeminiDecision.CONFIRM,
        severity=AlertLevel.MEDIUM,
        headline="Patient Alert — Attention Required",
        narrative=(
            "Sensor event detected requiring nurse awareness. "
            "Patient may be calling for assistance or showing signs of distress. "
            "No immediate life-threatening event confirmed."
        ),
        recommended_actions=[
            "Check patient status",
            "Respond if patient called for nurse",
        ],
        reasoning="Medium-priority event from sensor fusion. No critical indicators.",
        confidence=0.65,
    ),
}

# ---------------------------------------------------------------------------
# System prompt — the core instruction that makes Gemini the decision brain
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are the central AI intelligence of "Vital Guardian", a real-time ICU patient monitoring system.

You receive a live camera frame and structured sensor data from vision and audio modules.
A deterministic Reflex Engine has already issued a preliminary alert — your job is to
make the FINAL decision by carefully analyzing the frame and data.

YOUR DECISION OPTIONS:
- CONFIRM: Validate the reflex alert. You see evidence of the event.
- ESCALATE: The situation is MORE serious than the reflex detected. Raise the severity.
- SUPPRESS: This is a false alarm. The camera frame does NOT support the alert. Override it.
- OVERRIDE: Change the alert level (higher or lower) based on your analysis.

RULES:
- Never diagnose. Describe only what you observe in the frame and data.
- Be concise and factual. Nursing staff need actionable information.
- Recommended actions must be ordered by urgency.
- Your reasoning field should briefly explain WHAT in the frame led to your decision.
- Confidence is your certainty in your own final_decision (0.0–1.0).
- If the frame shows normal patient position but vision model flagged a fall → SUPPRESS.
- If the frame shows a person on the floor → CONFIRM or ESCALATE.
"""


class ReasoningEngine:
    """
    Gemini 3 Flash — the actual decision-making brain of Vital Guardian.

    Usage:
        engine = ReasoningEngine(config)
        report = await engine.analyse(reflex_alert, vision_events, audio_events, frame)
    """

    def __init__(self, config: dict):
        """
        Args:
            config: The 'cognitive.reasoning' section from config.yaml.
        """
        self.model_name = config.get("model", "gemini-3-flash-preview")
        self.thinking_level = config.get("thinking_level", "low")
        self.media_resolution = config.get("media_resolution", "medium")
        self.max_output_tokens = config.get("max_output_tokens", 400)
        self.temperature = config.get("temperature", 0.2)
        self.mock_mode = config.get("mock_mode", False)

        self._client = None
        self._initialized = False

    # -----------------------------------------------------------------------
    # Initialisation (lazy — only runs once when first needed)
    # -----------------------------------------------------------------------

    def _ensure_initialized(self) -> bool:
        """Lazy initialization of the Gemini client. Returns True if ready."""
        if self._initialized:
            return self._client is not None
        self._initialized = True

        if self.mock_mode:
            log.info("[ReasoningEngine] Mock mode enabled — no API calls will be made.")
            return False

        api_key = os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            log.warning("[ReasoningEngine] GEMINI_API_KEY not found in environment. "
                        "Falling back to mock mode.")
            self.mock_mode = True
            return False

        try:
            from google import genai
            from google.genai import types as genai_types

            self._client = genai.Client(api_key=api_key)
            self._genai_types = genai_types
            log.info(f"[ReasoningEngine] Initialized with model: {self.model_name}")
            return True
        except ImportError:
            log.error("[ReasoningEngine] google-genai not installed. "
                      "Run: pip install google-genai")
            self.mock_mode = True
            return False
        except Exception as exc:
            log.error(f"[ReasoningEngine] Initialization failed: {exc}")
            self.mock_mode = True
            return False

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def analyse(
        self,
        reflex_alert: ReflexAlert,
        vision_events: List[VisionEvent],
        audio_events: List[AudioEvent],
        frame: Optional[np.ndarray] = None,
    ) -> IncidentReport:
        """
        Send sensor data + frame to Gemini for multimodal analysis.
        Returns a structured IncidentReport with the final decision.

        This is synchronous — call from a thread pool executor in CognitiveCore.

        Args:
            reflex_alert:  The ReflexEngine's preliminary alert.
            vision_events: Recent vision events (last N seconds).
            audio_events:  Recent audio events (last N seconds).
            frame:         Current video frame as BGR numpy array (optional but recommended).

        Returns:
            IncidentReport with Gemini's final decision.
        """
        ready = self._ensure_initialized()

        if not ready or self.mock_mode:
            return self._mock_response(reflex_alert)

        try:
            prompt = self._build_prompt(reflex_alert, vision_events, audio_events)
            contents = self._build_contents(prompt, frame)
            return self._call_gemini(contents)
        except Exception as exc:
            log.error(f"[ReasoningEngine] Gemini call failed: {exc}")
            return self._fallback_from_reflex(reflex_alert, str(exc))

    # -----------------------------------------------------------------------
    # Prompt construction
    # -----------------------------------------------------------------------

    def _build_prompt(
        self,
        reflex_alert: ReflexAlert,
        vision_events: List[VisionEvent],
        audio_events: List[AudioEvent],
    ) -> str:
        """Build the structured text prompt."""
        lines = [
            "=== VITAL GUARDIAN SENSOR REPORT ===",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- REFLEX ENGINE PRELIMINARY ALERT ---",
            f"  Level      : {reflex_alert.level.value}",
            f"  Confidence : {reflex_alert.confidence:.2f}",
            f"  Message    : {reflex_alert.message}",
            f"  Corroborated (vision+audio): {reflex_alert.corroborated}",
            "",
            "--- VISION MODULE EVENTS (recent) ---",
        ]

        if vision_events:
            for ve in vision_events[-5:]:  # Last 5 events
                lines.append(
                    f"  [{ve.timestamp[-8:]}] type={ve.event_type} "
                    f"fall_conf={ve.fall_confidence:.2f} "
                    f"seizure_conf={ve.seizure_confidence:.2f} "
                    f"state={ve.state}"
                )
        else:
            lines.append("  (no recent vision events)")

        lines += ["", "--- AUDIO MODULE EVENTS (recent) ---"]

        if audio_events:
            for ae in audio_events[-5:]:
                lines.append(
                    f"  [{ae.timestamp[-8:]}] type={ae.event_type} "
                    f"sound={ae.sound_type} keyword={ae.keyword} "
                    f"conf={ae.confidence:.2f}"
                )
        else:
            lines.append("  (no recent audio events)")

        lines += [
            "",
            "--- CAMERA FRAME ---",
            "  (attached as image above — analyze the patient's position carefully)",
            "",
            "=== YOUR TASK ===",
            "Analyze the frame and sensor data. Issue your final CONFIRM/ESCALATE/SUPPRESS/OVERRIDE decision.",
            "Return structured JSON according to the IncidentReport schema.",
        ]

        return "\n".join(lines)

    def _build_contents(self, prompt: str, frame) -> list:
        """Build the multimodal contents list for the Gemini API."""
        contents = []

        # Attach the camera frame as an inline image
        if frame is not None:
            jpeg_bytes = self._frame_to_jpeg(frame)
            if jpeg_bytes:
                image_part = self._genai_types.Part.from_bytes(
                    data=jpeg_bytes,
                    mime_type="image/jpeg",
                )
                contents.append(image_part)

        # Append the structured text prompt
        contents.append(self._genai_types.Part.from_text(text=prompt))
        return contents

    @staticmethod
    def _frame_to_jpeg(frame, quality: int = 80):
        """Convert BGR OpenCV frame to JPEG bytes."""
        try:
            import cv2  # lazy import — only needed when frames are used
            success, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            )
            if success:
                return buf.tobytes()
        except Exception as exc:
            log.warning(f"[ReasoningEngine] Frame encoding failed: {exc}")
        return None

    # -----------------------------------------------------------------------
    # Gemini API call
    # -----------------------------------------------------------------------

    def _call_gemini(self, contents: list) -> IncidentReport:
        """Make the actual API call with structured output."""
        from google.genai import types as genai_types

        config = genai_types.GenerateContentConfig(
            system_instruction=_SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=IncidentReport,
            temperature=self.temperature,
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        # Pydantic auto-validates the structured response
        parsed = IncidentReport.model_validate_json(response.text)
        log.info(
            f"[ReasoningEngine] Gemini decision: {parsed.final_decision.value} "
            f"| severity: {parsed.severity.value} "
            f"| confidence: {parsed.confidence:.2f}"
        )
        return parsed

    # -----------------------------------------------------------------------
    # Fallbacks
    # -----------------------------------------------------------------------

    def _mock_response(self, reflex_alert: ReflexAlert) -> IncidentReport:
        """Return a pre-written realistic response based on reflex level."""
        level = reflex_alert.level
        # Round down to nearest available mock level
        for target in [AlertLevel.CRITICAL, AlertLevel.HIGH, AlertLevel.MEDIUM]:
            if level.is_at_least(target):
                return _MOCK_RESPONSES[target]
        return _MOCK_RESPONSES[AlertLevel.MEDIUM]

    @staticmethod
    def _fallback_from_reflex(
        reflex_alert: ReflexAlert, error_msg: str
    ) -> IncidentReport:
        """Generate a minimal report from the reflex alert if Gemini fails."""
        log.warning(f"[ReasoningEngine] Using reflex fallback. Error: {error_msg}")
        return IncidentReport(
            final_decision=GeminiDecision.CONFIRM,
            severity=reflex_alert.level,
            headline=reflex_alert.message,
            narrative=(
                f"{reflex_alert.message} (Note: AI reasoning engine temporarily unavailable. "
                "Alert based on deterministic sensor fusion.)"
            ),
            recommended_actions=["Assess patient immediately", "Check monitoring system"],
            reasoning=f"Fallback: API error — {error_msg[:100]}",
            confidence=reflex_alert.confidence * 0.8,
        )
