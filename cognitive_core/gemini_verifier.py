"""
Vital Guardian — Gemini 3 Flash Cognitive Core

Three-tier architecture:
  Tier 2 — Fast binary verify   (~1-2s)  : CONFIRMED / SUPPRESSED + 1-line reason
  Tier 3 — Clinical enrichment  (~4-8s)  : severity, narrative, actions (only if T2 = CONFIRMED)
  Risk    — Proactive monitor   (async)  : ambient 30s background scan, no alert required

All Gemini calls are synchronous and designed to be run inside asyncio.to_thread()
so they never block the main async event loop.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List

try:
    from google import genai
    from google.genai import types
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL = "gemini-3-flash-preview"


# ── Prompts ────────────────────────────────────────────────────────────────────

_TIER2_SYSTEM = (
    "You are a clinical AI performing a rapid alert audit. "
    "You will receive a short sequence of patient room frames. "
    "Your ONLY job is a fast binary decision: is this alert real or a false positive? "
    "Respond exclusively with valid JSON. No prose, no markdown, no explanation outside the JSON."
)

_TIER2_PROMPT = """\
Alert audit.
Patient: {patient_id}
Event  : {event_type} — ML model confidence {confidence:.0%}

You are reviewing {n_frames} frames spanning ~2 seconds around the trigger.

Answer ONE question: is this alert clinically real or a false positive?

Consider ONLY:
- Is body position consistent with {event_type}?
- Does the motion match {event_type} biomechanics?
- Is there an obvious benign explanation?

JSON output (no other text):
{{"decision": "CONFIRMED" or "SUPPRESSED", "reason": "one sentence max"}}
"""

_TIER3_SYSTEM = (
    "You are a consultant clinical AI providing enrichment context for a CONFIRMED alert. "
    "The nursing team has already been notified. Your job is clinical detail — NOT re-verification. "
    "Use structured, precise medical language. "
    "Never mention frames, cameras, images, or video. Treat the data as sensor input. "
    "Respond exclusively with valid JSON."
)

_TIER3_PROMPT = """\
Clinical enrichment — CONFIRMED {event_type} alert.
Patient  : {patient_id}
Confidence: {confidence:.0%}
Frames   : {n_frames} (spanning {window_s}s — pre-event through aftermath)

Analyse in order:

1. BODY POSITION — describe orientation, vertical height change, limb disposition in first vs last frame.
2. MOTION PATTERN — rapid/gradual? directional/oscillatory? controlled/uncontrolled?
3. CLINICAL SIGNIFICANCE — what makes this clinically significant? complicating factors?
4. SEVERITY — estimate severity and immediate risk.
5. ACTIONS — 2-3 immediate nursing actions, ordered by priority.

JSON output (no other text):
{{
  "headline"     : "under 10 words for dashboard display",
  "narrative"    : "1-2 sentences — clinical description of what occurred",
  "severity"     : "low" | "moderate" | "high" | "critical",
  "body_analysis": "step 1 findings",
  "motion_analysis": "step 2 findings",
  "clinical_notes": "step 3 findings",
  "actions"      : ["action 1", "action 2", "action 3"],
  "escalate"     : true | false,
  "gemini_confidence": 0.0
}}
"""

_RISK_SYSTEM = (
    "You are performing a routine patient safety background scan. "
    "No alert has fired. This is a proactive ambient assessment. "
    "Be conservative — only flag elevated risk if genuinely unusual behaviour is present. "
    "A resting patient should return low risk scores. "
    "Respond exclusively with valid JSON."
)

_RISK_PROMPT = """\
Routine safety assessment.
Patient: {patient_id}
Frames : {n_frames} sampled evenly from the last 10 seconds.

Assess:
- CURRENT STATE: calm / restless / sleeping / agitated?
- POSITION RISK: near bed edge? sitting up unexpectedly?
- MOVEMENT PATTERN: any repetitive, tremor, or rigidity?
- PRE-EVENT SIGNS: instability, reaching, unusual posture before fall? pre-ictal restlessness?

JSON output (no other text):
{{
  "patient_state"   : "stable" | "restless" | "concerning" | "critical",
  "fall_risk"       : 0.0,
  "seizure_risk"    : 0.0,
  "observations"    : "what you observe",
  "advisory"        : "one sentence for staff if risk > 0.6, else empty string",
  "recommend_check" : true | false
}}
"""


# ── Helpers ────────────────────────────────────────────────────────────────────

def _encode_frame(frame: np.ndarray) -> Optional[bytes]:
    """Convert BGR OpenCV frame → JPEG bytes."""
    if frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return buf.tobytes() if ok else None


def _sample_frames(frames: list, n: int) -> list:
    """Evenly sample n frames from a list (or fewer if list is shorter)."""
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def _frames_to_parts(frames: list, n: int) -> list:
    """Convert a list of OpenCV frames to exactly n Gemini Part objects."""
    sampled = _sample_frames(frames, n)
    parts = []
    for f in sampled:
        data = _encode_frame(f)
        if data:
            parts.append(types.Part.from_bytes(data=data, mime_type="image/jpeg"))
    return parts


# ── GeminiVerifier ─────────────────────────────────────────────────────────────

class GeminiVerifier:
    """
    Wraps the Gemini 3 Flash API across three operational tiers.

    All public methods are *synchronous* — call them inside asyncio.to_thread().
    """

    def __init__(self, mock_mode: bool = False):
        self.mock_mode  = mock_mode
        self.client     = None
        self.model_name = MODEL

        if not self.mock_mode and HAS_GENAI:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                log.warning("GEMINI_API_KEY not set — running in MOCK mode.")
                self.mock_mode = True
            else:
                try:
                    self.client = genai.Client(api_key=api_key)
                    log.info("Gemini 3 Flash Cognitive Core initialised (%s).", MODEL)
                except Exception as exc:
                    log.error("Gemini init failed: %s", exc)
                    self.mock_mode = True

    # ── internal call helper ───────────────────────────────────────────────────

    def _call(self,
              system_instruction: str,
              parts: list,
              prompt_text: str,
              temperature: float = 0.0,
              thinking: bool = False) -> dict:
        """
        Make a single Gemini API call and return parsed JSON dict.
        Raises on network/parse error — callers must handle.
        """
        config_kwargs = dict(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            temperature=temperature,
        )
        if thinking:
            # Gemini 3 Flash: thinking_level drives internal chain-of-thought
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=1024          # enough depth without over-spending tokens
            )

        cfg = types.GenerateContentConfig(**config_kwargs)

        all_parts = parts + [types.Part.from_text(text=prompt_text)]

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=all_parts,
            config=cfg,
        )
        return json.loads(response.text)

    # ── Tier 2 — Fast Binary Verify ────────────────────────────────────────────

    def verify_binary(self,
                      event_type: str,
                      confidence: float,
                      patient_id: str,
                      frames: list) -> Dict[str, Any]:
        """
        Tier 2: fast binary decision.
        Input  : 4 frames, short prompt, 0 temperature.
        Output : {"decision": "CONFIRMED"|"SUPPRESSED", "reason": str}
        Target latency: 1-2 seconds.
        """
        if self.mock_mode or not self.client:
            return self._mock_tier2(event_type, confidence)

        parts  = _frames_to_parts(frames, n=4)
        prompt = _TIER2_PROMPT.format(
            patient_id=patient_id,
            event_type=event_type,
            confidence=confidence,
            n_frames=len(parts),
        )

        try:
            result = self._call(
                system_instruction=_TIER2_SYSTEM,
                parts=parts,
                prompt_text=prompt,
                temperature=0.0,
                thinking=False,
            )
            # Normalise — ensure both keys always present
            result.setdefault("decision", "CONFIRMED")
            result.setdefault("reason",   "No reason provided.")
            return result
        except Exception as exc:
            log.error("[Tier2] API call failed: %s", exc)
            return self._mock_tier2(event_type, confidence, error=str(exc))

    # ── Tier 3 — Full Clinical Enrichment ─────────────────────────────────────

    def enrich_clinical(self,
                        event_type: str,
                        confidence: float,
                        patient_id: str,
                        frames: list,
                        window_s: float = 5.0) -> Dict[str, Any]:
        """
        Tier 3: full clinical enrichment. Runs ONLY after Tier 2 CONFIRMED.
        Input  : 12 frames, thinking enabled, extended temporal window.
        Output : full clinical report dict (headline, narrative, severity, actions, …)
        Target latency: 4-8 seconds.
        """
        if self.mock_mode or not self.client:
            return self._mock_tier3(event_type, confidence)

        parts  = _frames_to_parts(frames, n=12)
        prompt = _TIER3_PROMPT.format(
            patient_id=patient_id,
            event_type=event_type,
            confidence=confidence,
            n_frames=len(parts),
            window_s=f"{window_s:.0f}",
        )

        try:
            result = self._call(
                system_instruction=_TIER3_SYSTEM,
                parts=parts,
                prompt_text=prompt,
                temperature=0.1,    # small variance allowed for richer language
                thinking=True,      # Gemini 3 Flash chain-of-thought
            )
            # Normalise optional fields
            result.setdefault("headline",          f"{event_type.title()} Detected")
            result.setdefault("narrative",         "Clinical analysis complete.")
            result.setdefault("severity",          "moderate")
            result.setdefault("body_analysis",     "")
            result.setdefault("motion_analysis",   "")
            result.setdefault("clinical_notes",    "")
            result.setdefault("actions",           ["Respond immediately", "Assess patient"])
            result.setdefault("escalate",          False)
            result.setdefault("gemini_confidence", confidence)
            return result
        except Exception as exc:
            log.error("[Tier3] API call failed: %s", exc)
            return self._mock_tier3(event_type, confidence, error=str(exc))

    # ── Risk Monitor — Proactive Background Assessment ─────────────────────────

    def assess_risk(self,
                    patient_id: str,
                    frames: list) -> Dict[str, Any]:
        """
        Proactive risk monitor. No alert required — runs on a timer.
        Input  : 8 frames from the last 10 seconds.
        Output : {"patient_state", "fall_risk", "seizure_risk",
                  "observations", "advisory", "recommend_check"}
        """
        if self.mock_mode or not self.client:
            return self._mock_risk()

        parts  = _frames_to_parts(frames, n=8)
        prompt = _RISK_PROMPT.format(
            patient_id=patient_id,
            n_frames=len(parts),
        )

        try:
            result = self._call(
                system_instruction=_RISK_SYSTEM,
                parts=parts,
                prompt_text=prompt,
                temperature=0.0,
                thinking=False,
            )
            result.setdefault("patient_state",    "stable")
            result.setdefault("fall_risk",        0.0)
            result.setdefault("seizure_risk",     0.0)
            result.setdefault("observations",     "")
            result.setdefault("advisory",         "")
            result.setdefault("recommend_check",  False)
            return result
        except Exception as exc:
            log.error("[Risk] API call failed: %s", exc)
            return self._mock_risk()

    # ── Legacy shim — keeps demo_server's old call-site working ───────────────

    def verify_alert(self,
                     event_type: str,
                     confidence: float,
                     patient_id: str,
                     frames: list) -> Dict[str, Any]:
        """
        DEPRECATED SHIM — preserved for backwards compatibility.
        New code should call verify_binary() then enrich_clinical() separately.
        """
        t2 = self.verify_binary(event_type, confidence, patient_id, frames)
        if t2.get("decision") != "CONFIRMED":
            return {
                "decision":  "SUPPRESSED",
                "headline":  "Alert Suppressed",
                "narrative": t2.get("reason", "Event not confirmed by visual analysis."),
                "actions":   ["Continue routine monitoring"],
            }
        t3 = self.enrich_clinical(event_type, confidence, patient_id, frames)
        t3["decision"] = "CONFIRMED"
        return t3

    # ── Mock fallbacks ─────────────────────────────────────────────────────────

    def _mock_tier2(self, event_type, confidence, error="") -> dict:
        return {
            "decision": "CONFIRMED",
            "reason":   f"Mock: ML confidence {confidence:.0%} accepted." + (f" [{error}]" if error else ""),
        }

    def _mock_tier3(self, event_type, confidence, error="") -> dict:
        return {
            "headline":          f"{event_type.title()} Confirmed",
            "narrative":         f"Sensor data confirms a {event_type} event at {confidence:.0%} confidence.",
            "severity":          "moderate",
            "body_analysis":     "Mock — Gemini API unavailable.",
            "motion_analysis":   "Mock — Gemini API unavailable.",
            "clinical_notes":    f"Error: {error}" if error else "",
            "actions":           ["Respond immediately", "Assess patient vitals", "Document incident"],
            "escalate":          confidence > 0.80,
            "gemini_confidence": confidence,
        }

    def _mock_risk(self) -> dict:
        return {
            "patient_state":   "stable",
            "fall_risk":       0.1,
            "seizure_risk":    0.05,
            "observations":    "Mock — Gemini API unavailable.",
            "advisory":        "",
            "recommend_check": False,
        }
