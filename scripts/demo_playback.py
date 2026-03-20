"""
Vital Guardian — Demo Playback Script
"Reflex-then-Reason" AI pipeline with keyboard audio injection.

Plays a video file through the full pipeline:
  Video ──► VisionPipeline ──► CognitiveCore ──► Console alerts
                                    ▲
  Keyboard ──► AudioSimulator ───────┘

Controls:
    T  → Thud (impact sound)       H  → "Help" keyword
    S  → Scream                    N  → "Nurse" keyword
    G  → Groan                     M  → "Madad" (Urdu)
    P  → Gasp                      X  → Silence
    Q  → Quit

Usage:
    python scripts/demo_playback.py --video path/to/video.mp4 [--mock]
"""

import argparse
import sys
import os
import time
import threading
from pathlib import Path

import cv2
import yaml

# Add project root to path so we can import all modules
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from visual_guardian.pipeline import VisionPipeline
from cognitive_core import (
    CognitiveCore,
    AudioSimulator,
    VisionEvent,
    AlertLevel,
    GeminiDecision,
    CognitiveCoreAlert,
)


# ---------------------------------------------------------------------------
# ANSI colors & formatting
# ---------------------------------------------------------------------------
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    ORANGE  = "\033[91m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

LEVEL_COLORS = {
    AlertLevel.INFO:     C.WHITE,
    AlertLevel.LOW:      C.GREEN,
    AlertLevel.MEDIUM:   C.YELLOW,
    AlertLevel.HIGH:     C.ORANGE,
    AlertLevel.CRITICAL: C.RED,
}

DECISION_ICONS = {
    GeminiDecision.CONFIRM:  "✅ CONFIRMED",
    GeminiDecision.ESCALATE: "⬆️  ESCALATED",
    GeminiDecision.SUPPRESS: "🔕 SUPPRESSED (False Alarm)",
    GeminiDecision.OVERRIDE: "🔄 OVERRIDDEN",
}


def print_reflex_alert(alert: CognitiveCoreAlert):
    """Print the instant reflex alert to console."""
    r = alert.reflex
    color = LEVEL_COLORS.get(r.level, C.WHITE)
    cb = "⚡ CORROBORATED" if r.corroborated else "⏳ UNCORROBORATED"
    print(f"\n{C.BOLD}{color}{'═'*60}{C.RESET}")
    print(f"{C.BOLD}{color}  ⚡ REFLEX ALERT [{r.level.value}] — {cb}{C.RESET}")
    print(f"{color}  {r.message}{C.RESET}")
    print(f"{C.DIM}  Confidence: {r.confidence:.0%} | {r.trigger_reasoning}{C.RESET}")
    print(f"{color}{'─'*60}{C.RESET}")
    print(f"{C.DIM}  🧠 Gemini 3 Flash is analysing...{C.RESET}")


def print_gemini_report(alert: CognitiveCoreAlert):
    """Print Gemini's final decision — called from background thread via callback."""
    rep = alert.report
    if rep is None:
        return

    color = LEVEL_COLORS.get(rep.severity, C.WHITE)
    decision_str = DECISION_ICONS.get(rep.final_decision, str(rep.final_decision))

    print(f"\n{C.BOLD}{color}{'═'*60}{C.RESET}")
    print(f"{C.BOLD}{color}  🧠 GEMINI DECISION: {decision_str}{C.RESET}")
    print(f"{C.BOLD}{color}  [{rep.severity.value}] {rep.headline}{C.RESET}")
    print(f"{color}{'─'*60}{C.RESET}")
    print(f"{C.WHITE}  {rep.narrative}{C.RESET}")
    print(f"\n{C.CYAN}  Recommended Actions:{C.RESET}")
    for i, action in enumerate(rep.recommended_actions, 1):
        print(f"    {i}. {action}")
    print(f"\n{C.DIM}  Reasoning: {rep.reasoning}{C.RESET}")
    print(f"{C.DIM}  Confidence: {rep.confidence:.0%}{C.RESET}")
    print(f"{color}{'═'*60}{C.RESET}\n")


def overlay_alert_on_frame(frame, alert: CognitiveCoreAlert, audio_label: str | None):
    """Draw alert overlay on the video frame."""
    h, w = frame.shape[:2]
    r = alert.reflex

    # Alert level bar at top
    level_colors_bgr = {
        AlertLevel.INFO:     (200, 200, 200),
        AlertLevel.LOW:      (0, 200, 0),
        AlertLevel.MEDIUM:   (0, 200, 200),
        AlertLevel.HIGH:     (0, 100, 255),
        AlertLevel.CRITICAL: (0, 0, 255),
    }
    bar_color = level_colors_bgr.get(alert.final_level, (200, 200, 200))
    cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)

    if alert.report:
        rep = alert.report
        label = f"[{rep.severity.value}] {rep.final_decision.value}: {rep.headline[:50]}"
        conf_label = f"Confidence: {rep.confidence:.0%}"
    else:
        label = f"[{r.level.value}] {r.message[:55]} | Gemini thinking..."
        conf_label = f"Reflex: {r.confidence:.0%}"

    cv2.putText(frame, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    cv2.putText(frame, conf_label, (w - 200, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Audio injection label
    if audio_label:
        cv2.rectangle(frame, (0, h - 35), (300, h), (50, 50, 50), -1)
        cv2.putText(frame, f"Audio: {audio_label}", (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)


def run_demo(video_path: str, config: dict, mock_mode: bool):
    """Main demo loop."""
    if mock_mode:
        config["cognitive"]["reasoning"]["mock_mode"] = True
        print(f"{C.YELLOW}  ⚠ MOCK MODE — No Gemini API calls. Using pre-written responses.{C.RESET}")

    print(f"\n{C.CYAN}{C.BOLD}  Vital Guardian — Cognitive Core V2 Demo{C.RESET}")
    print(f"{C.DIM}  Model: gemini-3-flash-preview | Mode: Reflex-then-Reason{C.RESET}")
    print(f"{C.DIM}  {AudioSimulator.get_help_text()}{C.RESET}")
    print(f"{C.DIM}  Q = Quit{C.RESET}\n")

    # Initialize pipeline components
    vision = VisionPipeline(config)
    vision.load_models()
    print(f"{C.GREEN}  ✓ VisionPipeline loaded{C.RESET}")

    core = CognitiveCore(config)
    audio_sim = AudioSimulator(event_duration_sec=3.0)

    # Set the callback to print Gemini reports as they arrive
    core.on_report_ready = print_gemini_report
    print(f"{C.GREEN}  ✓ CognitiveCore initialized{C.RESET}\n")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"{C.RED}  ✗ Could not open video: {video_path}{C.RESET}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_delay = int(1000 / fps)

    current_alert = None
    audio_label_display = None
    audio_label_expiry = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # Loop video for demo
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Handle keyboard
            key = cv2.waitKey(frame_delay) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

            # Audio injection
            audio_event = None
            label = audio_sim.handle_keypress(key)
            if label:
                audio_label_display = label
                audio_label_expiry = time.monotonic() + 2.0

            audio_event = audio_sim.get_current_event()
            if time.monotonic() > audio_label_expiry:
                audio_label_display = None

            # Vision event from pipeline
            raw_event = vision.process_frame(frame)
            vision_event = VisionEvent.from_pipeline_event(raw_event)

            # Feed into Cognitive Core
            alert = core.process(vision_event, audio_event, frame)
            if alert:
                current_alert = alert
                print_reflex_alert(alert)

            # Draw overlay
            display_frame = frame.copy()
            if current_alert:
                overlay_alert_on_frame(display_frame, current_alert, audio_label_display)

            cv2.imshow("Vital Guardian — Cognitive Core V2", display_frame)

    finally:
        cap.release()
        core.shutdown()
        cv2.destroyAllWindows()
        print(f"\n{C.CYAN}  Demo ended.{C.RESET}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Vital Guardian Cognitive Core Demo")
    parser.add_argument("--video", required=True, help="Path to demo video file")
    parser.add_argument(
        "--mock", action="store_true",
        help="Run in mock mode (no Gemini API key required)"
    )
    parser.add_argument(
        "--config", default=str(ROOT / "config" / "config.yaml"),
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    if not Path(args.video).exists():
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_demo(args.video, config, args.mock)


if __name__ == "__main__":
    main()
