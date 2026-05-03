# Vital Guardian Technical Documentation

## 1. System Overview

Vital Guardian is a real-time ICU monitoring system with four runtime layers:

1. **Visual Guardian** (`visual_guardian/`): person detection + fall/seizure event generation.
2. **Auditory Watchdog** (`auditory_watchdog/`): distress and keyword detection from live or injected clip audio.
3. **Cognitive Core** (`cognitive_core/`): Gemini-based Tier-2/Tier-3 verification and enrichment.
4. **Web Runtime** (`scripts/demo/demo_server.py`): FastAPI + WebSocket dashboard and DB persistence.

## 2. Runtime Modes

### 2.1 Inference Modes

- `INFERENCE_MODE=LOCAL`
  - fall/seizure models loaded locally from configured model paths
- `INFERENCE_MODE=KAGGLE`
  - fall/seizure requests sent to `KAGGLE_ENDPOINT` asynchronously
  - local YOLO person detection still runs on host/container

### 2.2 Audio Modes

- `MIC_ENABLED=true|false`: controls real microphone capture.
- `AUDIO_ANALYTICS_ENABLED=true|false`: controls distress/keyword analysis pipeline.

In Docker, `MIC_ENABLED=false` and `AUDIO_ANALYTICS_ENABLED=true` supports audio analysis from pre-recorded clip injection without requiring host microphone passthrough.

## 3. Core Components

### 3.1 Vision

- Pipeline entry: `visual_guardian/pipeline.py`
- Person detector: `visual_guardian/person_detector.py`
- Fall classifier: `visual_guardian/fall_classifier.py`
- Seizure classifier: `visual_guardian/seizure_classifier.py`
- Temporal encoder and smoothing:
  - `visual_guardian/temporal_encoder.py`
  - `visual_guardian/smoother.py`

Key behavior:

- one shared person detection per frame
- temporal buffering and stride-based fall/seizure inference
- state-aware controls (darkness, inactivity, bed/safety context when enabled)

### 3.2 Audio

- Capture/injection stream: `auditory_watchdog/core/audio_capture.py`
- Distress classifier: `auditory_watchdog/core/distress_classifier.py`
- Keyword spotter: `auditory_watchdog/core/keyword_spotter.py`
- Runtime monitor loop: `AuditoryMonitor` in `scripts/demo/demo_server.py`

### 3.3 Cognitive Verification

- Verifier entry: `cognitive_core/gemini_verifier.py`
- Tier-2 binary confirmation/suppression
- Tier-3 report enrichment (narrative, severity, actions)

### 3.4 API and Database

- App entrypoint: `scripts/demo/demo_server.py`
- Database models and initialization: `scripts/demo/database.py`
- DB backend: PostgreSQL via SQLAlchemy

## 4. Deployment Notes

### 4.1 Docker

- Compose file: `docker-compose.yml`
- App runtime:
  - `OPENVINO_DEVICE=intel:cpu`
  - `PERSON_DETECTOR_PROCESS_EVERY=4` (improves CPU stability)
  - `YOLO_CONFIG_DIR=/tmp/.ultralytics`
- DB service: `postgres:15`

### 4.2 Bare Metal

Recommended for maximum local vision FPS on Intel iGPU:

- set `OPENVINO_DEVICE=intel:gpu`
- run `python scripts/demo/demo_server.py` directly
- optionally keep DB in Docker (`docker compose up -d db`)

## 5. Accuracy and Performance Notes

- Alert quality depends on:
  - model checkpoint quality,
  - configured thresholds in `config/config.yaml`,
  - inference mode latency (`LOCAL` vs `KAGGLE`).
- CPU-only Docker mode trades peak FPS for portability; frame-skip detection cadence is intentionally configurable via env vars.
