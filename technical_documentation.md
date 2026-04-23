# Vital Guardian Technical Documentation

## 1. Project Overview & Architecture
Vital Guardian is an AI-powered real-time ICU patient monitoring system. Its architecture is split into three tightly integrated modules operating over a bidirectional WebSocket-driven backend:
- **Visual Guardian**: Processes video streams (hardware-accelerated) to detect patient presence, falls, and seizures using robust temporal window encoders.
- **Auditory Watchdog**: A multi-threaded audio pipeline capturing acoustic data to spot critical environmental distress and verbal keyword triggers.
- **Cognitive Core**: A tiered LLM verification engine (Gemini) that dynamically processes ML-generated events, preventing false alarms and generating structured clinical reports.

These subsystems feed into an interactive, hospital-grade dashboard utilizing a Next.js-inspired Vanilla Javascript frontend backed by a FastAPI server and PostgreSQL database.

---

## 2. Implemented Modules & Features

### 2.1 Visual Guardian
- **Person Detection ([pipeline.py](file:///d:/project/FYP/visual_guardian/pipeline.py))**: Utilizes YOLO11n (optimized via OpenVINO for Intel GPUs) to continuously track patient bounding boxes.
- **Fall Detection ([fall_classifier.py](file:///d:/project/FYP/visual_guardian/fall_classifier.py))**: MoViNet-A2 trained on a 32-frame rolling temporal window. Efficiently recognizes high-velocity downward momentum.
- **Seizure Detection ([seizure_classifier.py](file:///d:/project/FYP/visual_guardian/seizure_classifier.py))**: MoViNet-A2 configured for a 64-frame temporal window. Specifically calibrated to capture the erratic, rhythmic movements indicative of seizures.
- **Pipeline Orchestration**: Downsamples real-time 60/120 fps video feeds into a controlled ~30fps stride ensuring ML windows precisely map to 1-2 seconds of actual patient history. Handles overlap smoothing and UI gauge mappings.

### 2.2 Auditory Watchdog
- **Media Engine & Audio Capture**: Seamlessly overrides physical hardware streams (via `PyAudio`) to automatically extract (`moviepy`) and sync (`librosa` at 16kHz) audio directly from demo testing clips when available.
- **Distress Classifier ([core/distress_classifier.py](file:///d:/project/FYP/auditory_watchdog/core/distress_classifier.py))**: Uses YAMNet to evaluate background anomalies and detect physical distress patterns including gasping, heavy breathing, sneezing, and coughing.
- **Keyword Spotter ([core/keyword_spotter.py](file:///d:/project/FYP/auditory_watchdog/core/keyword_spotter.py))**: Uses Faster-Whisper to provide highly accurate, zero-shot transcriptions of patient speech, identifying critical distress words (e.g., "Help", "Nurse").
- **Audio Accumulator**: Instead of firing immediate alerts, distress signals score points (0-10) pushed into a 15-second rolling deque. Alerts are only fired when the contextual risk score breaches the threshold, dramatically eliminating transient false positives.

### 2.3 Cognitive Core
- **Progressive Verification ([gemini_verifier.py](file:///d:/project/FYP/cognitive_core/gemini_verifier.py))**: Replaces the old, static reflex matrices with a dynamic, progressive verification schema via the Gemini API:
  - **Tier 2 (Binary Verification)**: Ultra-fast LLM assessment evaluating bounding boxes and raw frames to output a binary `CONFIRMED` or `SUPPRESSED`. Used to instantly mute false ML spikes.
  - **Tier 3 (Clinical Enrichment)**: If confirmed, a secondary deep-reasoning prompt is triggered, generating a structured JSON `IncidentReport` containing a clinical narrative, severity scaling, and recommended nursing actions.
- **High-Confidence Bypass**: Fallback deterministic bypasses auto-confirm ML predictions measuring >50% confidence, relying strictly on the highly-trained MoViNet checkpoints over generalized LLM visual interpretation.

### 2.4 FastAPI Web Server & Clinical Dashboard
- **WebSockets Engine ([demo_server.py](file:///d:/project/FYP/scripts/demo/demo_server.py))**: Pushes base64 encoded frames, pipeline metadata, audio/vision alerts, and Gemini LLM payloads at high concurrency without blocking the primary visual loop.
- **Database Architecture ([database.py](file:///d:/project/FYP/scripts/demo/database.py))**: SQLAlchemy mapped to PostgreSQL managing `Nurse` credentials, `Patient` assignments (with varying clip modes like live feeds, asthma, or whooping cough), full `IncidentLog` traceability, and security `AuditLogs`.
- **VGAudio UI Subsystem ([app.js](file:///d:/project/FYP/scripts/demo/public/app.js))**: An advanced, file-less audio synthesizer integrated directly into the DOM. Uses the Web `AudioContext` API to generate ADSR-modeled medical alarms (e.g., triangle waves for falls, square waves for seizures) and Web Speech API TTS voice readouts. 

---

## 3. Models Used & Performance

1. **Person Detection**
   - **Model:** YOLO11n (Intel GPU / OpenVINO)
   - **Purpose:** Reliable boundary extraction preventing background noise interference.
2. **Fall Classification**
   - **Model:** MoViNet-A2 (SavedModel format)
   - **Input:** 32-frame Temporal RGB clips.
3. **Seizure Classification**
   - **Model:** MoViNet-A2 (SavedModel format)
   - **Input:** 64-frame Temporal RGB clips.
4. **Auditory Distress**
   - **Model:** YAMNet 
   - **Purpose:** Captures involuntary medical distress markers (Coughs, Sneezing, Gasps).
5. **Keyword Spotter**
   - **Model:** Faster-Whisper (`tiny` scale) 
6. **Cognitive Core (Reasoning Engine)**
   - **Model:** Gemini-1.5-Flash / Gemini-3-Flash (`gemini-3-flash-preview`)

### System Performance
- **Latency Pipeline**: ML pipelines average sub-100ms inference on appropriate hardware.
- **LLM Verification**: Tier 2 completes in ~1-2 seconds. Tier 3 clinical enrichment averages ~4-6 seconds, during which the frontend executes "hold" UI states.

---

## 4. Deployment Status & Future Work
With the elimination of the legacy Flask app, the system now runs entirely on the sophisticated, asynchronous FastAPI framework. Future development will focus primarily on hardening the Dockerization layers for Kubernetes deployment and gathering longitudinal testing data in physical clinical trial environments.
