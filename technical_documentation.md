# Vital Guardian Technical Documentation

## 1. Project Overview & Architecture
Vital Guardian is an AI-powered real-time ICU patient monitoring system. Its primary architecture is split into three main modules:
- **Visual Guardian**: Processes video streams to detect falls, seizures, bed exits, and restlessness.
- **Auditory Watchdog**: Processes audio streams to spot critical keywords and distress sounds.
- **Cognitive Core**: A dual-layered decision engine that fuses vision and audio events to generate alerts.

The system uses a "Reflex-then-Reason" pipeline. The deterministic [ReflexEngine](file:///d:/project/FYP/cognitive_core/reflex_engine.py#17-177) provides immediate, sub-50ms alerts (Layer 1). Significant events are then passed to the [ReasoningEngine](file:///d:/project/FYP/cognitive_core/reasoning_engine.py#121-379) (Layer 2), which uses an LLM (Gemini 3 Flash) to generate a contextual, natural-language incident report within 2-5 seconds.

---

## 2. Implemented Modules & Features

### 2.1 Visual Guardian
- **Person Detection ([person_detector.py](file:///d:/project/FYP/visual_guardian/person_detector.py))**: Identifies patient bounding boxes to isolate regions of interest.
- **Fall Detection ([fall_classifier.py](file:///d:/project/FYP/visual_guardian/fall_classifier.py), [temporal_encoder.py](file:///d:/project/FYP/visual_guardian/temporal_encoder.py))**: Encodes the video stream into Temporal RGB triplets (past, current, future). Detects falls by passing these encoded images to an image classifier.
- **Seizure Detection ([seizure_classifier.py](file:///d:/project/FYP/visual_guardian/seizure_classifier.py))**: Uses a dual ensemble approach. Computes a Motion-Only Summary (mean/std/max diffs across frames) and a Temporal Map (2D spectrogram of motion).
- **Pose Analysis & Safety Nets ([pose_analyzer.py](file:///d:/project/FYP/visual_guardian/pose_analyzer.py))**: Uses skeleton landmarks to identify posture. Features include:
  - Bed Exit Detection: Tracks hips crossing predefined bed boundaries.
  - Digital Actigraphy: Measures gross body movement for sleep restlessness.
  - Fallen State Safety Net: Detects if the torso is horizontal and near the floor.
- **Pipeline Integration ([pipeline.py](file:///d:/project/FYP/visual_guardian/pipeline.py))**: Central orchestration script that handles fall/seizure frame buffering, applies overlapping window smoothers, and manages system state (IN_BED, EXITING, FALLEN).

### 2.2 Auditory Watchdog
- **Distress Classifier ([core/distress_classifier.py](file:///d:/project/FYP/auditory_watchdog/core/distress_classifier.py))**: Analyzes audio chunks to classify non-verbal distress sounds like gasps, moans, coughing, or cries using an offline YAMNet model.
- **Keyword Spotter ([core/keyword_spotter.py](file:///d:/project/FYP/auditory_watchdog/core/keyword_spotter.py))**: Uses an offline, zero-shot `faster-whisper` model to transcribe full sentences spoken by the patient in English or Urdu directly to text.
- **Privacy Shield ([core/privacy_shield.py](file:///d:/project/FYP/auditory_watchdog/core/privacy_shield.py))**: Employs Silero VAD to detect human speech and preserve privacy. Activates "Visitor Mode" during long conversations, only passing relevant distress bursts and distinct short sentences.
- **Audio Capture ([core/audio_capture.py](file:///d:/project/FYP/auditory_watchdog/core/audio_capture.py))**: Uses `PyAudio` to non-blockingly read a continuous rolling microphone buffer synchronously alongside the video stream.

### 2.3 Cognitive Core
- **Reflex Engine ([reflex_engine.py](file:///d:/project/FYP/tests/test_reflex_engine.py))**: A Bayesian-inspired scoring matrix. Evaluates ([VisionEvent](file:///d:/project/FYP/cognitive_core/models.py#45-81), [AudioEvent](file:///d:/project/FYP/cognitive_core/models.py#83-105)) tuples and triggers immediate deterministic alerts (e.g., Fall + Scream = CRITICAL).
- **Reasoning Engine ([reasoning_engine.py](file:///d:/project/FYP/cognitive_core/reasoning_engine.py))**: Interfaces with the Google Gemini API (specifically `gemini-3-flash-preview`). Consumes recent event logs and the raw frame to output a structured JSON [IncidentReport](file:///d:/project/FYP/cognitive_core/models.py#136-179) conforming to predefined Pydantic schemas. Supports a mock mode for environments without internet access.

---

## 3. Models Used & Performance

1. **Person Detection**
   - **Model:** YOLOv8n (Pretrained on COCO)
   - **Purpose:** Fast, real-time bounding box extraction for the patient.
2. **Fall Classification**
   - **Model:** EfficientNet-B0 (supports up to a 5-fold ensemble)
   - **Input:** 224x224 Temporal RGB images.
   - **Purpose:** Recognizes high-velocity, uncontrolled downward movement typical of falls.
3. **Seizure Classification**
   - **Model:** EfficientNet-B0 (up to a 10-model Dual Ensemble: 5 motion + 5 temporal models)
   - **Input:** Motion summary and Temporal Map images.
   - **Purpose:** Identifies rhythmic, erratic motions distinguishing seizures from normal movement.
4. **Pose Estimation**
   - **Model:** MediaPipe Pose (CPU-friendly, complexity level 1)
   - **Purpose:** Bed exit, safety nets, and actigraphy base tracking.
5. **Auditory Distress**
   - **Model:** YAMNet (Loaded via TensorFlow Hub)
   - **Purpose:** Pre-trained environmental audio classifier to identify distress classes (whimper, gasp, groan, etc.).
6. **Reasoning Engine**
   - **Model:** Gemini-3-Flash (`gemini-3-flash-preview`)
   - **Purpose:** Multimodal context synthesis. Confirms or suppresses reflex alerts based on the visual cue in the frame.
7. **Keyword Spotter**
   - **Model:** Faster-Whisper (`tiny` scale) 
   - **Purpose:** Full-sentence transcription of patient speech for LLM contextualization.

### System Performance
- **Latency**: Reflex alerts run locally on CPU/GPU and trigger in <50ms. Gemini reasoning reports take around 1-3 seconds per event.
- **Unit Tests**: The logic in the Cognitive Core layers ([ReflexEngine](file:///d:/project/FYP/cognitive_core/reflex_engine.py#17-177), [ReasoningEngine](file:///d:/project/FYP/cognitive_core/reasoning_engine.py#121-379), and models) has passing `pytest` unit tests, ensuring robust event evaluation and fallback protocols.

---

## 4. Remaining Implementation Work (Gap Analysis)

Based on a thorough audit of the source files versus the project requirements, the following tasks are still incomplete:

1. **Dashboard Enhancements:**
   - [dashboard/app.py](file:///d:/project/FYP/dashboard/app.py) is currently a rudimentary Flask app polling an in-memory alert list. For production/clinic deployment, this should be upgraded with a stable database (e.g., SQLite/PostgreSQL) and potentially Socket.IO for push-based alert updates.

4. **Production Testing:**
   - The individual neural networks are present, but their respective pre-trained weights (`*.pt` models) must be robustly validated on the staging machine to ensure `timm.create_model` has access to the correct local `.pt` checkpoints without throwing `FileNotFoundError`.

5. **Data Preprocessing Integration:**
   - The `data_preprocessing` module holds scripts necessary for retraining/fine-tuning. While not part of inference, it should be documented if future data ingestion is expected to be automated.
