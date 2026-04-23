# Vital Guardian: AI-Powered ICU Patient Monitoring System

**Vital Guardian** is a real-time, multimodal monitoring system designed to detect critical events (Falls, Seizures, Distress) in ICU environments. It fuses hardware-accelerated computer vision and concurrent auditory analysis with a cognitive reasoning engine to minimize false alarms and provide context-aware, progressive alerts to nursing staff.

---

## 🚀 System Architecture

### 1. Visual Guardian (Vision Module 3.0)
*   **Fall Detection:** MoViNet-A2 (32-frame temporal window) running via TensorFlow SavedModel.
*   **Seizure Detection:** MoViNet-A2 (64-frame temporal window) optimized for rhythmic anomalies.
*   **Person Detection:** YOLO11n optimized via OpenVINO for Intel GPU hardware acceleration.
*   **Safety Net:** Logic-based state tracking (Bed Exit, Restlessness, Patient Pose Tracking).

### 2. Auditory Watchdog (Audio Module)
*   **Distress Classification:** Dual-threaded PyAudio ingestion feeding YAMNet to classify non-verbal distress cues (Gasps, Coughing, Sneezing, Thuds).
*   **Keyword Spotting:** Faster-Whisper integration to accurately transcribe critical speech ("Help", "Nurse").
*   **Audio Accumulator:** A temporal scoring mechanism calculating a rolling severity score over a 15-second contextual window to mitigate false-alarm fatigue.
*   **Media Mode:** Automatic seamless audio extraction via `moviepy` and `librosa` for pre-recorded clinical videos.

### 3. Cognitive Core (The "Brain")
*   **Tier 1 (Reflex Engine):** Local ML inference providing instantaneous sub-second probabilistic risk tracking (0-100%).
*   **Tier 2 (Binary Verification):** Gemini API fast-pass verification (under 2 seconds) confirming or suppressing Reflex alerts to act as an aggressive false-alarm filter.
*   **Tier 3 (Clinical Enrichment):** Full multimodal LLM reasoning (under 6 seconds) mapping historical frames, bounding boxes, and audio cues into structured, clinical-grade JSON Incident Reports.

### 4. Interactive Clinical Dashboard (Web App)
*   **Web Framework:** FastAPI backend with bidirectional WebSockets.
*   **UI/UX:** State-of-the-art Vanilla JS frontend featuring real-time risk gauges (`FSIM`/`SZSIM`), dynamic event overlays, and full patient history review panes.
*   **Hospital-Grade Audio (`VGAudio`):** In-browser ADSR wave synthesizer generating medical-standard Triangle/Square wave alarms and Text-to-Speech (TTS) announcements, removing reliance on static sound files.
*   **Database Engine:** PostgreSQL via SQLAlchemy tracking `Nurses`, `Patients`, `IncidentLogs`, and `AuditLogs` for full clinical traceability.

---

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo Server
This boots the integrated FastAPI server, running the active multimodal inference pipeline.
```bash
# Ensure PostgreSQL is running on port 5432 and GEMINI_API_KEY is in your .env
python scripts/demo/demo_server.py
```
*   Access the dashboard at `http://localhost:8000`
*   Log in using the default Nurse or Admin credentials dynamically seeded in PostgreSQL.

---

## 📂 Project Structure

| Directory | Purpose |
|:---|:---|
| `visual_guardian/` | Core vision logic (MoViNet loaders, YOLO11 OpenVINO, Vision Pipeline) |
| `auditory_watchdog/` | Audio processing (YAMNet Distress Classifier, Whisper Keyword Spotter, Accumulators) |
| `cognitive_core/` | Progressive AI Verification (Gemini Tier 2 / Tier 3 Verifier) |
| `scripts/demo/` | FastAPI Web Server, WebSocket managers, SQLAlchemy schemas (`database.py`) |
| `scripts/demo/public/` | Dashboard Frontend (Vanilla JS, CSS, Hospital UI) |
| `demo_dataset/` | Pre-processed testing videos (Falls, Asthma, Whooping Cough, Normal) |
| `config/` | System configuration (`config.yaml`) |

## 📚 Documentation
*   **[Technical Documentation](technical_documentation.md):** Deep dive into module implementations and API flows.
*   **[Visual Guardian V2 Plan](Visual_Guardian_V2_Final_Plan.md):** Legacy progression roadmap document.

---

**Status:** Active Development (Final Phase: UI Overhaul & Multi-modal Integration)
