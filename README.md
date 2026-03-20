# Vital Guardian: AI-Powered ICU Patient Monitoring System

**Vital Guardian** is a real-time, multimodal monitoring system designed to detect critical events (Falls, Seizures, Distress) in ICU environments. It fuses computer vision and auditory analysis with a cognitive reasoning engine to minimize false alarms and provide context-aware alerts to nursing staff.

---

## 🚀 System Architecture

### 1. Visual Guardian (Vision Module 2.0)
*   **Fall Detection:** 5-model EfficientNet-B0 Ensemble (Temporal RGB Triplets).
    *   *Performance:* 85% Recall (Video-Level), 82.9% F1-Score.
*   **Seizure Detection:** Dual-Stream Ensemble (Motion + Temporal).
    *   *Performance:* 97.6% Recall, Robust Rhythm Verification.
*   **Safety Net:** Logic-based checks (Bed Exit, Floor Zone, Inactivity) to prevent misses.

### 2. Auditory Watchdog (Audio Module)
*   **Distress Classification:** Detection of Thuds, Screams, Gasps, Moans using offline PyAudio + YAMNet.
*   **Keyword Spotting:** Detection of "Help", "Nurse", "Madad" (Bilingual) using offline Faster-Whisper.
*   **Privacy Shield:** Real-time VAD filtering out non-distress background conversations.
*   **Status:** Fully Integrated (Concurrent non-blocking audio capture).

### 3. Cognitive Core (The "Brain")
*   **Layer 1 (Reflex Engine):** Instant (<50ms) Bayesian scoring for immediate safety triggers.
*   **Layer 2 (Reasoning Engine):** LLM-based (Gemini 3 Flash) multimodal reasoning mapping camera frames + sensor history into structured JSON Incident Reports.
*   **Corroboration:** Multi-sensor fusion (Vision + Audio) to gracefully validate or suppress critical alerts.

---

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Active Monitoring Pipeline
This runs the fully integrated system, accessing your hardware camera and microphone concurrently.
```bash
# Add GEMINI_API_KEY to your .env file in the root directory first.
python main.py
```
*   **Controls:** Press `q` in the video window to quit.
*   **Physical Testing:** Try coughing or groaning loudly to trigger YAMNet distress markers safely, or say "Nurse, I need help" to trigger the Faster-Whisper transcriber.

### 3. Run Evaluation
Validate the vision models on the test dataset.
```bash
# Evaluate Seizure Detection
python scripts/evaluate_on_dataset.py

# Evaluate Fall Detection
python scripts/evaluate_fall_detection.py
```

---

## 📂 Project Structure

| Directory | Purpose |
|:---|:---|
| `visual_guardian/` | Core vision logic (Pipeline, Classifiers, Pose Analysis) |
| `auditory_watchdog/` | Audio processing (Classifiers, Keyword Spotting) |
| `cognitive_core/` | Fusion & Reasoning (Reflex Engine, Gemini Integration) |
| `scripts/` | Production scripts (Demo, Eval, Prep) |
| `scripts/deprecated/` | Archived experiments (YOLO, V1 attempts) |
| `config/` | System configuration (`config.yaml`) |
| `docs/` | Documentation & Guides |

## 📚 Documentation
*   **[Quick Start Guide](docs/QUICK_START.md):** Detailed setup and running instructions.
*   **[Dataset Guide](docs/DATASET_GUIDE.md):** Explanation of training data.
*   **[Training Results](docs/TRAINING_RESULTS_GUIDE.md):** Model performance metrics.
*   **[Architecture Overview](docs/guides/ARCHITECTURE_OVERVIEW.md):** Deep dive into system design.

---

## 🛠 Configuration
Configuration is managed in `config/config.yaml`.
*   **Vision:** Adjust thresholds (`fall_classifier.threshold`), enable/disable modules.
*   **Cognitive Core:** Set API keys, alert sensitivities.
*   **Hardware:** Toggle GPU/CPU inference.

---

**Status:** Active Development (Phase 3: Cognitive Core Integration)
