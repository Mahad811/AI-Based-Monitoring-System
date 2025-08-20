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
*   **Distress Classification:** Detection of Thuds, Screams, Gasps, Moans.
*   **Keyword Spotting:** Detection of "Help", "Nurse", "Madad" (Bilingual).
*   **Status:** In Development (Simulated for Demos).

### 3. Cognitive Core (The "Brain")
*   **Layer 1 (Reflex Engine):** Instant (<50ms) Bayesian scoring for immediate safety triggers.
*   **Layer 2 (Reasoning Engine):** LLM-based (Gemini 2.5 Pro) analysis of event history for structured clinical alerts.
*   **Corroboration:** Multi-sensor fusion (Vision + Audio) to validate critical alerts.

---

## ⚡ Quick Start

### 1. Setup Environment
```bash
# Activate virtual environment
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Live Demo
This runs the full `VisionPipeline` on your webcam, with the `SimulatedAudio` interface enabled.
```bash
python scripts/demo_live.py
```
*   **Controls:** `q` to quit, `d` to toggle debug overlay.
*   **Audio Simulation:** Press `T` (Thud), `H` (Help), `S` (Silence) to inject audio events.

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
