# Quick Start Guide

## 1. Environment Setup

**Prerequisite:** Python 3.10+ installed.

```bash
# 1. Clone/Navigate to repository
cd d:\project\FYP

# 2. Create Virtual Environment (if not exists)
python -m venv venv

# 3. Activate Virtual Environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
# source venv/bin/activate

# 4. Install Dependencies
pip install -r requirements.txt
```

---

## 2. Running the Live Demo

The core of the project is the **VisionPipeline**. You can run it live on your webcam.

```bash
python scripts/demo_live.py
```

### Controls & Features
*   **Webcam Feed:** Main window showing detection bounding boxes and posture analysis.
*   **State Display:** Shows `IN_BED`, `OUT_OF_BED`, or `FALLEN`.
*   **Debug Mode:** Press `d` to toggle detailed skeleton and ROI visualization.
*   **Quit:** Press `q` to exit.

### Simulating Audio Inputs (Wizard of Oz)
Since the full audio hardware might not be available, you can **simulate** audio events during the demo to test the Cognitive Core's reaction:

*   **Press `T`**: Simulates a **"Thud"** (impact sound).
    *   *Effect:* Corroborates a fall detection.
*   **Press `H`**: Simulates **"Help"** keyword.
    *   *Effect:* Escalates alert to CRITICAL.
*   **Press `S`**: Simulates **Silence**.
    *   *Effect:* Could indicate unconsciousness if post-fall.

---

## 3. Running Evaluations

To verify the performance of the models on the test dataset:

### Fall Detection Evaluation
Evaluates the 5-model EfficientNet Ensemble.
```bash
python scripts/evaluate_fall_detection.py
```
*   **Output:** Precision, Recall, F1-Score (frame-level and video-level).
*   **Current Baseline:** ~85% Recall (Video-Level).

### Seizure Detection Evaluation
Evaluates the Dual-Stream (Motion + Temporal) Ensemble.
```bash
python scripts/evaluate_on_dataset.py
```
*   **Output:** Detailed classification report.
*   **Current Baseline:** ~97% Recall.

---

## 4. Configuration

All system parameters are in `config/config.yaml`.

*   **Adjust Sensitivity:**
    *   Change `vision.fall_classifier.threshold` (Default: 0.64)
    *   Change `vision.seizure_classifier.threshold` (Default: 0.24)
*   **Enable/Disable Modules:**
    *   `vision.bed_exit.enabled: true/false`
    *   `vision.safety_net.enabled: true/false`

---

## 5. Troubleshooting

**Q: "No module named..." error?**
A: Ensure you activated the venv: `venv\Scripts\activate`.

**Q: Camera not opening?**
A: Check if another app (Zoom/Teams) is using the webcam.

**Q: Model not found?**
A: Ensure models are in `fall_detection/fall_v2_ensemble/` and `seizure_detection/`.
