# Vision Pipeline Architectures Summary

The **Visual Guardian** vision pipeline employs a tiered, multi-model approach for real-time patient monitoring in the ICU. To ensure high performance, accuracy, and a CPU-friendly runtime, the system integrates three distinct neural network architectures that work together in a synchronized pipeline.

---

### 1. Person Detection: YOLOv11n (You Only Look Once, version 11 nano)
- **Pipeline Role:** Centralized Region of Interest (ROI) extraction.
- **Component File:** `visual_guardian/person_detector.py`
- **How It Works:**
  - Deployed as the first step in the pipeline, YOLOv11n scans the frame to detect the patient and generate bounding boxes.
  - It runs exactly once per frame (using frame-skipping and bounding-box caching optimizations) to avoid redundant computations. 
  - The generated bounding boxes are used to crop the video frame. This ensures that downstream classifiers only process the patient, eliminating background noise and drastically saving computational resources.
- **Acceleration:** Pre-compiled and optimized via **Intel OpenVINO** (`yolo11n_openvino_model`) to accelerate inference on standard CPUs or integrated GPUs.

---

### 2. Spatio-Temporal Action Classification: MoViNet-A2 (Mobile Video Networks)
- **Pipeline Role:** Streaming video classification for acute events (Falls and Seizures).
- **Component Files:** `visual_guardian/fall_classifier.py` & `visual_guardian/seizure_classifier.py`
- **How It Works:**
  - Developed by Google, MoViNets are designed specifically for streaming video. They use causal operations, allowing them to classify video constantly frame-by-frame instead of waiting for a full video clip.
  - **Fall Detection Branch:** Analyzes a temporal buffer of **16 contiguous frames** representing a short time window to capture the rapid mechanics of a fall. Output is smoothed via a sliding window.
  - **Seizure Detection Branch:** Analyzes an extended temporal window. It buffers 64 raw frames but evaluates them with a stride of 2 (feeding **32 frames** to the model) to capture the prolonged, rhythmic nature of a seizure.
  - Models for both workflows are managed via the `movinet_loader.py` utility.

---

### 3. Pose & Biomechanical Analysis: MediaPipe Pose
- **Pipeline Role:** Granular actigraphy, state verification, and safety-net checks.
- **Component File:** `visual_guardian/pose_analyzer.py`
- **How It Works:**
  - Extracts 33 anatomical 3D keypoints (landmarks) across the human body in real-time. It is highly robust and CPU-efficient.
  - **Sub-Systems built on MediaPipe:**
    - **Bed-Exit Detection:** Tracks the normalized Y-coordinates of the patient's hips relative to a configurable bed boundary.
    - **Fallen State Safety Net:** Calculates torso angles (identifying if the patient is horizontal) and center of mass (checking if they are on the floor) as a secondary validation for missed falls.
    - **Seizure Rhythm Verification (Seizure 2.0):** Employs kinematic tracking of the wrists. It measures displacement variance to suppress false alarms, mathematically verifying whether a patient is shaking or laying completely still.
    - **Sleep Restlessness (Vision 3.0):** Functions as digital actigraphy. It tracks the "Core Block" (shoulders and hips) to measure accumulated kinetic energy, determining if the patient is tossing and turning during rest.

---

### Pipeline Orchestration
These three architectures are bound together inside `visual_guardian/pipeline.py`. When a video frame arrives:
1. **YOLO** determines *where* the patient is.
2. **MediaPipe** determines *what posture* the patient is holding (and tracks micro-movements for context).
3. **MoViNet** processes the cropped temporal sequence to determine *what action* the patient is currently performing.
