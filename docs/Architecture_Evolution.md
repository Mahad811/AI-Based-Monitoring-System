# Vision Pipeline Evolution: The 3 Stages of Architecture

This document outlines the evolutionary steps taken to develop the final models for Fall and Seizure detection in the ICU, detailing what was implemented at each stage and the technical reasons for moving forward.

---

### Stage 1: Single-Frame Object Detection (YOLO)
**Approach:** We initially treated anomaly detection as a static image classification problem, using YOLO (e.g., YOLOv8) to classify individual frames as "Normal" or "Fall".

**Why We Moved Away:**
* **Zero Temporal Context:** YOLO has no concept of time or movement. It analyzes a single frozen frame.
* **Appearance Bias (Shortcuts):** The model learned to identify "fall poses" (a horizontal person) rather than the actual dynamic mechanics of falling. This led to severe false positives, such as misclassifying a patient safely sleeping in bed as having fallen.
* **Incompatible with Seizures:** Seizures are defined by rhythmic, temporal motion. A seizure cannot be diagnosed from a single frozen frame, making YOLO fundamentally incapable of serving a dual-detection purpose.

---

### Stage 2: 2D Temporal Ensembles (EfficientNet Stacking)
**Approach:** To introduce motion without leaving 2D architectures, we began stacking consecutive video frames (e.g., 3 to 12 frames) encoded across the RGB color channels. This stacked "temporal image" was then fed into a 5-fold ensemble of 2D EfficientNet classifiers.

**Why We Moved Away:**
* **"Hacky" Temporal Encoding:** Stacking time into color channels is a makeshift solution. The network still viewed the input as a single 2D image with weird colors, rather than truly understanding temporal flow.
* **Computational Bloat:** Running 5 independent EfficientNet models across sliding temporal windows was extremely CPU-heavy and caused dangerous latency in real-time execution.
* **Residual Appearance Traps:** Extensive testing showed that despite motion encoding, the 2D networks still relied heavily on background and clothing appearances, meaning changes in ICU room lighting or patient attire compromised accuracy.

---

### Stage 3: Streaming Spatio-Temporal Networks (MoViNet) — *Current*
**Approach:** We entirely abandoned 2D image models and adopted **MoViNet-A2 (Mobile Video Networks)**, an architecture built natively by Google for streaming video processing.

**Why We Moved to It (And Succeeded):**
* **Native 3D Convolutions:** MoViNets are designed to handle Spatio-Temporal data intrinsically. They evaluate both spatial features (what the person looks like) and temporal trajectories (how they are moving over time) in a unified way.
* **Causal Memory Buffers:** Instead of analyzing overlapping windows from scratch every frame, MoViNets use causal operations and internal states (buffers). They remember the previous frames, allowing them to classify video seamlessly frame-by-frame with minimal latency.
* **Streamlined Performance:** We dropped the bloated 5-fold ensemble arrays. A single streaming MoViNet vastly outperformed the old ensembles while being incredibly CPU-efficient. 
* **Extended Horizons:** This architecture effortlessly handles long temporal horizons, allowing us to evaluate 16 frames for falls and 32 continuous frames for rhythmic seizure verification with zero framerate drops.
