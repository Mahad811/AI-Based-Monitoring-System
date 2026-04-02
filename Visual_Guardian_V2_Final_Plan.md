# Visual Guardian V2 — Final Rebuild Plan
### FYP Architecture Upgrade: MoViNet-A2 Pipeline

---

## 1. Why We Are Rebuilding

### Documented Failures of Current System

| Component | Problem | Impact |
|---|---|---|
| Fall Classifier | 5× EfficientNet-B0 on 3-frame RGB triplets | ~75.6% recall — 1 in 4 falls missed |
| Seizure Classifier | 10× EfficientNet-B0 on collapsed motion images | F1 ~0.71, Precision ~0.63 — too many false alarms |
| Temporal Encoding | Grayscale-to-RGB channel stacking trick | 2D CNN cannot truly understand motion trajectories |
| Seizure Encoding | 60-frame motion collapsed to single image | Temporal dynamics destroyed before model sees data |
| Runtime | 15 EfficientNet forward passes per cycle | Severe compute overhead on all hardware |
| Gemini | Blocks the alert path for 2–5 seconds | Unacceptable latency for a life-safety system |

### Root Cause
EfficientNet-B0 is a 2D image classifier. It was never designed to understand time, velocity, or rhythm. The entire 15-model ensemble is an approximation of temporal understanding using image-classification workarounds. MoViNet-A2 processes actual frame sequences natively. Two models replace fifteen. Additionally, YOLOv8n is upgraded to YOLO11n for improved occlusion handling and more stable person crops — a direct benefit to MoViNet's temporal input quality.

---

## 2. Datasets

### Fall Detection Dataset
**Source:** Kaggle Compiled Fall Dataset (all 3 internal sources combined)

| Internal Source | Description | Clips |
|---|---|---|
| FallVision (Harvard Dataverse) | 58 volunteers, falls from bed/chair/standing, MP4, multi-background | ~600 |
| Figshare 29-Subject Dataset | 999 falls (left/right/front/back/sitting/elevated), 1,017 ADL clips | ~2,016 |
| Montreal Multi-Camera Dataset | 22 fall scenarios × 8 synchronized IP cameras, 24 confounding events | ~176+ |
| **Total** | **Balanced fall and no-fall clips** | **~3,000** |

> **Critical note:** The Harvard FallVision dataset is already included inside the Kaggle compiled set. Do NOT add the standalone Harvard dataset separately — this would create duplicate clips and artificially inflate training metrics.

### Seizure Detection Dataset
**Source:** IEEE DataPort — Seizure Videos of Epilepsy Patients (Jiuxing Liang, 2024)

| Class | Clips | Duration Each |
|---|---|---|
| Seizure (tonic-clonic + absence) | 403 | 5 seconds |
| Normal (ADL activities) | 403 | 5 seconds |
| **Total** | **806** | — |

> **Known limitation:** Source dataset explicitly excluded occluded patients (those under blankets). This is a training-deployment mismatch for real patient monitoring. The augmentation strategy below directly compensates for this.

---

## 3. Final Architecture

### Component Overview

| Component | Model | Status |
|---|---|---|
| Person Crop | YOLO11n | Upgraded from YOLOv8n — one-line change, drop-in compatible |
| Fall Detection | MoViNet-A2 | Replaces 5× EfficientNet-B0 ensemble |
| Seizure Detection | MoViNet-A2 | Replaces 10× EfficientNet-B0 ensemble |
| Alert | Instant trigger | Gemini removed from critical path |
| Reporting | Gemini Flash (async) | Generates medical incident report in background |

### Why YOLO11n Over YOLOv8n
YOLO11n introduces spatial attention mechanisms (C2PSA module) giving it better handling of partial occlusion and producing more stable bounding boxes across frames. Stable crops directly improve MoViNet's frame sequence quality. It has 22% fewer parameters than YOLOv8m while achieving higher mAP, and the nano variant is a direct drop-in replacement. Impact on overall system is minor but it costs one line of code and zero additional compute.

### System Flow

```
Camera Feed (30fps)
        |
   YOLO11n (UPGRADED from YOLOv8n)
   Person detection + crop
        |
   224x224 Person Crop
        |
   SharedFrameBuffer
   deque(maxlen=64)
        |
   +----+----+
   |         |
FALL      SEIZURE
MoViNet   MoViNet
A2        A2
16f/s2    32f/s2
~1.07s    ~2.13s
   |         |
   +----+----+
        |
SlidingWindowSmoother (UNCHANGED)
        |
   Instant Alert Fired
        |
   Gemini Flash (ASYNC background)
   Generates medical incident report
   Does NOT block the alert path
```

### Model Specifications

| | Fall Model | Seizure Model |
|---|---|---|
| Architecture | MoViNet-A2 | MoViNet-A2 |
| Input frames | 16 frames | 32 frames |
| Temporal stride | 2 (every other frame) | 2 (every other frame) |
| Effective window | ~1.07 seconds | ~2.13 seconds |
| Raw buffer needed | 32 frames | 64 frames |
| Classification | Binary: Fall / Normal | Binary: Seizure / Normal |
| Pretrained weights | Kinetics-600 | Kinetics-600 |
| Parameters | ~4.8M | ~4.8M |
| Training data | Kaggle fall dataset | IEEE seizure dataset |

**Why stride-2:** Doubles effective temporal coverage without increasing model input size. Fall model covers pre-fall instability + the fall itself. Seizure model matches the existing proven 2-second window from the current config.

### Pipeline Configuration (config.yaml)

```yaml
fall_classifier:
  model: fall_detection/movinet_a2/fall_model.h5
  threshold: 0.60           # Start here — tune after validation
  window_size: 10           # Smoother window — keep unchanged
  inference_stride: 8       # Run every 8 frames (~3.75x per second)
  clip_frames: 16
  temporal_stride: 2

seizure_classifier:
  model: seizure_detection/movinet_a2/seizure_model.h5
  threshold: 0.55           # NOT 0.40 — precision is primary metric for seizures
  window_size: 8            # Smoother window — keep unchanged
  inference_stride: 15      # Run every 15 frames (~2x per second)
  clip_frames: 32
  temporal_stride: 2
```

> **Threshold rationale:** Seizure threshold starts at 0.55, not 0.40. Your current system already has precision issues (~0.63). A 0.40 threshold fires at 40% model confidence — that will make false alarms significantly worse. Tune downward only after validating recall is insufficient.

### Simultaneous Event Rule
When both models fire within the same inference cycle (e.g., seizure causing a fall):
- **Fall alert takes immediate priority** — fires to the UI first
- Seizure flag is appended to the alert metadata
- Gemini async report receives both flags and generates a combined incident description

---

## 4. Files to Rewrite, Update, and Keep

### Files to REWRITE Completely

**`visual_guardian/fall_classifier.py`**
- Remove: 5× EfficientNet-B0 ensemble, temporal RGB triplet encoder, ensemble voting logic
- Replace with: Single MoViNet-A2 TF model, load saved .h5 weights
- Expose: `classify(clip_tensor: np.ndarray) -> float` interface (same as current)

**`visual_guardian/seizure_classifier.py`**
- Remove: 10× EfficientNet-B0 ensemble, dual-stream motion/temporal map logic
- Replace with: Single MoViNet-A2 TF model
- Expose: Same `classify(clip_tensor) -> float` interface

**`visual_guardian/temporal_encoder.py`**
- Remove: Frame stacking, grayscale-to-RGB trick, spectrogram generation
- Replace with: `SharedFrameBuffer` class
  - `deque(maxlen=64)` of person-cropped 224×224 frames
  - `sample_clip(n_frames, stride) -> np.ndarray` returning shape `[n_frames, 224, 224, 3]`
  - Used by both fall (n=16, stride=2) and seizure (n=32, stride=2) branches

**`visual_guardian/pipeline.py`**
- Rewrite `process_frame()`:
  1. YOLO person detect (same as now)
  2. Crop person → resize 224×224 → push to SharedFrameBuffer
  3. Every 8 frames: if buffer ≥ 32 frames → sample fall clip → run fall MoViNet
  4. Every 15 frames: if buffer ≥ 64 frames → sample seizure clip → run seizure MoViNet
  5. Feed raw confidences to existing SlidingWindowSmoother
  6. Apply thresholds → return event dict (same interface as current)

### Files to UPDATE

**`config/config.yaml`**
- New model paths, updated thresholds and window parameters (see Section 3 above)
- Remove all ensemble references

**`scripts/demo/demo_server.py`**
- Move GeminiVerifier fully off the critical alert path
- Change to `asyncio.create_task()` that fires AFTER alert is sent to UI
- Update `move_models_to_gpu()` for TF models:
  ```python
  gpus = tf.config.list_physical_devices('GPU')
  for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  ```
- Update threshold overrides for new model characteristics

**`requirements.txt`**
- Add: `tf-models-official`
- Check: Remove `timm` if no remaining code references it

### Files to KEEP Unchanged

| File | Reason |
|---|---|
| `visual_guardian/person_detector.py` | One-line model change only — `YOLO('yolo11n.pt')` replaces `YOLO('yolov8n.pt')`. Logic unchanged |
| `visual_guardian/smoother.py` | SlidingWindowSmoother is architecture-agnostic |
| `cognitive_core/gemini_verifier.py` | Keep — just call it async instead of blocking |
| `scripts/demo/public/` | Frontend unchanged — same alert interface |

---

## 5. Preprocessing Pipeline

### Step 1 — Offline YOLO Crop (Run Once Before Training)

Do not run YOLO during training. Preprocess all clips once and save to disk. This saves GPU cycles and prevents train/test distribution mismatch.

**Model:** YOLO11n — upgraded from YOLOv8n. One-line change, drop-in compatible via ultralytics. Better occlusion handling and more stable bounding boxes directly improve MoViNet's frame sequence quality.

```python
# One-line upgrade in person_detector.py
model = YOLO('yolo11n.pt')  # replaces YOLO('yolov8n.pt')
```

```
For each video clip in dataset:
    Run YOLOv8n on every frame
    Crop person bounding box with 20% padding
    Resize crop to 224×224
    Save as sequential PNG frames

Output structure:
preprocessed/
  fall_dataset/
    train/fall/clip_001/frame_0000.png ...
    train/normal/clip_001/frame_0000.png ...
    val/...
    test/...
  seizure_dataset/
    train/seizure/...
    train/normal/...
    val/...
    test/...
```

Minimum clip length filter:
- Fall clips: must have ≥ 32 raw frames (to support 16 frames × stride 2)
- Seizure clips: must have ≥ 64 raw frames (to support 32 frames × stride 2)
- Discard anything shorter

### Step 2 — Train/Val/Test Split

```
70% train / 15% val / 15% test
Split by subject/volunteer ID — NEVER by clip
Save split manifest as CSV: clip_id, subject_id, split, label
Verify no subject appears in more than one split
```

> **Why subject-level split matters:** If clips from the same volunteer appear in both train and test sets, the model memorizes that person's body proportions and movement style. Test accuracy will look inflated but the model will fail to generalize to new patients.

### Step 3 — Augmentation

**Fall Model Augmentations** (applied during training, not preprocessing):

| Augmentation | Parameters |
|---|---|
| Random horizontal flip | 50% probability |
| Random brightness/contrast | ±30% |
| Gaussian noise | sigma = 0.05 |
| Random crop + resize | Scale 0.85–1.0 → resize to 224 |
| Temporal jitter | Random start offset ±2 frames within clip |

**Seizure Model Augmentations** (all fall augmentations plus):

| Augmentation | Parameters | Reason |
|---|---|---|
| Random rectangular occlusion | 10–40% of frame area, gray/dark fill | Simulates patients under blankets — critical compensation for dataset gap |
| Temporal resampling | Speed 0.8×–1.2× | Seizure rhythm varies between patients |
| MixUp between seizure clips | alpha = 0.2 | Synthetic examples from small dataset |
| Vertical flip | 30% probability | Patients in varied orientations |

---

## 6. Training Strategy

### Framework
- MoViNet is TensorFlow-native via `tf-models-official`
- TensorFlow is already in your `requirements.txt` — no framework conflict
- YOLO (PyTorch/ultralytics) and MoViNet (TensorFlow) coexist on the same GPU
- Train on Kaggle notebooks (free T4 16GB GPU, TF/Keras first-class support)

### TF + PyTorch GPU Coexistence — Critical Setup

```python
# MUST run before any TF model loading — prevents TF from grabbing all GPU memory
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Load TF models FIRST, then YOLO
# If OOM errors persist during full pipeline, run YOLO on CPU:
# model = YOLO('yolov8n.pt')
# model.to('cpu')  # YOLOv8n is fast enough on CPU
```

### Weights Verification — Do This on Day 1 of Phase 1

```python
import tensorflow_hub as hub
# Verify this resolves and loads before building anything else
movinet = hub.load(
    "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"
)
```
If this URL fails, use the PyTorch community port (`torch-movinet`) as a drop-in alternative. Do not discover this problem in Phase 2.

### Training Loop (Both Models)

**Phase A — Head Only (5 epochs)**
- Freeze backbone
- Train classification head only
- Learning rate: 1e-3
- Loss: Binary cross-entropy

**Phase B — Full Fine-Tune (15–20 epochs)**
- Unfreeze all layers
- Learning rate: 1e-4 with cosine decay
- Loss: Binary cross-entropy
- Batch size: 8–16 (depending on GPU memory)

**Seizure Model Additional Settings:**
- Weight decay: 1e-4 (stronger regularization for small dataset)
- Label smoothing: 0.1 (if val F1 plateaus)
- Early stopping: monitor val loss

### Evaluation Metrics

| Model | Primary Metric | Secondary | Target |
|---|---|---|---|
| Fall | Recall (missed falls are dangerous) | F1, AUC | Recall > 90%, F1 > 85% |
| Seizure | Precision (false alarms cause alert fatigue) | F1, AUC | F1 > 0.78, Precision > 0.70 |

Save threshold sweep results from training — use them for pipeline threshold calibration in Phase 4.

---

## 7. Known Warmup Delay

The fall model requires 32 raw frames in the SharedFrameBuffer before first inference can run. At 30fps this is approximately 1 second of warmup after a person enters frame.

This is an inherent property of any temporal video model — it is not a bug. Handle it explicitly:
- Add a `warming_up` state to the pipeline
- Display a visual indicator in the demo UI during warmup
- Document this in the FYP write-up as a known and expected system property

---

## 8. Phase-by-Phase Plan

### Phase 1 — Environment and Data Preparation
**Duration: 2–3 days | Risk: Low**

- [ ] Verify MoViNet-A2 Kinetics-600 weights download (Day 1, before anything else)
- [ ] Download YOLO11n weights — verify `YOLO('yolo11n.pt')` loads and runs correctly
- [ ] Update `visual_guardian/person_detector.py` — replace `yolov8n.pt` with `yolo11n.pt` (one line)
- [ ] Install `tf-models-official`
- [ ] Download Kaggle fall dataset — verify all entries are video clips, not images
- [ ] Download IEEE seizure dataset (806 clips)
- [ ] Filter clips below minimum frame count
- [ ] Verify label consistency across all three fall dataset sources
- [ ] Write and run offline YOLO crop preprocessing script
- [ ] Create subject-level train/val/test splits, save as CSV manifests

---

### Phase 2 — Fall Model Training
**Duration: 2–3 days | Risk: Low**

- [ ] Upload preprocessed fall crops to Kaggle as dataset
- [ ] Load MoViNet-A2 with Kinetics-600 pretrained weights
- [ ] Replace final classification head: `GlobalAveragePooling → Dense(1, sigmoid)`
- [ ] Build `tf.data.Dataset` pipeline loading 16-frame clips with stride-2 sampling
- [ ] Run Phase A training (head only, 5 epochs)
- [ ] Run Phase B training (full fine-tune, 15–20 epochs)
- [ ] Evaluate on held-out test set
- [ ] Run 2–3 training iterations tuning LR and augmentation intensity
- [ ] Save best weights and threshold sweep results

*Expected training time per run: 2–3 hours on Kaggle T4*

---

### Phase 3 — Seizure Model Training
**Duration: 3–5 days | Risk: High**

- [ ] Same training loop as Phase 2 with seizure-specific differences
- [ ] 32-frame input, stride-2 sampling from 64 raw frames
- [ ] Apply aggressive seizure augmentation suite
- [ ] Stronger weight decay (1e-4), earlier stopping
- [ ] Run 3–5 training iterations
- [ ] Per-subject breakdown on test set to identify generalization failures

**If val F1 plateaus below 0.75 after 3 runs:**
1. Increase augmentation diversity (more occlusion patterns, stronger color shift)
2. Add label smoothing (0.1)
3. Do NOT change the architecture — tune the data pipeline only

*Expected training time per run: 1–2 hours on Kaggle T4*

---

### Phase 4 — Pipeline Integration
**Duration: 5–6 days | Risk: Medium**

- [ ] Set up TF memory growth before any model loading
- [ ] Rewrite `temporal_encoder.py` → `SharedFrameBuffer`
- [ ] Rewrite `fall_classifier.py` → single MoViNet-A2
- [ ] Rewrite `seizure_classifier.py` → single MoViNet-A2
- [ ] Rewrite `pipeline.py` → new inference stride logic, simultaneous event rule
- [ ] Update `config.yaml` with new model paths and parameters
- [ ] Update `demo_server.py` → Gemini fully async, TF GPU setup
- [ ] Update `requirements.txt`
- [ ] Verify YOLO + MoViNet GPU coexistence under full pipeline load
- [ ] Test CPU fallback for YOLO if VRAM pressure occurs

---

### Phase 5 — Testing and Stabilization
**Duration: 2–3 days | Risk: Low**

- [ ] Run full pipeline on test clips from both datasets
- [ ] Benchmark FPS — target > 15 FPS on GPU
- [ ] Test edge cases:
  - [ ] No person detected — buffer handles gracefully
  - [ ] Buffer not yet full — inference skipped cleanly
  - [ ] Rapid person movement — bbox jitter between frames
  - [ ] Both models fire simultaneously — fall priority rule works
  - [ ] Gemini API timeout — alert fires regardless
  - [ ] Warmup state displays correctly in demo UI
- [ ] Threshold calibration on validation set using real pipeline
- [ ] Final demo preparation with `demo_server.py`

---

## 9. Realistic Timeline

| Phase | Duration | Risk |
|---|---|---|
| Phase 1: Data Preparation | 2–3 days | Low |
| Phase 2: Fall Training | 2–3 days | Low |
| Phase 3: Seizure Training | 3–5 days | High |
| Phase 4: Integration | 5–6 days | Medium |
| Phase 5: Testing | 2–3 days | Low |
| **Total** | **14–20 days** | — |

The critical path risk is Phase 3. If the seizure model underperforms after 3–4 training runs, increase augmentation diversity — do not change the architecture or pivot to a different model.

---

## 10. Key Principles Applied From Past Failures

| Past Failure | Fix Applied in This Plan |
|---|---|
| YOLO used for action classification | YOLO11n kept strictly for cropping only — upgraded for better occlusion handling |
| 2D CNN on stacked frames as motion proxy | MoViNet-A2 processes native frame sequences |
| 15 models causing compute bottleneck | 2 models replace 15 |
| Gemini blocking alert path (2–5s delay) | Gemini moved fully async — never blocks alert |
| Dataset merging causing imbalance | Dedicated video-only datasets, no cross-domain merging |
| Clip-level splits inflating test metrics | Subject-level splits enforced throughout |
| Seizure classifier destroying temporal dynamics | Native frame sequence input preserves all temporal information |

---

*Plan version: Final — approved after full architectural review*
*Supersedes: EfficientNet-B0 ensemble pipeline (V1)*
