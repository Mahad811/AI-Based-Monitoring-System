# Seizure Detection: Why Results Stay Poor Despite Model Size and Epochs

**Status:** The seizure model gives consistently bad results. You have tried YOLOv8 small (n), medium (m), large (l) and trained for many epochs with no real improvement.  
**Conclusion:** The main issue is not hyperparameters or model size—it is a **poor fit between the task and the method** (single-frame YOLO).

---

## The core problem: task vs. method

**What seizure is:** A **temporal** event. It is convulsive or abnormal **movement over time** (seconds), not a fixed “object” or pose in one image.

**What YOLO does:** **Single-image** object detection. It looks at one frame and predicts boxes and classes. It does not see motion or sequences.

So we are asking: *“From this one static snapshot, is this person having a seizure?”*  
Often, that one snapshot (person in bed, same room, same pose) looks the same in normal and seizure clips. The **discriminative signal is in the motion**, which YOLO never sees. No amount of bigger YOLO or more epochs can add that information.

---

## Other factors that make it harder

**1. Frame-level labels are noisy**  
- Clips are labeled “seizure” or “normal” as a whole.  
- Every frame from a seizure clip gets the label “seizure.”  
- In a 5-second seizure clip, many frames are pre-ictal (before the seizure) or post-ictal (after), or just “person lying in bed”—so they look normal.  
- The model is trained to predict “seizure” from many frames that don’t actually look like seizure, which confuses learning.

**2. Single frame is weakly informative**  
- Same setting (ICU, bed, same person).  
- In one snapshot, “normal” and “seizure” often look very similar.  
- The real difference is **how** the person moves across frames, not a single pose.

**3. Dataset size**  
- ~806 clips, ~20k frames, ~50 patients.  
- For a subtle, temporal pattern and noisy frame labels, this is a tough learning problem even with a well-suited model.

---

## Why changing model size and epochs didn’t help

- **YOLO n / m / l:** All are single-frame detectors. A larger backbone might fit the training set a bit better, but it still never sees time or motion.  
- **More epochs:** The model can overfit to frame-level noise; it cannot learn “seizure = this kind of movement over time” from one image.  

So the ceiling is set by the **task formulation** (one frame → one label) and the **nature of the signal** (temporal), not by lack of capacity or training time.

---


## Short summary

| Aspect | Issue |
|--------|--------|
| **Task** | Seizure = temporal (movement over time). |
| **Method** | YOLO = single image, no time, no motion. |
| **Labels** | Clip-level → frame-level labels are noisy (many “seizure” frames don’t look like seizure). |
| **Result** | Poor results regardless of YOLO size or epochs; the bottleneck is the approach, not the optimizer. |
