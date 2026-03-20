# Complete Fall Detection Approaches: All Versions

**Date:** February 7, 2026  
**Status:** Comprehensive Comparison Document

---

## Evolution Timeline

| Version | Approach | Window | Encoding | Performance | Status |
|---------|----------|--------|----------|-------------|--------|
| **V1** | YOLOv8 Single-Frame | 1 frame | Raw RGB | mAP50: 88.19% | Legacy |
| **V2** | Temporal RGB Triplets | 3 frames | RGB stacking | F1: 0.78-0.84 | Current baseline |
| **V3** | Motion-Emphasized | 5 frames | Motion + appearance | Not trained | Intermediate |
| **V4** | Motion-Only Optimized | 12 frames | Motion-only | Expected F1: 0.82-0.88 | **Current** |

---

## V1: Original YOLOv8 Single-Frame Detection

### Approach
- **Model:** YOLOv8n (nano variant)
- **Input:** Single RGB frame (640×480)
- **Method:** Direct object detection (fall vs normal)
- **Classes:** 0 = Normal, 1 = Fall

### Characteristics
- ✅ Single-frame classification
- ❌ No temporal context
- ⚠️ Appearance-based (learns "fall pose" vs "normal pose")
- ✅ Fast inference

### Performance
- **mAP50:** 88.19%
- **Fall Recall:** 100% (no missed falls on test set)
- **Overall Accuracy:** 80.6%
- **Weights:** `fall_detection/weights/best.pt`

### Limitations
- ❌ No temporal motion information
- ❌ May miss falls that don't look like "fall pose" in single frame
- ❌ Can confuse lying down (normal) with fallen (fall)

### Code Location
- Original YOLOv8 training
- Replaced by temporal approaches

---

## V2: Temporal RGB Triplets (Ensemble)

### Approach
- **Model:** EfficientNet-B0
- **Input:** Temporal RGB image (224×224×3)
- **Window:** 3 consecutive frames
- **Encoding:** RGB stacking

### Encoding Method
```
R channel = grayscale(frame[t-1])  # Past frame
G channel = grayscale(frame[t])     # Current frame (APPEARANCE)
B channel = grayscale(frame[t+1])  # Future frame
```

### Characteristics
- ✅ Temporal context: ~0.1 seconds (3 frames @ 30fps)
- ⚠️ Includes appearance (G channel = current frame)
- ✅ Motion visible through color differences
- ✅ 5-fold ensemble for robustness

### Training Strategy
- 5-fold video-level cross-validation
- Partial backbone freeze (blocks 5-6 only)
- Strong regularization (dropout 0.5, drop_path 0.2)
- Square-root class balancing
- MixUp augmentation (alpha=0.4)
- Test-Time Augmentation (horizontal flip)

### Performance
- **Val F1:** 0.70-0.78 per fold
- **Ensemble Test F1:** 0.78-0.84
- **Test Recall:** 0.80-0.83
- **Test Precision:** 0.72-0.75

### Limitations
- ❌ Short temporal window (~0.1s) misses full fall trajectory
- ❌ Appearance shortcut: model can learn "horizontal person = fall"
- ⚠️ G channel contains appearance, not pure motion

### Code Location
- **Preprocessing:** `scripts/prepare_fall_classification.py` (old version)
- **Training:** `notebooks/train-fall-classifier-v2-kfold.ipynb`
- **Runtime:** `visual_guardian/temporal_encoder.py` (old version with buffer_size=3)

---

## V3: Motion-Emphasized (5-Frame)

### Approach
- **Model:** EfficientNet-B0 (same as V2)
- **Input:** Temporal RGB image (224×224×3)
- **Window:** 5 consecutive frames
- **Encoding:** Motion-emphasized RGB

### Encoding Method
```
R channel = backward motion (|frame[t-1] - frame[t-2]|)  # Motion
G channel = current frame[t]                              # APPEARANCE (still present!)
B channel = forward motion (|frame[t+2] - frame[t+1]|)  # Motion
```

### Characteristics
- ✅ Temporal context: ~0.17 seconds (5 frames @ 30fps)
- ✅ Motion channels in R and B
- ⚠️ Still includes appearance in G channel
- ✅ Better temporal context than V2

### Performance
- ⚠️ Not fully trained/tested
- Expected similar to V2 but with better temporal context

### Limitations
- ❌ Still allows appearance shortcuts (G channel = current frame)
- ⚠️ Window may still be short for full fall trajectory
- ⚠️ Motion channels normalized but appearance remains

### Code Location
- Implemented but replaced by V4
- Was intermediate step between V2 and V4

---

## V4: Motion-Only Optimized (Current)

### Approach
- **Model:** EfficientNet-B0 (same architecture)
- **Input:** Motion-only RGB image (224×224×3)
- **Window:** 12 consecutive frames
- **Encoding:** Pure motion (no appearance)

### Encoding Method
```python
# First-order differences
diffs[i] = |frame[i] - frame[i-1]|  # Shape: (11, H, W)

# R channel: Mean motion intensity
R = mean(diffs, axis=0)

# G channel: Acceleration pattern (second-order motion)
accel[i] = |diffs[i+1] - diffs[i]|  # Rate of change in motion
G = mean(accel, axis=0)

# B channel: Peak motion burst
B = max(diffs, axis=0)
```

### Characteristics
- ✅ Temporal context: ~0.4 seconds (12 frames @ 30fps)
- ✅ **NO appearance information** - prevents shortcuts
- ✅ Acceleration pattern optimized for falls (rapid acceleration, not rhythmic)
- ✅ Captures full fall trajectory without post-fall stillness

### Why Acceleration Pattern?
- **Falls:** Rapid acceleration → stillness (high accel variance)
- **Normal:** Steady motion (low accel variance)
- **Different from seizures:** Seizures are rhythmic, falls are acceleration events

### Performance (Expected)
- **F1:** 0.82-0.88 (+2-4% over V2)
- **Recall:** 0.85-0.90 (better fall detection)
- **Precision:** 0.78-0.85 (fewer false alarms)

### Advantages
- ✅ **No appearance shortcuts** - model MUST learn motion
- ✅ Longer window captures full trajectory
- ✅ Fall-optimized acceleration pattern
- ✅ More robust to lighting/clothing/background

### Trade-offs
- ⚠️ Requires retraining (V1-V3 models won't work)
- ⚠️ Longer warmup (12 frames vs 3-5 frames)
- ⚠️ Slightly more computation (12 frames vs 3-5)

### Code Location
- **Preprocessing:** `scripts/prepare_fall_classification.py` (current)
- **Runtime:** `visual_guardian/temporal_encoder.py` (current, buffer_size=12)
- **Config:** `config/config.yaml` (buffer_size: 12)
- **Documentation:** `docs/FALL_V4_MOTION_ONLY.md`

---

## Detailed Comparison Table

| Aspect | V1 (YOLOv8) | V2 (3-Frame) | V3 (5-Frame) | V4 (12-Frame Motion-Only) |
|--------|-------------|--------------|--------------|----------------------------|
| **Window** | 1 frame | 3 frames | 5 frames | 12 frames |
| **Duration** | ~0.03s | ~0.1s | ~0.17s | ~0.4s |
| **Encoding** | Raw RGB | RGB stacking | Motion + appearance | Motion-only |
| **Appearance** | ✅ Yes | ✅ Yes (G channel) | ✅ Yes (G channel) | ❌ No |
| **Motion Info** | ❌ No | ⚠️ Implicit | ✅ Explicit (R, B) | ✅ Explicit (R, G, B) |
| **Model** | YOLOv8n | EfficientNet-B0 | EfficientNet-B0 | EfficientNet-B0 |
| **Ensemble** | ❌ No | ✅ 5-fold | ✅ 5-fold | ✅ 5-fold |
| **F1 Score** | N/A* | 0.78-0.84 | Not tested | 0.82-0.88 (expected) |
| **Shortcuts** | ✅ Yes | ✅ Yes | ✅ Yes | ❌ No |
| **Fall Dynamics** | ❌ No | ⚠️ Limited | ⚠️ Limited | ✅ Optimized |

*V1 uses mAP50 metric, not F1

---

## Key Insights

### Why V4 is Better

1. **Motion-only prevents appearance shortcuts**
   - V1-V3: Can learn "horizontal person = fall"
   - V4: Must learn motion trajectory

2. **Acceleration pattern matches fall dynamics**
   - Falls: Rapid acceleration (not rhythmic like seizures)
   - G channel captures this specifically

3. **Optimal window size**
   - 12 frames captures full trajectory
   - Avoids post-fall stillness
   - Balance between context and signal purity

### Evolution Rationale

- **V1 → V2:** Added temporal context
- **V2 → V3:** Increased window, emphasized motion
- **V3 → V4:** Removed appearance, optimized for fall dynamics

---

## Current Status

- **V1:** Legacy (YOLOv8 single-frame)
- **V2:** Baseline (3-frame temporal RGB, F1: 0.78-0.84)
- **V3:** Intermediate (not fully deployed)
- **V4:** Current (12-frame motion-only, ready for training)

---

## Next Steps for V4

1. **Run Preprocessing:**
   ```bash
   python scripts/prepare_fall_classification.py
   ```

2. **Train Models:**
   - Use V2 training notebook (architecture unchanged)
   - Train on V4 dataset
   - Expected improvement: +2-4% F1

3. **Evaluate:**
   - Compare V4 vs V2 performance
   - Verify motion-only prevents shortcuts
   - Validate acceleration pattern effectiveness

---

## Summary

**V4 (Current)** implements motion-only encoding with 12-frame window and acceleration pattern optimized for fall dynamics. This removes appearance shortcuts and should improve performance over previous versions.

---

**Document Status:** ✅ Complete - All versions documented
