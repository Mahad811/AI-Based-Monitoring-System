# Fall Detection V4: Motion-Only Encoding (Optimized for Falls)

**Date:** February 7, 2026  
**Status:** Implementation Complete  
**Version:** V4 - Motion-Only Optimized

---

## Overview

Fall Detection V4 implements **motion-only encoding** optimized specifically for fall dynamics. This approach prevents appearance shortcuts (e.g., "horizontal person = fall") and forces the model to learn motion trajectories.

**Key Innovation:** Unlike seizure detection (which uses rhythmic motion patterns), falls are characterized by **rapid acceleration** followed by stillness. V4 captures this with an acceleration pattern channel instead of rhythmicity.

---

## Architecture

### Motion-Only Encoding (12-Frame Window)

**Window Size:** 12 frames (~0.4 seconds at 30fps)
- Captures full fall trajectory
- Avoids post-fall stillness that dilutes signal
- Optimal balance between temporal context and signal purity

**Channel Encoding:**
- **R Channel:** Mean of absolute frame differences (motion intensity)
- **G Channel:** Acceleration pattern (rate of change in motion - captures fall dynamics)
- **B Channel:** Max of absolute frame differences (peak motion burst)

**Key Difference from Seizure V3:**
- Seizure V3: R=mean, G=std (rhythmicity), B=max
- Fall V4: R=mean, G=acceleration, B=max
- **Why:** Falls are NOT rhythmic - they're rapid acceleration events. Acceleration pattern (second-order motion) captures this better than std.

---

## Why Motion-Only?

### Problem with Previous Approaches

1. **V1-V2 (3-frame RGB stacking):**
   - Included appearance (current frame)
   - Model could cheat: "horizontal person = fall"
   - Short temporal window (~0.1s) missed full trajectory

2. **V3 (5-frame motion-emphasized):**
   - Still included appearance in G channel
   - Better temporal context but still allowed shortcuts

### Solution: Motion-Only Encoding

**Benefits:**
- ✅ **No appearance shortcuts:** Model MUST learn motion patterns
- ✅ **Longer temporal window:** 12 frames captures full fall trajectory
- ✅ **Fall-optimized:** Acceleration pattern matches fall dynamics (rapid acceleration, not rhythmic)
- ✅ **Robust:** Less sensitive to lighting, clothing, background

**Trade-offs:**
- ⚠️ Requires retraining (models trained on V1-V3 won't work)
- ⚠️ Slightly longer warmup (12 frames vs 5 frames)

---

## Implementation Details

### Preprocessing (`scripts/prepare_fall_classification.py`)

**Method:** `create_motion_only_rgb(frames_window, bbox)`

1. **Frame Extraction:**
   - Extract 12-frame window centered on detection frame
   - Crop all frames using same bbox (preserves relative motion)

2. **Motion Computation:**
   ```python
   # First-order differences
   diffs[i] = |frame[i] - frame[i-1]|  # Shape: (11, H, W)
   
   # R channel: Mean motion intensity
   R = mean(diffs, axis=0)
   
   # G channel: Acceleration pattern (second-order)
   accel[i] = |diffs[i+1] - diffs[i]|  # Rate of change in motion
   G = mean(accel, axis=0)
   
   # B channel: Peak motion burst
   B = max(diffs, axis=0)
   ```

3. **Normalization:**
   - Per-channel contrast stretching to [0, 255]
   - Ensures full dynamic range utilization

### Runtime (`visual_guardian/temporal_encoder.py`)

**Class:** `TemporalEncoder`

- **Buffer:** 12-frame rolling buffer (`deque(maxlen=12)`)
- **Encoding:** Same motion-only encoding as preprocessing
- **Output:** (224, 224, 3) motion-only RGB image

### Configuration (`config/config.yaml`)

```yaml
temporal_encoder:
  frame_size: 224
  buffer_size: 12  # V4: motion-only, optimized for falls
```

---

## Expected Performance

**Baseline (V2 Ensemble):**
- F1: 0.78-0.84
- Recall: 0.80-0.83
- Precision: 0.72-0.75

**Expected V4 Improvement:**
- **F1: 0.82-0.88** (+2-4% absolute)
- **Recall: 0.85-0.90** (better fall detection)
- **Precision: 0.78-0.85** (fewer false alarms)

**Why Improvement Expected:**
1. Motion-only prevents appearance shortcuts → better generalization
2. Acceleration pattern captures fall dynamics → better discrimination
3. Longer window captures full trajectory → better recall

---

## Migration Guide

### For Preprocessing

**Old (V3):**
```bash
python scripts/prepare_fall_classification.py
# Generated 5-frame motion-emphasized RGB
```

**New (V4):**
```bash
python scripts/prepare_fall_classification.py
# Generates 12-frame motion-only RGB
# Same command, different encoding
```

**Note:** V4 preprocessing will overwrite V3 dataset. If you need both, use different output directories.

### For Runtime

**No code changes needed** - `VisionPipeline` automatically uses new encoding via `TemporalEncoder`.

**Configuration:**
- Update `config.yaml` → `buffer_size: 12` (already done)

### For Training

**Important:** Models trained on V1-V3 will NOT work with V4 encoding.

**Steps:**
1. Run V4 preprocessing to generate new dataset
2. Train new models on V4 dataset
3. Use V4 models for inference

---

## Technical Details

### Acceleration Pattern Computation

The acceleration pattern (G channel) captures how motion is changing:

```python
# First-order: motion between consecutive frames
diffs = [|frame[i] - frame[i-1]| for i in 1..N]

# Second-order: how motion is changing
accel = [|diffs[i+1] - diffs[i]| for i in 0..N-2]

# Average acceleration pattern
G = mean(accel, axis=0)
```

**Why This Works for Falls:**
- Falls: Rapid acceleration (high accel) → stillness (low accel)
- Normal: Steady motion (low accel variance)
- The acceleration pattern distinguishes these dynamics

### Window Size Rationale

**12 frames chosen because:**
- Falls typically last 0.5-1 second
- 12 frames = ~0.4 seconds at 30fps
- Captures active fall motion without post-fall stillness
- Balance between temporal context and signal purity

**Comparison:**
- 3 frames (~0.1s): Too short, misses trajectory
- 5 frames (~0.17s): Better but still short
- 12 frames (~0.4s): Optimal for falls
- 20 frames (~0.67s): Too long, includes stillness
- 60 frames (2s): Designed for seizures, not falls

---

## Files Changed

1. **`scripts/prepare_fall_classification.py`**
   - New: `create_motion_only_rgb()` method
   - Updated: `process_video()` for 12-frame windows
   - Updated: Documentation and config tracking

2. **`visual_guardian/temporal_encoder.py`**
   - Updated: `encode()` method for motion-only encoding
   - Updated: Buffer size to 12 frames
   - New: `normalize_channel()` helper method

3. **`config/config.yaml`**
   - Updated: `buffer_size: 12`

4. **`visual_guardian/fall_classifier.py`**
   - Updated: Documentation for V4

5. **`visual_guardian/__init__.py`**
   - Updated: Module documentation

---

## Next Steps

1. **Run Preprocessing:**
   ```bash
   python scripts/prepare_fall_classification.py
   ```

2. **Train Models:**
   - Use existing V2 training notebook (architecture unchanged)
   - Train on new V4 dataset
   - Expected: Better F1, recall, precision

3. **Evaluate:**
   - Compare V4 vs V2 performance
   - Verify motion-only prevents appearance shortcuts
   - Check acceleration pattern discriminative power

---

## References

- **Seizure V3:** Motion-only encoding for rhythmic patterns (60 frames, std channel)
- **Fall V4:** Motion-only encoding for acceleration patterns (12 frames, acceleration channel)
- **Key Insight:** Different motion patterns require different encodings

---

**Status:** ✅ Implementation Complete - Ready for Preprocessing and Training
