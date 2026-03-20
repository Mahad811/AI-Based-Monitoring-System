# Fall Detection: Dataset Imbalance Summary

**Status:** You trained the fall model on **yolo_merged** (merged dataset).  
**Problem:** The merged dataset is heavily **imbalanced** (far more “normal” than “fall”), which degrades model performance—especially **fall recall** (missed falls).

---

## How the merged dataset was built

1. **Base fall dataset (`yolo`)**  
   - Built from `create_splits.py` + `prepare_yolo_dataset.py`.  
   - Source videos: **3,140 fall** and **3,848 normal** (more normal than fall from the start).  
   - 10 frames per video → ~31k fall frames, ~38k normal frames (**~55% normal, ~45% fall**).

2. **Merge step (`merge_slp_with_falls.py`)**  
   - Takes `datasets/vision/yolo` and adds **SLP (Simulated Lying Postures)** data.  
   - **All 13,770 SLP images are added as “normal” only** (no fall class).  
   - Output: `datasets/vision/yolo_merged/`.

3. **Result**  
   - **yolo_merged** = base (already normal-heavy) + **13,770 extra normal** images.  
   - Training set ends up with **far more normal than fall** examples.

---

## Why this hurts the model

- The model is trained with many more “normal” than “fall” examples.  
- It learns to predict “normal” more often.  
- **Fall recall drops** (more missed falls); you may also get more false “normal” predictions on fall frames.  
- Your eval showed **fall recall ~75.6%** (about 1 in 4 falls missed), which matches this kind of imbalance.

---

## What to do next

- **Rebalance** the training data (e.g. equal frame counts for fall vs normal, or use class weights / balanced sampling).  
- Prefer training on a **balanced** set (or **yolo** only with balancing) rather than the current **yolo_merged** as-is.  
- See the main analysis (chat/documentation) for concrete rebalancing options and scripts.

---

*Short summary for fall model improvement: train on balanced data or correct for imbalance when using yolo_merged.*
