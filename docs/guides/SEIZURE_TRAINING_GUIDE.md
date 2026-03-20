# Seizure Detection Training Guide (V3 System)

## Overview
The V3 Seizure Detection system uses a **Dual-Stream Architecture**:
1.  **Motion Stream:** Analyzes rhythmic variance in keypoints.
2.  **Temporal Stream:** EfficientNet-B0 analyzing RGB frames.
3.  **Fusion:** Weighted average of both streams.

---

## 1. Data Preparation

### Step 1: Preprocess Videos
Extract frames and pose data from raw videos.
```bash
python scripts/prepare_seizure_classification.py
```
*   **Input:** `datasets/vision/raw/seizure/`
*   **Output:** `datasets/vision/seizure_classification/`
    *   `frames/`: Extracted RGB frames
    *   `pose/`: JSON pose keypoints

### Step 2: Create Splits
Stratified split (70/15/15) by VIDEO ID (not frame).
*   *Note:* This is handled automatically by the preprocessing script.
*   *Verification:* Check `datasets/vision/seizure_classification/splits.json`.

---

## 2. Training (Kaggle/Colab)

We use Jupyter Notebooks for training to leverage GPU acceleration.

### Notebook: `notebooks/train-seizure-v3-dual-stream.ipynb`

1.  **Upload Data:** Zip `datasets/vision/seizure_classification/` and upload to Kaggle.
2.  **Model Config:**
    *   `backbone`: EfficientNet-B0
    *   `pretrained`: True (ImageNet)
    *   `dropout`: 0.5
    *   `learning_rate`: 1e-4
3.  **Run Training:** Execute all cells. The notebook handles:
    *   Custom Dataset loading (Dual stream)
    *   Augmentations (Rotation, Noise)
    *   Training Loop (with Early Stopping)
    *   Model Checkpointing (`best_model.pt`)

---

## 3. Evaluation

After training, download `best_model.pt` and place it in `seizure_detection/seizure_v3_ensemble/`.

**Run Evaluation Script:**
```bash
python scripts/evaluate_on_dataset.py
```
**Expected Metrics:**
*   Recall: >95%
*   Precision: >85%
*   F1-Score: >90%

---

## 4. Troubleshooting

*   **Low Recall?** Check if `minor_jerks` are included in the negative class.
*   **Overfitting?** Increase dropout or enable `heavy_augmentation`.
