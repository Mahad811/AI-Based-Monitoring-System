# Fall Detection Training Guide (V2 System)

## Overview
The V2 Fall Detection system uses a **5-Fold Ensemble Architecture**:
1.  **Input:** Temporal RGB Triplets (frames t-1, t, t+1) stacked as channels.
2.  **Model:** EfficientNet-B0 (Pretrained).
3.  **Strategy:** 5-Fold Cross-Validation (Video-level split) to handle robust generalization.

---

## 1. Data Preparation

### Step 1: Preprocess Videos
Extract temporal RGB triplets (stacking 3 frames) from raw videos.
```bash
python scripts/prepare_visual_datasets.py
```
*   **Input:** `datasets/vision/raw/falls/` and `datasets/vision/raw/adls/`
*   **Output:** `datasets/vision/fall_classification/triplets/`
    *   Images are saved as `.jpg` sets.

### Step 2: Create Splits
Because we use K-Fold, we don't need a single fixed train/test split for training, but we do need a **Test Set** held out entirely.
*   The script automatically reserves 15% of videos for the `test` set.
*   The remaining 85% are used for 5-fold cross-validation (Train/Val) in the notebook.

---

## 2. Training (Kaggle/Colab)

We use Jupyter Notebooks for training to leverage GPU acceleration.

### Notebook: `notebooks/train-fall-classifier-v2-kfold.ipynb`

1.  **Upload Data:** Zip `datasets/vision/fall_classification/` and upload to Kaggle.
2.  **Model Config:**
    *   `backbone`: EfficientNet-B0
    *   `dropout`: 0.5 (Heavy regularization)
    *   `k_folds`: 5
3.  **Run Training:** Execute all cells. The notebook handles:
    *   Training 5 separate models (Fold 0 to Fold 4)
    *   Saving `foldX.pt` for each fold
    *   Generating Ensemble Metrics

---

## 3. Deployment

After training, you will have 5 model files (`fold0.pt` ... `fold4.pt`).

1.  **Place Models:** Put all 5 files in:
    `fall_detection/fall_v2_ensemble/`
2.  **Update Config:**
    ```yaml
    fall_classifier:
      model: fall_detection/fall_v2_ensemble/  # Directory path
      threshold: 0.64
    ```

---

## 4. Evaluation

**Run Evaluation Script:**
```bash
python scripts/evaluate_fall_detection.py
```
**Expected Metrics:**
*   Recall (Video-Level): ~85%
*   F1-Score (Video-Level): ~83%
*   Precision: ~80%

---

## 5. Troubleshooting

*   **Low Precision?** Increase the `threshold` in `config.yaml`.
*   **Missing Detections?** Decrease `threshold` or increase `window_size`.
