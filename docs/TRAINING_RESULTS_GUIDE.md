# Training Results Guide

## 📊 What You'll Get After Training

Both **Fall Detection** (EfficientNet) and **Seizure Detection** (Dual-Stream) output similar artifacts.

### 1. Model Files (`weights/`)
*   `best.pt`: The model checkpoint with the highest validation F1-score. **(Use this for deployment)**
*   `last.pt`: The checkpoint from the final epoch (useful if you want to resume training).
*   `fold*.pt`: (For Ensembles) The `best.pt` for each cross-validation fold.

### 2. Metrics File (`metrics.json`)
Contains the final performance numbers:
```json
{
  "test_loss": 0.3421,
  "test_accuracy": 0.8650,
  "test_precision": 0.8210,
  "test_recall": 0.8540,
  "test_f1": 0.8372
}
```

### 3. Plots (`plots/`)
*   `loss_curve.png`: Training vs Validation Loss. (Should go down and converge).
*   `accuracy_curve.png`: Training vs Validation Accuracy. (Should go up).
*   `confusion_matrix.png`: Visualizes misclassifications.

---

## 📈 Interpreting the Metrics

### 1. Classification Report
We focus on **Recall** (Sensitivity) because missing a critical event is worse than a false alarm.

| Metric | Target (Fall) | Target (Seizure) | Meaning |
|:---|:---|:---|:---|
| **Recall** | > 85% | > 95% | Out of all actual events, how many did we catch? |
| **Precision** | > 75% | > 85% | Out of all alerts, how many were real? |
| **F1-Score** | > 80% | > 90% | Balance between Recall and Precision. |

### 2. Confusion Matrix
Shows exactly where the model is confused.

```
                 Predicted
              Normal   Fall
Actual Normal   TN      FP  (False Alarms)
Actual Fall     FN      TP  (Correct Detections)
                ^
          (Missed Events - BAD!)
```

**Goal:** Minimize **FN** (False Negatives). High **FP** (False Positives) can be tolerated if the Cognitive Core filters them out later.

---

## 🔍 Common Issues & Fixes

### Issue: High Training Accuracy, Low Validation Accuracy (Overfitting)
*   **Cause:** Model memorizing data.
*   **Fix:** Increase `dropout` (e.g., 0.5), use `early_stopping`, or enable `data_augmentation`.

### Issue: Low Recall (Missing Falls/Seizures)
*   **Cause:** Imbalanced dataset (too many normal samples).
*   **Fix:** Use `weighted_loss` (give more weight to the minority class) or oversample the minority class.

### Issue: Loss not decreasing (Underfitting)
*   **Cause:** Learning rate too high/low, or model too simple.
*   **Fix:** Adjust `learning_rate` (try 1e-4), unfreeze more layers in the backbone.

---

## 🚀 deployment

1.  **Select Best Model:** Pick the `best.pt` (or `fold*.pt` ensemble) with the highest **Validation F1**.
2.  **Move to Project:**
    *   Falls: `d:/project/FYP/fall_detection/fall_v2_ensemble/`
    *   Seizures: `d:/project/FYP/seizure_detection/seizure_v3_ensemble/`
3.  **Update Config:** Ensure `config.yaml` points to the correct directory.
