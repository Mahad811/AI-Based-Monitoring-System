"""
Evaluate fall detection model on the merged YOLO test set with fixed confidence.
Outputs key metrics and a few concise visuals (confusion matrix, metrics bars,
confidence histogram, sample predictions).

Usage:
    python scripts/eval_fall_best.py
"""

from pathlib import Path
from collections import defaultdict
import json
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from ultralytics import YOLO

plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_PATH = Path("fall_detection/weights/best.pt")
TEST_IMAGES_DIR = Path("datasets/vision/yolo/test/images")
TEST_LABELS_DIR = Path("datasets/vision/yolo/test/labels")
# Optional dataset YAML (if present, used for mAP computation)
DATASET_YAML = Path("datasets/vision/yolo/dataset.yaml")
CONF_THRESHOLD = 0.25
MAX_SAMPLES = 2000  # sample this many test images (random, reproducible via seed)
OUTPUT_DIR = Path("fall_detection/report_eval")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_ground_truth(label_path: Path):
    if not label_path.exists():
        return []
    classes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                classes.append(int(parts[0]))
    return classes


def predict_image(model, image_path: Path, conf_threshold: float):
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False,
        save=False,
    )
    if not results or len(results) == 0:
        return []

    preds = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                preds.append(
                    {
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                    }
                )
    return preds


def calculate_metrics(results):
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = (correct / total * 100) if total > 0 else 0

    # Confusion-style counts
    fall_tp = sum(1 for r in results if r["gt_class"] == 1 and r["pred_class"] == 1)
    fall_fp = sum(1 for r in results if r["gt_class"] != 1 and r["pred_class"] == 1)
    fall_fn = sum(1 for r in results if r["gt_class"] == 1 and r["pred_class"] != 1)
    fall_tn = sum(1 for r in results if r["gt_class"] != 1 and r["pred_class"] != 1)

    normal_tp = sum(1 for r in results if r["gt_class"] == 0 and r["pred_class"] == 0)
    normal_fp = sum(1 for r in results if r["gt_class"] == 0 and r["pred_class"] != 0)
    normal_fn = sum(1 for r in results if r["gt_class"] == 0 and r["pred_class"] != 0)
    normal_tn = sum(1 for r in results if r["gt_class"] != 0 and r["pred_class"] != 0)

    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return prec * 100, rec * 100, f1 * 100

    fall_precision, fall_recall, fall_f1 = prf(fall_tp, fall_fp, fall_fn)
    normal_precision, normal_recall, normal_f1 = prf(normal_tp, normal_fp, normal_fn)

    avg_conf = (
        sum(r["confidence"] for r in results if r["pred_class"] is not None)
        / max(1, sum(1 for r in results if r["pred_class"] is not None))
    ) * 100

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "fall_tp": fall_tp,
        "fall_fp": fall_fp,
        "fall_fn": fall_fn,
        "fall_tn": fall_tn,
        "fall_precision": fall_precision,
        "fall_recall": fall_recall,
        "fall_f1": fall_f1,
        "normal_tp": normal_tp,
        "normal_fp": normal_fp,
        "normal_fn": normal_fn,
        "normal_tn": normal_tn,
        "normal_precision": normal_precision,
        "normal_recall": normal_recall,
        "normal_f1": normal_f1,
        "avg_confidence": avg_conf,
        "map50": None,
        "map50_95": None,
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("EVALUATE FALL MODEL (conf=0.25)")
    print("=" * 70)
    print(f"Model:   {MODEL_PATH}")
    print(f"Images:  {TEST_IMAGES_DIR}")
    print(f"Labels:  {TEST_LABELS_DIR}")
    print(f"Output:  {OUTPUT_DIR}")

    if not TEST_IMAGES_DIR.exists():
        raise FileNotFoundError(f"Test images dir not found: {TEST_IMAGES_DIR}")
    if not TEST_LABELS_DIR.exists():
        raise FileNotFoundError(f"Test labels dir not found: {TEST_LABELS_DIR}")

    all_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
    if MAX_SAMPLES and len(all_images) > MAX_SAMPLES:
        all_images = random.sample(all_images, MAX_SAMPLES)
        print(f"Sampling {len(all_images)} images from {len(list(TEST_IMAGES_DIR.glob('*')))} total (reproducible)")
    else:
        print(f"Using all {len(all_images)} test images")

    model = YOLO(str(MODEL_PATH))

    results = []
    for i, image_path in enumerate(all_images, 1):
        if i % 200 == 0:
            print(f"  Progress: {i}/{len(all_images)}")

        label_path = TEST_LABELS_DIR / (image_path.stem + ".txt")
        gt_classes = read_ground_truth(label_path)
        gt_class = gt_classes[0] if gt_classes else None

        preds = predict_image(model, image_path, CONF_THRESHOLD)
        preds_sorted = sorted(preds, key=lambda x: x["confidence"], reverse=True)
        pred_class = preds_sorted[0]["class_id"] if preds_sorted else None
        conf = preds_sorted[0]["confidence"] if preds_sorted else 0.0
        correct = (pred_class == gt_class) if (pred_class is not None and gt_class is not None) else False

        results.append(
            {
                "image_path": image_path,
                "image_name": image_path.name,
                "gt_class": gt_class,
                "pred_class": pred_class,
                "confidence": conf,
                "correct": correct,
            }
        )

    metrics = calculate_metrics(results)
    # Optional: compute mAP if dataset YAML exists
    if DATASET_YAML.exists():
        try:
            print("\n📊 Computing mAP via model.val (full test split)...")
            res_val = model.val(
                data=str(DATASET_YAML),
                split="test",
                conf=CONF_THRESHOLD,
                verbose=False,
            )
            metrics["map50"] = res_val.box.map50 * 100
            metrics["map50_95"] = res_val.box.map * 100
        except Exception as e:
            print(f"⚠️  mAP computation failed: {e}")

    print("\n📊 METRICS (conf=0.25)")
    for k in [
        "accuracy",
        "fall_precision",
        "fall_recall",
        "fall_f1",
        "normal_precision",
        "normal_recall",
        "normal_f1",
        "avg_confidence",
        "map50",
        "map50_95",
    ]:
        val = metrics.get(k)
        if val is None:
            print(f"  {k}: N/A")
        else:
            print(f"  {k}: {val:.2f}")

    # Save JSON/CSV (minimal)
    summary = {
        "model_path": str(MODEL_PATH),
        "dataset": str(DATASET_YAML) if DATASET_YAML.exists() else "N/A",
        "confidence": CONF_THRESHOLD,
        "metrics": metrics,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(summary, indent=2))

    df_results = pd.DataFrame(
        [
            {
                "image": r["image_name"],
                "ground_truth": "fall" if r["gt_class"] == 1 else "normal" if r["gt_class"] == 0 else "unknown",
                "prediction": "fall" if r["pred_class"] == 1 else "normal" if r["pred_class"] == 0 else "none",
                "confidence": r["confidence"],
                "correct": r["correct"],
            }
            for r in results
        ]
    )
    df_results.to_csv(OUTPUT_DIR / "results.csv", index=False)

    # -----------------------------------------------------------------------
    # Visualizations
    # -----------------------------------------------------------------------
    print("\n📊 Generating visuals...")

    # Confusion matrix
    gt = [r["gt_class"] for r in results if r["gt_class"] is not None]
    pred = [r["pred_class"] if r["pred_class"] is not None else -1 for r in results if r["gt_class"] is not None]
    cm = confusion_matrix(gt, pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Fall"],
        yticklabels=["Normal", "Fall"],
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (conf=0.25)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Metrics bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fall_vals = [metrics["fall_precision"], metrics["fall_recall"], metrics["fall_f1"]]
    normal_vals = [metrics["normal_precision"], metrics["normal_recall"], metrics["normal_f1"]]
    axes[0].bar(["Prec", "Recall", "F1"], fall_vals, color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[0].set_ylim(0, 100)
    axes[0].set_title("Fall Metrics")
    for i, v in enumerate(fall_vals):
        axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[1].bar(["Prec", "Recall", "F1"], normal_vals, color=["#3498db", "#2ecc71", "#e74c3c"])
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Normal Metrics")
    for i, v in enumerate(normal_vals):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold")
    plt.suptitle("Performance @ conf=0.25", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "performance_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Confidence histogram
    confidences = [r["confidence"] * 100 for r in results if r["confidence"] > 0]
    plt.figure(figsize=(8, 5))
    plt.hist(confidences, bins=30, color="#9b59b6", alpha=0.7, edgecolor="black")
    plt.axvline(metrics["avg_confidence"], color="red", linestyle="--", linewidth=2, label=f"Mean: {metrics['avg_confidence']:.1f}%")
    plt.xlabel("Confidence (%)")
    plt.ylabel("Frequency")
    plt.title("Confidence Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confidence_hist.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Sample predictions grid (12 samples)
    samples = results[:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    for idx, result in enumerate(samples):
        ax = axes[idx]
        img = cv2.imread(str(result["image_path"]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_results = model.predict(
            source=str(result["image_path"]),
            conf=CONF_THRESHOLD,
            verbose=False,
            save=False,
        )
        if pred_results and len(pred_results) > 0:
            annotated_img = pred_results[0].plot()
            img = annotated_img
        ax.imshow(img)
        ax.axis("off")
        gt_label = "Fall" if result["gt_class"] == 1 else "Normal"
        pred_label = (
            "Fall" if result["pred_class"] == 1 else "Normal" if result["pred_class"] == 0 else "None"
        )
        status = "✓" if result["correct"] else "✗"
        color = "green" if result["correct"] else "red"
        title = f"{status} GT: {gt_label}\nPred: {pred_label} ({result['confidence']:.2f})"
        ax.set_title(title, fontsize=9, color=color, fontweight="bold")
    plt.suptitle("Sample Predictions (conf=0.25)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\n✅ Done. Outputs in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()

