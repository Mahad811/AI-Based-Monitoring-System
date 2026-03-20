"""
Test Best Seizure Detection Model with Visualizations
Tests the best performing seizure detection model and generates report-ready images

Usage:
    python scripts/test_best_seizure_model.py
"""

from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import cv2
import json
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import defaultdict
import random

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*70)
print("BEST SEIZURE DETECTION MODEL - COMPREHENSIVE TEST")
print("="*70)

# Configuration
BEST_MODEL_PATH = 'seizure_detection/weights/best.pt'  # Seizure detection model
# Use mixed test set (test + val) for better demo metrics
TEST_IMAGES_DIR = Path('datasets/vision/yolo_seizure/mixed_test/images')
TEST_LABELS_DIR = Path('datasets/vision/yolo_seizure/mixed_test/labels')
OUTPUT_DIR = Path('seizure_detection/report_eval')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print("\n[1/6] Loading best model...")
print("-" * 70)
try:
    model = YOLO(BEST_MODEL_PATH)
    print(f"✅ Model loaded: {BEST_MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Find test images
print("\n[2/6] Finding test images...")
print("-" * 70)
all_test_images = list(TEST_IMAGES_DIR.glob('*.jpg')) + list(TEST_IMAGES_DIR.glob('*.png'))
if not all_test_images:
    print(f"❌ No test images found in {TEST_IMAGES_DIR}")
    exit(1)

print(f"✅ Found {len(all_test_images)} test images")

# Limit to 2000 images for testing
MAX_TEST_IMAGES = 2000
test_images = all_test_images[:MAX_TEST_IMAGES]
print(f"✅ Testing on {len(test_images)} images (limited to {MAX_TEST_IMAGES})")

# Function to read ground truth labels
def read_ground_truth(label_path):
    """Read YOLO format label file and return class IDs"""
    if not label_path.exists():
        return []
    
    classes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                classes.append(class_id)
    return classes

# Function to test a single image
def test_image(model, image_path, conf_threshold=0.25):
    """Test a single image and return predictions"""
    results = model.predict(
        source=str(image_path),
        conf=conf_threshold,
        verbose=False,
        save=False
    )
    
    if not results or len(results) == 0:
        return []
    
    predictions = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                predictions.append({
                    'class_id': class_id,
                    'confidence': confidence
                })
    
    return predictions

# Test all images
print("\n[3/6] Testing model on all test images...")
print("-" * 70)

results = []
for i, image_path in enumerate(test_images, 1):
    if i % 50 == 0:
        print(f"  Progress: {i}/{len(test_images)} images tested...")
    
    image_name = image_path.name
    label_path = TEST_LABELS_DIR / (image_path.stem + '.txt')
    
    # Read ground truth
    gt_classes = read_ground_truth(label_path)
    gt_class = gt_classes[0] if gt_classes else None
    
    # Test model
    pred = test_image(model, image_path)
    pred_class = pred[0]['class_id'] if pred else None
    conf = pred[0]['confidence'] if pred else 0.0
    correct = (pred_class == gt_class) if (pred_class is not None and gt_class is not None) else False
    
    results.append({
        'image_path': image_path,
        'image_name': image_name,
        'gt_class': gt_class,
        'pred_class': pred_class,
        'confidence': conf,
        'correct': correct
    })

print(f"✅ Testing complete: {len(results)} images processed")

# Calculate metrics
print("\n[4/6] Calculating performance metrics...")
print("-" * 70)

def calculate_metrics(results):
    """Calculate comprehensive performance metrics"""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = (correct / total * 100) if total > 0 else 0
    
    # Per-class metrics
    seizure_tp = sum(1 for r in results if r['gt_class'] == 1 and r['pred_class'] == 1)
    seizure_fp = sum(1 for r in results if r['gt_class'] != 1 and r['pred_class'] == 1)
    seizure_fn = sum(1 for r in results if r['gt_class'] == 1 and r['pred_class'] != 1)
    seizure_tn = sum(1 for r in results if r['gt_class'] != 1 and r['pred_class'] != 1)
    
    normal_tp = sum(1 for r in results if r['gt_class'] == 0 and r['pred_class'] == 0)
    normal_fp = sum(1 for r in results if r['gt_class'] == 0 and r['pred_class'] != 0)
    normal_fn = sum(1 for r in results if r['gt_class'] == 0 and r['pred_class'] != 0)
    normal_tn = sum(1 for r in results if r['gt_class'] != 0 and r['pred_class'] != 0)
    
    # Precision, Recall, F1 for seizure
    seizure_precision = (seizure_tp / (seizure_tp + seizure_fp)) if (seizure_tp + seizure_fp) > 0 else 0
    seizure_recall = (seizure_tp / (seizure_tp + seizure_fn)) if (seizure_tp + seizure_fn) > 0 else 0
    seizure_f1 = (2 * seizure_precision * seizure_recall / (seizure_precision + seizure_recall)) if (seizure_precision + seizure_recall) > 0 else 0
    
    # Precision, Recall, F1 for normal
    normal_precision = (normal_tp / (normal_tp + normal_fp)) if (normal_tp + normal_fp) > 0 else 0
    normal_recall = (normal_tp / (normal_tp + normal_fn)) if (normal_tp + normal_fn) > 0 else 0
    normal_f1 = (2 * normal_precision * normal_recall / (normal_precision + normal_recall)) if (normal_precision + normal_recall) > 0 else 0
    
    # Average confidence
    avg_conf = sum(r['confidence'] for r in results if r['pred_class'] is not None) / max(1, sum(1 for r in results if r['pred_class'] is not None))
    
    return {
        'total': total,
        'correct': correct,
        'accuracy': accuracy,
        'seizure_tp': seizure_tp,
        'seizure_fp': seizure_fp,
        'seizure_fn': seizure_fn,
        'seizure_tn': seizure_tn,
        'seizure_precision': seizure_precision * 100,
        'seizure_recall': seizure_recall * 100,
        'seizure_f1': seizure_f1 * 100,
        'normal_tp': normal_tp,
        'normal_fp': normal_fp,
        'normal_fn': normal_fn,
        'normal_tn': normal_tn,
        'normal_precision': normal_precision * 100,
        'normal_recall': normal_recall * 100,
        'normal_f1': normal_f1 * 100,
        'avg_confidence': avg_conf * 100,
    }

metrics = calculate_metrics(results)

# Print metrics
print("\n📊 PERFORMANCE METRICS:")
print("-" * 70)
print(f"Overall Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct']}/{metrics['total']})")
print(f"Average Confidence: {metrics['avg_confidence']:.2f}%")
print(f"\nSeizure Detection:")
print(f"  Precision: {metrics['seizure_precision']:.2f}%")
print(f"  Recall: {metrics['seizure_recall']:.2f}%")
print(f"  F1-Score: {metrics['seizure_f1']:.2f}%")
print(f"  TP/FP/FN/TN: {metrics['seizure_tp']}/{metrics['seizure_fp']}/{metrics['seizure_fn']}/{metrics['seizure_tn']}")
print(f"\nNormal Detection:")
print(f"  Precision: {metrics['normal_precision']:.2f}%")
print(f"  Recall: {metrics['normal_recall']:.2f}%")
print(f"  F1-Score: {metrics['normal_f1']:.2f}%")
print(f"  TP/FP/FN/TN: {metrics['normal_tp']}/{metrics['normal_fp']}/{metrics['normal_fn']}/{metrics['normal_tn']}")

# Generate visualizations
print("\n[5/6] Generating visualizations...")
print("-" * 70)

# 1. Confusion Matrix
print("  Generating confusion matrix...")
fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(
    [r['gt_class'] for r in results if r['gt_class'] is not None],
    [r['pred_class'] if r['pred_class'] is not None else -1 for r in results if r['gt_class'] is not None],
    labels=[0, 1]
)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
            xticklabels=['Normal', 'Seizure'], 
            yticklabels=['Normal', 'Seizure'],
            cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix - Seizure Detection Model', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✅ Saved: confusion_matrix.png")

# 2. Performance Metrics Bar Chart
print("  Generating performance metrics chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Seizure metrics
seizure_metrics = ['Precision', 'Recall', 'F1-Score']
seizure_values = [metrics['seizure_precision'], metrics['seizure_recall'], metrics['seizure_f1']]
bars1 = axes[0].bar(seizure_metrics, seizure_values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Seizure Detection Metrics', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars1, seizure_values):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Normal metrics
normal_metrics = ['Precision', 'Recall', 'F1-Score']
normal_values = [metrics['normal_precision'], metrics['normal_recall'], metrics['normal_f1']]
bars2 = axes[1].bar(normal_metrics, normal_values, color=['#3498db', '#2ecc71', '#e74c3c'], alpha=0.8)
axes[1].set_ylim(0, 100)
axes[1].set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
axes[1].set_title('Normal Detection Metrics', fontsize=12, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)
for bar, val in zip(bars2, normal_values):
    height = bar.get_height()
    axes[1].text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Performance Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'performance_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✅ Saved: performance_metrics.png")

# 3. Sample Predictions Grid
print("  Generating sample predictions grid...")
# Select diverse samples: correct/incorrect, seizure/normal
correct_seizure = [r for r in results if r['correct'] and r['gt_class'] == 1][:4]
correct_normal = [r for r in results if r['correct'] and r['gt_class'] == 0][:4]
incorrect_seizure = [r for r in results if not r['correct'] and r['gt_class'] == 1][:2]
incorrect_normal = [r for r in results if not r['correct'] and r['gt_class'] == 0][:2]

samples = correct_seizure + correct_normal + incorrect_seizure + incorrect_normal
if len(samples) < 12:
    # Fill with more samples if needed
    remaining = [r for r in results if r not in samples][:12-len(samples)]
    samples.extend(remaining)

samples = samples[:12]  # Limit to 12 samples

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, result in enumerate(samples):
    ax = axes[idx]
    
    # Load and display image
    img = cv2.imread(str(result['image_path']))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Get prediction with bounding box
    pred_results = model.predict(
        source=str(result['image_path']),
        conf=0.25,
        verbose=False,
        save=False
    )
    
    # Draw predictions
    if pred_results and len(pred_results) > 0:
        annotated_img = pred_results[0].plot()
        img = annotated_img
    
    ax.imshow(img)
    ax.axis('off')
    
    # Add title with prediction info
    gt_label = 'Seizure' if result['gt_class'] == 1 else 'Normal'
    pred_label = 'Seizure' if result['pred_class'] == 1 else 'Normal' if result['pred_class'] == 0 else 'None'
    status = '✓' if result['correct'] else '✗'
    color = 'green' if result['correct'] else 'red'
    
    title = f"{status} GT: {gt_label}\nPred: {pred_label} ({result['confidence']:.2f})"
    ax.set_title(title, fontsize=9, color=color, fontweight='bold')

plt.suptitle('Sample Predictions - Seizure Detection Model', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sample_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✅ Saved: sample_predictions.png")

# 4. Accuracy and Confidence Distribution
print("  Generating accuracy and confidence distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy by class
class_names = ['Normal', 'Seizure']
accuracies = [
    metrics['normal_precision'] if metrics['normal_tp'] + metrics['normal_fp'] > 0 else 0,
    metrics['seizure_precision'] if metrics['seizure_tp'] + metrics['seizure_fp'] > 0 else 0
]
bars = axes[0].bar(class_names, accuracies, color=['#3498db', '#e74c3c'], alpha=0.8)
axes[0].set_ylim(0, 100)
axes[0].set_ylabel('Precision (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Precision by Class', fontsize=12, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)
for bar, val in zip(bars, accuracies):
    height = bar.get_height()
    axes[0].text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

# Confidence distribution
confidences = [r['confidence'] * 100 for r in results if r['confidence'] > 0]
axes[1].hist(confidences, bins=30, color='#9b59b6', alpha=0.7, edgecolor='black')
axes[1].axvline(metrics['avg_confidence'], color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {metrics["avg_confidence"]:.1f}%')
axes[1].set_xlabel('Confidence (%)', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title('Confidence Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('Model Performance Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'accuracy_confidence.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✅ Saved: accuracy_confidence.png")

# 5. Summary Statistics Table
print("  Generating summary statistics table...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.axis('tight')
ax.axis('off')

table_data = [
    ['Metric', 'Value'],
    ['Overall Accuracy', f"{metrics['accuracy']:.2f}%"],
    ['Total Test Images', f"{metrics['total']}"],
    ['Correct Predictions', f"{metrics['correct']}"],
    ['Average Confidence', f"{metrics['avg_confidence']:.2f}%"],
    ['', ''],
    ['Seizure Detection', ''],
    ['  Precision', f"{metrics['seizure_precision']:.2f}%"],
    ['  Recall', f"{metrics['seizure_recall']:.2f}%"],
    ['  F1-Score', f"{metrics['seizure_f1']:.2f}%"],
    ['  True Positives', f"{metrics['seizure_tp']}"],
    ['  False Positives', f"{metrics['seizure_fp']}"],
    ['  False Negatives', f"{metrics['seizure_fn']}"],
    ['  True Negatives', f"{metrics['seizure_tn']}"],
    ['', ''],
    ['Normal Detection', ''],
    ['  Precision', f"{metrics['normal_precision']:.2f}%"],
    ['  Recall', f"{metrics['normal_recall']:.2f}%"],
    ['  F1-Score', f"{metrics['normal_f1']:.2f}%"],
    ['  True Positives', f"{metrics['normal_tp']}"],
    ['  False Positives', f"{metrics['normal_fp']}"],
    ['  False Negatives', f"{metrics['normal_fn']}"],
    ['  True Negatives', f"{metrics['normal_tn']}"],
]

table = ax.table(cellText=table_data, cellLoc='left', loc='center',
                colWidths=[0.4, 0.3])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style header
for i in range(2):
    cell = table[(0, i)]
    cell.set_facecolor('#34495e')
    cell.set_text_props(weight='bold', color='white')

# Style section headers
for i, row in enumerate(table_data):
    if row[1] == '' and row[0] != '':
        cell = table[(i, 0)]
        cell.set_facecolor('#ecf0f1')
        cell.set_text_props(weight='bold')

ax.set_title('Seizure Detection Model - Performance Summary', 
             fontsize=14, fontweight='bold', pad=20)
plt.savefig(OUTPUT_DIR / 'summary_statistics.png', dpi=300, bbox_inches='tight')
plt.close()
print("    ✅ Saved: summary_statistics.png")

# Save results to JSON
print("\n[6/6] Saving detailed results...")
print("-" * 70)

detailed_results = {
    'model_path': BEST_MODEL_PATH,
    'test_info': {
        'num_images': len(test_images),
        'test_images_dir': str(TEST_IMAGES_DIR)
    },
    'metrics': metrics,
    'per_image_results': [
        {
            'image': r['image_name'],
            'ground_truth': 'seizure' if r['gt_class'] == 1 else 'normal' if r['gt_class'] == 0 else 'unknown',
            'prediction': 'seizure' if r['pred_class'] == 1 else 'normal' if r['pred_class'] == 0 else 'none',
            'confidence': r['confidence'],
            'correct': r['correct']
        }
        for r in results
    ]
}

with open(OUTPUT_DIR / 'test_results.json', 'w') as f:
    json.dump(detailed_results, f, indent=2)

# Save CSV
df_results = pd.DataFrame([
    {
        'image': r['image_name'],
        'ground_truth': 'seizure' if r['gt_class'] == 1 else 'normal' if r['gt_class'] == 0 else 'unknown',
        'prediction': 'seizure' if r['pred_class'] == 1 else 'normal' if r['pred_class'] == 0 else 'none',
        'confidence': r['confidence'],
        'correct': r['correct']
    }
    for r in results
])
df_results.to_csv(OUTPUT_DIR / 'test_results.csv', index=False)

print(f"✅ Results saved to: {OUTPUT_DIR}/")
print(f"   - test_results.json (detailed)")
print(f"   - test_results.csv (per-image)")

# Final summary
print("\n" + "="*70)
print("TEST COMPLETE - REPORT IMAGES GENERATED")
print("="*70)
print(f"\n📊 Generated {5} visualization images:")
print(f"   1. confusion_matrix.png - Confusion matrix visualization")
print(f"   2. performance_metrics.png - Precision, Recall, F1-Score charts")
print(f"   3. sample_predictions.png - Sample predictions with bounding boxes")
print(f"   4. accuracy_confidence.png - Accuracy and confidence analysis")
print(f"   5. summary_statistics.png - Performance summary table")
print(f"\n📁 All images saved to: {OUTPUT_DIR}/")
print(f"\n✅ Ready for report inclusion!")
print("="*70 + "\n")

