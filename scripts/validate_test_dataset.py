"""
Comprehensive Test Dataset Validation Script
Checks all aspects of data quality before training/testing

Usage:
    python scripts/validate_test_dataset.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
from PIL import Image
import cv2
from collections import defaultdict, Counter
import json
import random

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*70)
print("TEST DATASET VALIDATION - COMPREHENSIVE CHECK")
print("="*70)

# Configuration
TEST_IMAGES_DIR = Path('datasets/vision/yolo/test/images')
TEST_LABELS_DIR = Path('datasets/vision/yolo/test/labels')
OUTPUT_DIR = Path('fall_detection/data_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {0: 'normal', 1: 'fall'}

# ============================================================================
# CHECK 1: Directory Structure and File Counts
# ============================================================================
print("\n[1/7] Checking directory structure and file counts...")
print("-" * 70)

if not TEST_IMAGES_DIR.exists():
    print(f"❌ ERROR: Test images directory not found: {TEST_IMAGES_DIR}")
    exit(1)
else:
    print(f"✅ Test images directory exists: {TEST_IMAGES_DIR}")

if not TEST_LABELS_DIR.exists():
    print(f"❌ ERROR: Test labels directory not found: {TEST_LABELS_DIR}")
    exit(1)
else:
    print(f"✅ Test labels directory exists: {TEST_LABELS_DIR}")

# Count files
image_files = sorted(list(TEST_IMAGES_DIR.glob('*.jpg')) + list(TEST_IMAGES_DIR.glob('*.png')))
label_files = sorted(list(TEST_LABELS_DIR.glob('*.txt')))

print(f"\n📊 File counts:")
print(f"   Images: {len(image_files)}")
print(f"   Labels: {len(label_files)}")

if len(image_files) == 0:
    print("❌ ERROR: No images found!")
    exit(1)

if len(label_files) == 0:
    print("❌ ERROR: No label files found!")
    exit(1)

# ============================================================================
# CHECK 2: Image-Label Pairing
# ============================================================================
print("\n[2/7] Checking image-label pairing...")
print("-" * 70)

image_stems = {img.stem for img in image_files}
label_stems = {lbl.stem for lbl in label_files}

images_without_labels = image_stems - label_stems
labels_without_images = label_stems - image_stems

print(f"✅ Images with labels: {len(image_stems & label_stems)}")

if images_without_labels:
    print(f"⚠️  Images without labels: {len(images_without_labels)}")
    print(f"   First 10: {list(images_without_labels)[:10]}")
else:
    print(f"✅ All images have corresponding label files")

if labels_without_images:
    print(f"⚠️  Labels without images: {len(labels_without_images)}")
    print(f"   First 10: {list(labels_without_images)[:10]}")
else:
    print(f"✅ All labels have corresponding image files")

# Use only paired files
paired_stems = sorted(image_stems & label_stems)
print(f"\n✅ Total valid pairs: {len(paired_stems)}")

# ============================================================================
# CHECK 3: Label Format and Class Distribution
# ============================================================================
print("\n[3/7] Analyzing label format and class distribution...")
print("-" * 70)

class_counts = Counter()
bboxes_per_image = []
invalid_labels = []
empty_labels = []
label_format_errors = []

for stem in paired_stems:
    label_path = TEST_LABELS_DIR / f"{stem}.txt"
    
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        if not lines or all(not line.strip() for line in lines):
            empty_labels.append(stem)
            bboxes_per_image.append(0)
            continue
        
        bbox_count = 0
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            
            # Validate format: class_id x_center y_center width height
            if len(parts) != 5:
                label_format_errors.append({
                    'file': stem,
                    'line': line_num,
                    'content': line,
                    'error': f'Expected 5 values, got {len(parts)}'
                })
                continue
            
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                
                # Validate class ID
                if class_id not in CLASS_NAMES:
                    invalid_labels.append({
                        'file': stem,
                        'line': line_num,
                        'class_id': class_id,
                        'error': f'Invalid class ID (expected 0 or 1)'
                    })
                    continue
                
                # Validate bbox coordinates (should be normalized 0-1)
                if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and 
                       0 <= width <= 1 and 0 <= height <= 1):
                    invalid_labels.append({
                        'file': stem,
                        'line': line_num,
                        'error': f'Coordinates out of range [0,1]: {x_center:.3f}, {y_center:.3f}, {width:.3f}, {height:.3f}'
                    })
                    continue
                
                class_counts[class_id] += 1
                bbox_count += 1
                
            except ValueError as e:
                label_format_errors.append({
                    'file': stem,
                    'line': line_num,
                    'content': line,
                    'error': f'Parse error: {e}'
                })
        
        bboxes_per_image.append(bbox_count)
        
    except Exception as e:
        label_format_errors.append({
            'file': stem,
            'line': 0,
            'error': f'File read error: {e}'
        })

# Report results
print("\n📊 Class Distribution:")
total_boxes = sum(class_counts.values())
for class_id in sorted(CLASS_NAMES.keys()):
    count = class_counts[class_id]
    percentage = (count / total_boxes * 100) if total_boxes > 0 else 0
    print(f"   Class {class_id} ({CLASS_NAMES[class_id]:8s}): {count:5d} boxes ({percentage:5.2f}%)")

print(f"\n📊 Bounding Boxes per Image:")
print(f"   Mean:   {np.mean(bboxes_per_image):.2f}")
print(f"   Median: {np.median(bboxes_per_image):.0f}")
print(f"   Min:    {np.min(bboxes_per_image):.0f}")
print(f"   Max:    {np.max(bboxes_per_image):.0f}")

if empty_labels:
    print(f"\n⚠️  Empty label files: {len(empty_labels)}")
    print(f"   First 10: {empty_labels[:10]}")

if invalid_labels:
    print(f"\n⚠️  Invalid labels found: {len(invalid_labels)}")
    for err in invalid_labels[:10]:
        print(f"   {err}")

if label_format_errors:
    print(f"\n⚠️  Label format errors: {len(label_format_errors)}")
    for err in label_format_errors[:10]:
        print(f"   {err}")

# Check balance
fall_count = class_counts[1]
normal_count = class_counts[0]
if total_boxes > 0:
    imbalance_ratio = max(fall_count, normal_count) / min(fall_count, normal_count) if min(fall_count, normal_count) > 0 else float('inf')
    print(f"\n📊 Class Balance:")
    print(f"   Ratio (fall:normal) = {fall_count}:{normal_count}")
    print(f"   Imbalance ratio: {imbalance_ratio:.2f}:1")
    
    if imbalance_ratio > 3:
        print(f"   ⚠️  WARNING: Significant class imbalance detected!")
    else:
        print(f"   ✅ Classes are reasonably balanced")

# ============================================================================
# CHECK 4: Image Integrity
# ============================================================================
print("\n[4/7] Checking image integrity...")
print("-" * 70)

corrupted_images = []
image_sizes = []
image_formats = Counter()

sample_size = min(len(paired_stems), 500)  # Check first 500 images
print(f"Checking {sample_size} images for corruption...")

for i, stem in enumerate(paired_stems[:sample_size], 1):
    if i % 100 == 0:
        print(f"  Progress: {i}/{sample_size}...")
    
    img_path = None
    for ext in ['.jpg', '.png']:
        test_path = TEST_IMAGES_DIR / f"{stem}{ext}"
        if test_path.exists():
            img_path = test_path
            image_formats[ext] = image_formats.get(ext, 0) + 1
            break
    
    if not img_path:
        corrupted_images.append({'file': stem, 'error': 'File not found'})
        continue
    
    try:
        # Try opening with PIL
        with Image.open(img_path) as img:
            width, height = img.size
            image_sizes.append((width, height))
            
            # Verify it's not corrupted
            img.verify()
        
        # Also try with OpenCV
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            corrupted_images.append({'file': stem, 'error': 'OpenCV cannot read'})
        elif cv_img.size == 0:
            corrupted_images.append({'file': stem, 'error': 'Empty image'})
            
    except Exception as e:
        corrupted_images.append({'file': stem, 'error': str(e)})

if corrupted_images:
    print(f"\n⚠️  Corrupted images found: {len(corrupted_images)}")
    for err in corrupted_images[:10]:
        print(f"   {err}")
else:
    print(f"\n✅ All checked images are valid")

print(f"\n📊 Image Statistics:")
print(f"   Image formats: {dict(image_formats)}")
if image_sizes:
    widths = [s[0] for s in image_sizes]
    heights = [s[1] for s in image_sizes]
    print(f"   Width  - Mean: {np.mean(widths):.0f}, Min: {np.min(widths)}, Max: {np.max(widths)}")
    print(f"   Height - Mean: {np.mean(heights):.0f}, Min: {np.min(heights)}, Max: {np.max(heights)}")

# ============================================================================
# CHECK 5: Visual Inspection of Sample Images
# ============================================================================
print("\n[5/7] Generating visual inspection samples...")
print("-" * 70)

# Select diverse samples
fall_samples = [s for s in paired_stems if class_counts[1] > 0][:6]
normal_samples = [s for s in paired_stems if class_counts[0] > 0][:6]

# Randomly select if not enough
if len(fall_samples) < 6:
    fall_samples = random.sample(paired_stems, min(6, len(paired_stems)))
if len(normal_samples) < 6:
    normal_samples = random.sample(paired_stems, min(6, len(paired_stems)))

samples = fall_samples[:6] + normal_samples[:6]

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.flatten()

for idx, stem in enumerate(samples[:12]):
    ax = axes[idx]
    
    # Find image file
    img_path = None
    for ext in ['.jpg', '.png']:
        test_path = TEST_IMAGES_DIR / f"{stem}{ext}"
        if test_path.exists():
            img_path = test_path
            break
    
    if not img_path:
        ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        ax.set_title(stem, fontsize=8)
        ax.axis('off')
        continue
    
    # Load image
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    
    ax.imshow(img)
    
    # Draw bounding boxes
    label_path = TEST_LABELS_DIR / f"{stem}.txt"
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center, y_center, w, h = map(float, parts[1:])
                
                # Convert normalized to pixel coordinates
                x_center_px = x_center * width
                y_center_px = y_center * height
                w_px = w * width
                h_px = h * height
                
                # Calculate top-left corner
                x1 = x_center_px - w_px / 2
                y1 = y_center_px - h_px / 2
                
                # Draw rectangle
                color = 'red' if class_id == 1 else 'blue'
                rect = patches.Rectangle((x1, y1), w_px, h_px, 
                                         linewidth=2, edgecolor=color, 
                                         facecolor='none')
                ax.add_patch(rect)
                
                # Add label
                label_text = CLASS_NAMES[class_id]
                ax.text(x1, y1 - 10, label_text, 
                       color='white', fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
    
    ax.axis('off')
    ax.set_title(f"{stem[:30]}...", fontsize=8)

plt.suptitle('Sample Images with Labels (Red=Fall, Blue=Normal)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'sample_images_with_labels.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_DIR}/sample_images_with_labels.png")

# ============================================================================
# CHECK 6: Class Distribution Visualization
# ============================================================================
print("\n[6/7] Generating distribution visualizations...")
print("-" * 70)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
if total_boxes > 0:
    labels = [f"{CLASS_NAMES[i]}\n({class_counts[i]} boxes)" for i in sorted(CLASS_NAMES.keys())]
    sizes = [class_counts[i] for i in sorted(CLASS_NAMES.keys())]
    colors = ['#3498db', '#e74c3c']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=90, textprops={'fontsize': 12, 'fontweight': 'bold'})
    axes[0].set_title('Class Distribution in Test Set', fontsize=14, fontweight='bold')

# Bar chart
axes[1].bar(CLASS_NAMES.values(), 
           [class_counts[i] for i in sorted(CLASS_NAMES.keys())],
           color=colors, alpha=0.8)
axes[1].set_ylabel('Number of Bounding Boxes', fontsize=12, fontweight='bold')
axes[1].set_title('Class Counts', fontsize=14, fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

for i, (class_name, count) in enumerate(zip(CLASS_NAMES.values(), 
                                            [class_counts[j] for j in sorted(CLASS_NAMES.keys())])):
    axes[1].text(i, count + max(class_counts.values())*0.02, 
                str(count), ha='center', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'class_distribution.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_DIR}/class_distribution.png")

# ============================================================================
# CHECK 7: Generate Comprehensive Report
# ============================================================================
print("\n[7/7] Generating comprehensive validation report...")
print("-" * 70)

report = {
    'test_dataset_path': str(TEST_IMAGES_DIR),
    'summary': {
        'total_images': len(image_files),
        'total_labels': len(label_files),
        'valid_pairs': len(paired_stems),
        'images_without_labels': len(images_without_labels),
        'labels_without_images': len(labels_without_images),
        'corrupted_images': len(corrupted_images),
        'empty_labels': len(empty_labels),
        'label_format_errors': len(label_format_errors),
        'invalid_labels': len(invalid_labels)
    },
    'class_distribution': {
        CLASS_NAMES[i]: class_counts[i] for i in sorted(CLASS_NAMES.keys())
    },
    'class_balance': {
        'fall_count': fall_count,
        'normal_count': normal_count,
        'ratio': f"{fall_count}:{normal_count}",
        'imbalance_ratio': float(imbalance_ratio) if 'imbalance_ratio' in locals() else None,
        'is_balanced': imbalance_ratio <= 3 if 'imbalance_ratio' in locals() else None
    },
    'bboxes_stats': {
        'mean_per_image': float(np.mean(bboxes_per_image)),
        'median_per_image': float(np.median(bboxes_per_image)),
        'min_per_image': int(np.min(bboxes_per_image)),
        'max_per_image': int(np.max(bboxes_per_image))
    },
    'image_stats': {
        'formats': dict(image_formats),
        'width_mean': float(np.mean(widths)) if image_sizes else None,
        'height_mean': float(np.mean(heights)) if image_sizes else None,
        'width_range': [int(np.min(widths)), int(np.max(widths))] if image_sizes else None,
        'height_range': [int(np.min(heights)), int(np.max(heights))] if image_sizes else None
    },
    'issues': {
        'corrupted_images': corrupted_images[:20],
        'empty_labels': empty_labels[:20],
        'label_format_errors': label_format_errors[:20],
        'invalid_labels': invalid_labels[:20]
    }
}

with open(OUTPUT_DIR / 'validation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"✅ Saved: {OUTPUT_DIR}/validation_report.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("VALIDATION COMPLETE - SUMMARY")
print("="*70)

issues_found = []

if len(paired_stems) < len(image_files) * 0.95:
    issues_found.append(f"⚠️  {len(images_without_labels)} images missing labels")

if corrupted_images:
    issues_found.append(f"⚠️  {len(corrupted_images)} corrupted images")

if empty_labels:
    issues_found.append(f"⚠️  {len(empty_labels)} empty label files")

if label_format_errors:
    issues_found.append(f"⚠️  {len(label_format_errors)} label format errors")

if invalid_labels:
    issues_found.append(f"⚠️  {len(invalid_labels)} invalid labels")

if 'imbalance_ratio' in locals() and imbalance_ratio > 3:
    issues_found.append(f"⚠️  Significant class imbalance ({imbalance_ratio:.1f}:1)")

if issues_found:
    print("\n🔴 ISSUES FOUND:")
    for issue in issues_found:
        print(f"   {issue}")
    print(f"\n⚠️  Data quality issues detected! Review before training.")
else:
    print("\n✅ NO CRITICAL ISSUES FOUND!")
    print("   Dataset appears to be properly formatted and balanced.")

print(f"\n📊 Key Statistics:")
print(f"   Valid image-label pairs: {len(paired_stems)}")
print(f"   Total bounding boxes: {total_boxes}")
print(f"   Fall boxes:   {fall_count} ({fall_count/total_boxes*100:.1f}%)")
print(f"   Normal boxes: {normal_count} ({normal_count/total_boxes*100:.1f}%)")

print(f"\n📁 Reports saved to: {OUTPUT_DIR}/")
print(f"   - validation_report.json")
print(f"   - sample_images_with_labels.png")
print(f"   - class_distribution.png")

print("\n" + "="*70 + "\n")

