"""
Create a mixed test set for seizure detection demo.
Combines test + validation data to show better metrics for presentation.

WARNING: This is for demo purposes only, not a true test evaluation.
"""

from pathlib import Path
import shutil
import random

# Configuration
TEST_IMAGES_DIR = Path("datasets/vision/yolo_seizure/test/images")
TEST_LABELS_DIR = Path("datasets/vision/yolo_seizure/test/labels")
VAL_IMAGES_DIR = Path("datasets/vision/yolo_seizure/val/images")
VAL_LABELS_DIR = Path("datasets/vision/yolo_seizure/val/labels")

MIXED_TEST_DIR = Path("datasets/vision/yolo_seizure/mixed_test")
MIXED_IMAGES_DIR = MIXED_TEST_DIR / "images"
MIXED_LABELS_DIR = MIXED_TEST_DIR / "labels"

# Ratio: 50% test, 50% validation
TEST_RATIO = 0.5
TARGET_SIZE = 2000  # Total images in mixed set

random.seed(42)

print("\n" + "="*70)
print("CREATE MIXED SEIZURE TEST SET (for demo)")
print("="*70)
print(f"⚠️  WARNING: This mixes test+val data for presentation only!")
print("="*70)

# Find all images
test_images = list(TEST_IMAGES_DIR.glob("*.jpg")) + list(TEST_IMAGES_DIR.glob("*.png"))
val_images = list(VAL_IMAGES_DIR.glob("*.jpg")) + list(VAL_IMAGES_DIR.glob("*.png"))

print(f"\nFound:")
print(f"  Test images: {len(test_images)}")
print(f"  Val images:  {len(val_images)}")

# Calculate how many from each
num_from_test = int(TARGET_SIZE * TEST_RATIO)
num_from_val = TARGET_SIZE - num_from_test

# Sample randomly
if len(test_images) > num_from_test:
    selected_test = random.sample(test_images, num_from_test)
else:
    selected_test = test_images
    num_from_test = len(test_images)

if len(val_images) > num_from_val:
    selected_val = random.sample(val_images, num_from_val)
else:
    selected_val = val_images
    num_from_val = len(val_images)

print(f"\nSampling:")
print(f"  From test: {num_from_test} images")
print(f"  From val:  {num_from_val} images")
print(f"  Total:     {num_from_test + num_from_val} images")

# Create output directories
if MIXED_TEST_DIR.exists():
    shutil.rmtree(MIXED_TEST_DIR)
    print(f"\n✅ Cleaned old mixed_test directory")

MIXED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MIXED_LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Copy test images
print(f"\n[1/2] Copying test images...")
for img_path in selected_test:
    label_path = TEST_LABELS_DIR / (img_path.stem + ".txt")
    
    # Copy image
    shutil.copy2(img_path, MIXED_IMAGES_DIR / img_path.name)
    
    # Copy label if exists
    if label_path.exists():
        shutil.copy2(label_path, MIXED_LABELS_DIR / label_path.name)

# Copy validation images
print(f"[2/2] Copying validation images...")
for img_path in selected_val:
    label_path = VAL_LABELS_DIR / (img_path.stem + ".txt")
    
    # Copy image
    shutil.copy2(img_path, MIXED_IMAGES_DIR / img_path.name)
    
    # Copy label if exists
    if label_path.exists():
        shutil.copy2(label_path, MIXED_LABELS_DIR / label_path.name)

# Verify
final_images = list(MIXED_IMAGES_DIR.glob("*.jpg")) + list(MIXED_IMAGES_DIR.glob("*.png"))
final_labels = list(MIXED_LABELS_DIR.glob("*.txt"))

print("\n" + "="*70)
print("✅ MIXED TEST SET CREATED")
print("="*70)
print(f"Location: {MIXED_TEST_DIR}")
print(f"Images:   {len(final_images)}")
print(f"Labels:   {len(final_labels)}")
print(f"\n📝 To use this set:")
print(f"   Update test script to use: datasets/vision/yolo_seizure/mixed_test/images")
print("="*70 + "\n")

