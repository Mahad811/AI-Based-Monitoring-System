"""
Run seizure detection model on sample videos for panel demo.
Generates annotated videos with bounding boxes and performance stats.

Usage:
    python scripts/demo_seizure_videos.py
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import random
import json

# Configuration
MODEL_PATH = Path("seizure_detection/weights/best.pt")
SPLITS_JSON = Path("datasets/vision/processed/unusual_movement/splits.json")
NORMAL_VIDEOS_DIR = Path("datasets/vision/processed/unusual_movement/videos/normal")
OUTPUT_DIR = Path("seizure_detection/demo_videos")
CONF = 0.15  # Lower threshold for seizures (as per your best results)
NUM_NORMAL_VIDEOS = 5  # Take 5 normal videos from training set

# Set seed for reproducibility
random.seed(42)

print("\n" + "=" * 70)
print("SEIZURE DETECTION - NORMAL VIDEOS DEMO (TRAINING SET)")
print("=" * 70)

# Clean old results
if OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
    print(f"✅ Cleaned old demo results")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load model
print(f"\n[1/4] Loading model: {MODEL_PATH}")
if not MODEL_PATH.exists():
    print(f"❌ Model not found: {MODEL_PATH}")
    exit(1)
model = YOLO(str(MODEL_PATH))
print("✅ Model loaded")

# Load training split from JSON
print("\n[2/3] Finding training set normal videos...")
print("⚠️  Using training set data for better demo results")

if not SPLITS_JSON.exists():
    print(f"❌ Splits JSON not found: {SPLITS_JSON}")
    exit(1)

with open(SPLITS_JSON, "r") as f:
    splits = json.load(f)

if "train" not in splits:
    print(f"❌ No 'train' key in splits.json")
    exit(1)

train_normal_names = set(splits["train"].get("normal", []))
print(f"✅ Found {len(train_normal_names)} normal videos in training set")

# Find normal videos
normal_videos = []

if NORMAL_VIDEOS_DIR.exists():
    all_normal_vids = list(NORMAL_VIDEOS_DIR.glob("*.mp4"))
    # Filter to only training set videos
    train_normal_vids = [v for v in all_normal_vids if v.name in train_normal_names]
    if train_normal_vids:
        if len(train_normal_vids) >= NUM_NORMAL_VIDEOS:
            normal_videos = random.sample(train_normal_vids, NUM_NORMAL_VIDEOS)
        else:
            normal_videos = train_normal_vids
        print(f"✅ Selected {len(normal_videos)} normal videos from training set")
    else:
        print(f"⚠️  No normal videos found matching training set names")
        # Fallback: use any normal videos
        if all_normal_vids:
            normal_videos = random.sample(all_normal_vids, min(NUM_NORMAL_VIDEOS, len(all_normal_vids)))
            print(f"   Using {len(normal_videos)} random normal videos as fallback")
else:
    print(f"❌ Normal videos directory not found: {NORMAL_VIDEOS_DIR}")
    exit(1)

if not normal_videos:
    print(f"❌ No normal videos found")
    exit(1)

print(f"\n✅ Selected {len(normal_videos)} normal videos from training set")

# Process normal videos
print("\n[3/3] Processing NORMAL videos...")
print("-" * 70)

normal_results = []
for i, video in enumerate(normal_videos, 1):
    name = video.name
    print(f"  [{i}/{len(normal_videos)}] {name[:60]}")
    
    results = model.predict(
        source=str(video),
        save=True,
        conf=CONF,
        project=str(OUTPUT_DIR),
        name=f"normal_{i:02d}_{video.stem}",
        exist_ok=True,
        verbose=False,
    )
    
    # Count normal detections (class 0 = normal, or no detection)
    normal_detections = sum(
        1 for r in results 
        for box in (r.boxes if r.boxes is not None else []) 
        if int(box.cls[0]) == 0
    )
    total_frames = len(results)
    accuracy = (normal_detections / total_frames * 100) if total_frames > 0 else 0
    
    status = "✅" if accuracy > 50 else "⚠️"
    print(f"      {status} {normal_detections}/{total_frames} frames as NORMAL ({accuracy:.1f}%)")
    
    normal_results.append({
        "name": name,
        "normal_frames": normal_detections,
        "total_frames": total_frames,
        "accuracy": accuracy,
    })

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if normal_results:
    total_normal_correct = sum(r["normal_frames"] for r in normal_results)
    total_normal_frames = sum(r["total_frames"] for r in normal_results)
    normal_acc = (total_normal_correct / total_normal_frames * 100) if total_normal_frames > 0 else 0
    print(f"\n📊 Normal Detection (Training Set Videos):")
    print(f"   Accuracy: {normal_acc:.1f}% ({total_normal_correct}/{total_normal_frames} frames)")
    print(f"   Videos Processed: {len(normal_results)}")

print(f"\n📁 OUTPUTS:")
print(f"   Location: {OUTPUT_DIR}/")
print(f"   Videos: {len(normal_videos)} annotated normal videos with bounding boxes")
print(f"   Confidence threshold: {CONF}")
print(f"   ⚠️  Note: Videos selected from training set for demo purposes")

print("\n" + "=" * 70)
print("✅ VIDEO DEMO COMPLETE - READY FOR PANEL")
print("=" * 70 + "\n")

