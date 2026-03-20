"""
Run fall detection model on sample videos for panel demo.
Generates annotated videos with bounding boxes and performance stats.

Usage:
    python scripts/demo_fall_videos.py
"""

from ultralytics import YOLO
from pathlib import Path
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Configuration
MODEL_PATH = Path("fall_detection/weights/best.pt")
FALL_VIDEOS_DIR = Path("datasets/vision/processed/falls")
NORMAL_VIDEOS_DIR = Path("datasets/vision/processed/normal")
OUTPUT_DIR = Path("fall_detection/demo_videos")
CONF = 0.25
NUM_VIDEOS = 5  # Take 5 fall + 5 normal videos

print("\n" + "=" * 70)
print("FALL DETECTION - VIDEO DEMO")
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

# Find videos
print("\n[2/4] Finding test videos...")
if FALL_VIDEOS_DIR.exists():
    all_fall = list(FALL_VIDEOS_DIR.glob("*.mp4"))
    fall_videos = random.sample(all_fall, min(NUM_VIDEOS, len(all_fall))) if all_fall else []
else:
    fall_videos = []

if NORMAL_VIDEOS_DIR.exists():
    all_normal = list(NORMAL_VIDEOS_DIR.glob("*.mp4"))
    normal_videos = random.sample(all_normal, min(NUM_VIDEOS, len(all_normal))) if all_normal else []
else:
    normal_videos = []

if not fall_videos and not normal_videos:
    print(f"❌ No videos found in {FALL_VIDEOS_DIR} or {NORMAL_VIDEOS_DIR}")
    exit(1)

print(f"✅ Randomly selected: {len(fall_videos)} fall videos, {len(normal_videos)} normal videos")

# Process fall videos
print("\n[3/4] Processing FALL videos...")
print("-" * 70)

fall_results = []
for i, video in enumerate(fall_videos, 1):
    name = video.name
    print(f"  [{i}/{len(fall_videos)}] {name[:60]}")
    
    results = model.predict(
        source=str(video),
        save=True,
        conf=CONF,
        project=str(OUTPUT_DIR),
        name=f"fall_{i:02d}_{video.stem}",
        exist_ok=True,
        verbose=False,
    )
    
    # Count detections
    fall_detections = sum(
        1 for r in results 
        for box in (r.boxes if r.boxes is not None else []) 
        if int(box.cls[0]) == 1
    )
    total_frames = len(results)
    accuracy = (fall_detections / total_frames * 100) if total_frames > 0 else 0
    
    status = "✅" if accuracy > 70 else "⚠️"
    print(f"      {status} {fall_detections}/{total_frames} frames detected as FALL ({accuracy:.1f}%)")
    
    fall_results.append({
        "name": name,
        "fall_frames": fall_detections,
        "total_frames": total_frames,
        "accuracy": accuracy,
    })

# Process normal videos
print("\n[4/4] Processing NORMAL videos...")
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
    
    # Count normal detections (no fall)
    normal_detections = sum(
        1 for r in results 
        for box in (r.boxes if r.boxes is not None else []) 
        if int(box.cls[0]) == 0
    )
    total_frames = len(results)
    accuracy = (normal_detections / total_frames * 100) if total_frames > 0 else 0
    
    status = "✅" if accuracy > 70 else "⚠️"
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

if fall_results:
    total_fall_correct = sum(r["fall_frames"] for r in fall_results)
    total_fall_frames = sum(r["total_frames"] for r in fall_results)
    fall_acc = (total_fall_correct / total_fall_frames * 100) if total_fall_frames > 0 else 0
    print(f"\n📊 Fall Detection:")
    print(f"   Accuracy: {fall_acc:.1f}% ({total_fall_correct}/{total_fall_frames} frames)")

if normal_results:
    total_normal_correct = sum(r["normal_frames"] for r in normal_results)
    total_normal_frames = sum(r["total_frames"] for r in normal_results)
    normal_acc = (total_normal_correct / total_normal_frames * 100) if total_normal_frames > 0 else 0
    print(f"\n📊 Normal Detection:")
    print(f"   Accuracy: {normal_acc:.1f}% ({total_normal_correct}/{total_normal_frames} frames)")

if fall_results and normal_results:
    overall_correct = total_fall_correct + total_normal_correct
    overall_frames = total_fall_frames + total_normal_frames
    overall_acc = (overall_correct / overall_frames * 100) if overall_frames > 0 else 0
    print(f"\n📊 Overall:")
    print(f"   Accuracy: {overall_acc:.1f}%")

print(f"\n📁 OUTPUTS:")
print(f"   Location: {OUTPUT_DIR}/")
print(f"   Videos: {len(fall_videos) + len(normal_videos)} annotated videos with bounding boxes")
print(f"   Confidence threshold: {CONF}")

print("\n" + "=" * 70)
print("✅ VIDEO DEMO COMPLETE - READY FOR PANEL")
print("=" * 70 + "\n")

