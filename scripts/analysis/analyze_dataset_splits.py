"""
Analyze Train/Val/Test Splits Balance
Checks if classes are distributed proportionally across all splits

Usage:
    python scripts/analyze_dataset_splits.py
"""

from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("\n" + "="*70)
print("DATASET SPLITS ANALYSIS - CLASS BALANCE CHECK")
print("="*70)

# Configuration
DATASET_BASE = Path('datasets/vision/yolo')
SPLITS = ['train', 'val', 'test']
CLASS_NAMES = {0: 'normal', 1: 'fall'}
OUTPUT_DIR = Path('fall_detection/splits_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analyze each split
split_stats = {}

for split in SPLITS:
    print(f"\n[{split.upper()}] Analyzing...")
    print("-" * 70)
    
    images_dir = DATASET_BASE / split / 'images'
    labels_dir = DATASET_BASE / split / 'labels'
    
    if not images_dir.exists() or not labels_dir.exists():
        print(f"⚠️  {split} directory not found, skipping...")
        continue
    
    # Count files
    image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))
    label_files = list(labels_dir.glob('*.txt'))
    
    # Count classes
    class_counts = Counter()
    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    try:
                        class_id = int(parts[0])
                        class_counts[class_id] += 1
                    except ValueError:
                        pass
    
    total_boxes = sum(class_counts.values())
    normal_count = class_counts[0]
    fall_count = class_counts[1]
    
    split_stats[split] = {
        'num_images': len(image_files),
        'num_labels': len(label_files),
        'total_boxes': total_boxes,
        'normal_count': normal_count,
        'fall_count': fall_count,
        'normal_percentage': (normal_count / total_boxes * 100) if total_boxes > 0 else 0,
        'fall_percentage': (fall_count / total_boxes * 100) if total_boxes > 0 else 0
    }
    
    print(f"  Images: {len(image_files):,}")
    print(f"  Labels: {len(label_files):,}")
    print(f"  Total boxes: {total_boxes:,}")
    print(f"  Normal: {normal_count:,} ({normal_count/total_boxes*100:.1f}%)")
    print(f"  Fall:   {fall_count:,} ({fall_count/total_boxes*100:.1f}%)")

# Calculate overall statistics
print("\n" + "="*70)
print("OVERALL ANALYSIS")
print("="*70)

total_images = sum(s['num_images'] for s in split_stats.values())
total_boxes = sum(s['total_boxes'] for s in split_stats.values())
total_normal = sum(s['normal_count'] for s in split_stats.values())
total_fall = sum(s['fall_count'] for s in split_stats.values())

print(f"\nTotal across all splits:")
print(f"  Images: {total_images:,}")
print(f"  Boxes:  {total_boxes:,}")
print(f"  Normal: {total_normal:,} ({total_normal/total_boxes*100:.1f}%)")
print(f"  Fall:   {total_fall:,} ({total_fall/total_boxes*100:.1f}%)")

# Check proportional distribution
print(f"\nSplit distribution:")
for split in SPLITS:
    if split in split_stats:
        stats = split_stats[split]
        img_pct = stats['num_images'] / total_images * 100
        box_pct = stats['total_boxes'] / total_boxes * 100
        print(f"  {split.capitalize():5s}: {stats['num_images']:6,} images ({img_pct:5.1f}%), "
              f"{stats['total_boxes']:6,} boxes ({box_pct:5.1f}%)")

# Check class balance within each split
print(f"\nClass balance check:")
print(f"{'Split':<8} {'Normal %':<12} {'Fall %':<12} {'Ratio':<15} {'Balanced?'}")
print("-" * 60)

for split in SPLITS:
    if split in split_stats:
        stats = split_stats[split]
        normal_pct = stats['normal_percentage']
        fall_pct = stats['fall_percentage']
        ratio = stats['fall_count'] / stats['normal_count'] if stats['normal_count'] > 0 else 0
        balanced = "✅" if 0.75 <= ratio <= 1.33 else "⚠️"  # Within 25% is reasonable
        print(f"{split:<8} {normal_pct:<12.1f} {fall_pct:<12.1f} {ratio:<15.2f} {balanced}")

# Check if splits maintain proportions
print(f"\nProportional distribution check:")
print(f"{'Split':<8} {'Normal %':<15} {'Fall %':<15} {'Expected %':<15}")
print("-" * 60)

overall_normal_pct = total_normal / total_boxes * 100
overall_fall_pct = total_fall / total_boxes * 100

for split in SPLITS:
    if split in split_stats:
        stats = split_stats[split]
        normal_in_split = stats['normal_count'] / total_normal * 100
        fall_in_split = stats['fall_count'] / total_fall * 100
        expected = stats['total_boxes'] / total_boxes * 100
        
        deviation = abs(normal_in_split - fall_in_split)
        status = "✅" if deviation < 5 else "⚠️"
        
        print(f"{split:<8} {normal_in_split:<15.1f} {fall_in_split:<15.1f} {expected:<15.1f} {status}")

# Visualization 1: Class distribution per split
print("\nGenerating visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Bar chart - absolute counts
ax1 = axes[0, 0]
x = range(len(SPLITS))
width = 0.35
normal_counts = [split_stats[s]['normal_count'] for s in SPLITS if s in split_stats]
fall_counts = [split_stats[s]['fall_count'] for s in SPLITS if s in split_stats]

ax1.bar([i - width/2 for i in x], normal_counts, width, label='Normal', color='#3498db', alpha=0.8)
ax1.bar([i + width/2 for i in x], fall_counts, width, label='Fall', color='#e74c3c', alpha=0.8)
ax1.set_xlabel('Split', fontweight='bold')
ax1.set_ylabel('Number of Boxes', fontweight='bold')
ax1.set_title('Absolute Class Counts per Split', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([s.capitalize() for s in SPLITS if s in split_stats])
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for i, (n, f) in enumerate(zip(normal_counts, fall_counts)):
    ax1.text(i - width/2, n + max(normal_counts)*0.02, f'{n:,}', 
            ha='center', fontsize=9, fontweight='bold')
    ax1.text(i + width/2, f + max(fall_counts)*0.02, f'{f:,}', 
            ha='center', fontsize=9, fontweight='bold')

# Stacked percentage
ax2 = axes[0, 1]
normal_pcts = [split_stats[s]['normal_percentage'] for s in SPLITS if s in split_stats]
fall_pcts = [split_stats[s]['fall_percentage'] for s in SPLITS if s in split_stats]

ax2.bar(range(len(SPLITS)), normal_pcts, label='Normal', color='#3498db', alpha=0.8)
ax2.bar(range(len(SPLITS)), fall_pcts, bottom=normal_pcts, label='Fall', color='#e74c3c', alpha=0.8)
ax2.set_xlabel('Split', fontweight='bold')
ax2.set_ylabel('Percentage (%)', fontweight='bold')
ax2.set_title('Class Percentage per Split', fontweight='bold')
ax2.set_xticks(range(len(SPLITS)))
ax2.set_xticklabels([s.capitalize() for s in SPLITS if s in split_stats])
ax2.legend()
ax2.set_ylim(0, 100)
ax2.grid(axis='y', alpha=0.3)

# Split size distribution
ax3 = axes[1, 0]
split_sizes = [split_stats[s]['num_images'] for s in SPLITS if s in split_stats]
colors = ['#3498db', '#2ecc71', '#e74c3c']
ax3.pie(split_sizes, labels=[s.capitalize() for s in SPLITS if s in split_stats], 
       autopct='%1.1f%%', colors=colors, startangle=90,
       textprops={'fontsize': 11, 'fontweight': 'bold'})
ax3.set_title('Dataset Split Distribution (by images)', fontweight='bold')

# Class balance comparison
ax4 = axes[1, 1]
splits_list = [s.capitalize() for s in SPLITS if s in split_stats]
normal_ratios = [(split_stats[s]['normal_count'] / split_stats[s]['total_boxes'] * 100) 
                 for s in SPLITS if s in split_stats]
fall_ratios = [(split_stats[s]['fall_count'] / split_stats[s]['total_boxes'] * 100) 
               for s in SPLITS if s in split_stats]

x_pos = range(len(splits_list))
ax4.plot(x_pos, normal_ratios, marker='o', linewidth=2, markersize=8, 
        label='Normal %', color='#3498db')
ax4.plot(x_pos, fall_ratios, marker='s', linewidth=2, markersize=8, 
        label='Fall %', color='#e74c3c')
ax4.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Balance')
ax4.set_xlabel('Split', fontweight='bold')
ax4.set_ylabel('Class Percentage (%)', fontweight='bold')
ax4.set_title('Class Balance Consistency Across Splits', fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(splits_list)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)
ax4.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'splits_balance_analysis.png', dpi=200, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {OUTPUT_DIR}/splits_balance_analysis.png")

# Save report
report = {
    'summary': {
        'total_images': total_images,
        'total_boxes': total_boxes,
        'total_normal': total_normal,
        'total_fall': total_fall,
        'overall_balance': {
            'normal_percentage': total_normal / total_boxes * 100,
            'fall_percentage': total_fall / total_boxes * 100
        }
    },
    'splits': split_stats,
    'balance_assessment': {}
}

# Assess balance
for split in SPLITS:
    if split in split_stats:
        stats = split_stats[split]
        ratio = stats['fall_count'] / stats['normal_count'] if stats['normal_count'] > 0 else 0
        is_balanced = 0.75 <= ratio <= 1.33
        
        report['balance_assessment'][split] = {
            'ratio': ratio,
            'is_balanced': is_balanced,
            'normal_vs_expected': (stats['normal_count'] / total_normal * 100) - (stats['total_boxes'] / total_boxes * 100),
            'fall_vs_expected': (stats['fall_count'] / total_fall * 100) - (stats['total_boxes'] / total_boxes * 100)
        }

with open(OUTPUT_DIR / 'splits_analysis.json', 'w') as f:
    json.dump(report, f, indent=2)

print(f"✅ Saved: {OUTPUT_DIR}/splits_analysis.json")

# Final verdict
print("\n" + "="*70)
print("VERDICT")
print("="*70)

all_balanced = all(0.75 <= (split_stats[s]['fall_count'] / split_stats[s]['normal_count']) <= 1.33 
                   for s in SPLITS if s in split_stats and split_stats[s]['normal_count'] > 0)

proportional_splits = all(abs((split_stats[s]['normal_count'] / total_normal * 100) - 
                              (split_stats[s]['fall_count'] / total_fall * 100)) < 5
                         for s in SPLITS if s in split_stats)

if all_balanced and proportional_splits:
    print("\n✅ EXCELLENT: Dataset splits are well-balanced!")
    print("   - Each split has balanced classes")
    print("   - Classes are distributed proportionally across splits")
    print("   - Ready for training/testing")
elif all_balanced:
    print("\n✅ GOOD: Each split is balanced internally")
    print("   ⚠️  But class proportions vary slightly across splits")
    print("   - Still acceptable for training")
elif proportional_splits:
    print("\n⚠️  Classes distributed proportionally across splits")
    print("   ⚠️  But some splits have internal class imbalance")
    print("   - May need class weights during training")
else:
    print("\n⚠️  WARNING: Imbalanced splits detected!")
    print("   - Consider re-splitting the dataset")
    print("   - Or use class weights during training")

print("\n" + "="*70 + "\n")

