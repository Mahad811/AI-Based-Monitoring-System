# SLP Dataset Preprocessing Guide

## Quick Start (Complete Before Tomorrow)

### Step 1: Quick Test (5 minutes)
```bash
# Test preprocessing on 5 subjects only
python scripts/preprocess_slp.py --quick
```

**Expected output:**
- ~600-700 images processed
- `datasets/vision/processed/slp/images/` created
- `datasets/vision/processed/slp/labels/` created

---

### Step 2: Full Preprocessing (30-60 minutes)
```bash
# Process all 102 subjects
python scripts/preprocess_slp.py
```

**Expected output:**
- ~13,000+ images processed
- All subjects from danaLab
- 3 cover conditions (uncover, cover1, cover2)

---

### Step 3: Merge with Fall Detection Dataset (10 minutes)
```bash
# Merge SLP normal poses with existing fall dataset
python scripts/merge_slp_with_falls.py
```

**Expected output:**
- `datasets/vision/yolo_merged/` created
- Combined dataset with 2 classes: normal (0), fall (1)
- dataset.yaml for training

---

## What Gets Created

### Directory Structure
```
datasets/vision/
├── processed/
│   └── slp/
│       ├── images/           # RGB images from SLP
│       ├── labels/           # YOLO format annotations
│       └── metadata.json     # Processing stats
│
└── yolo_merged/             # Final training dataset
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── val/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── dataset.yaml         # YOLO config
```

---

## File Formats

### SLP Images
- **Format:** PNG
- **Source:** `SLP/danaLab/{subject}/RGB/{cover}/image_*.png`
- **Output:** `{subject}_{cover}_image_*.png`

### YOLO Labels
- **Format:** TXT (one per image)
- **Content:** `class x_center y_center width height`
- **Example:** `0 0.5 0.5 0.8 0.8`
- **Classes:** 0 = normal pose

---

## Processing Details

### What the Script Does

1. **Reads SLP structure:**
   - 102 subjects in danaLab
   - 3 cover conditions per subject
   - ~45 poses per condition
   - RGB images + pose annotations

2. **Extracts pose annotations:**
   - Loads `joints_gt_RGB.mat` (14 body joints)
   - Converts joints to bounding boxes
   - Creates YOLO format labels

3. **Copies and renames:**
   - RGB images → `processed/slp/images/`
   - YOLO labels → `processed/slp/labels/`
   - Unique naming: `{subject}_{cover}_{frame}.png`

4. **Saves metadata:**
   - Total images processed
   - Cover condition breakdown
   - Subject count

---

## Advanced Options

### Process Subset of Subjects
```bash
# Process first 20 subjects only
python scripts/preprocess_slp.py --max-subjects 20
```

### Sample Frames
```bash
# Take every 2nd frame (reduce dataset size)
python scripts/preprocess_slp.py --sample-rate 2
```

---

## Expected Statistics

### Full Dataset (All 102 Subjects)
- **Total images:** ~13,770
- **Subjects:** 102
- **Cover conditions:**
  - Uncover: ~4,590 images
  - Cover1: ~4,590 images
  - Cover2: ~4,590 images

### After Merging with Falls
- **Normal poses:** ~70,000 (existing) + ~13,770 (SLP) = ~83,770
- **Fall poses:** ~9,000 (existing)
- **Total:** ~92,770 images
- **Class balance:** ~90% normal, ~10% fall

---

## Troubleshooting

### Error: "scipy not installed"
```bash
pip install scipy
```

### Error: "SLP not found"
- Check: `datasets/vision/raw/normal/SLP/` exists
- Should contain: `danaLab/` and `simLab/` folders

### Error: "No subjects found"
- Check: `SLP/danaLab/00001/` exists
- Should contain: `RGB/`, `joints_gt_RGB.mat`

### Processing too slow?
```bash
# Use quick mode (5 subjects only)
python scripts/preprocess_slp.py --quick
```

---

## Timeline (Before Tomorrow)

### Option A: Quick (1 hour total)
1. **Test preprocessing:** 5 min
2. **Full preprocessing:** 30 min
3. **Merge datasets:** 10 min
4. **Validation:** 5 min
5. **Documentation:** 10 min

### Option B: Thorough (2 hours total)
1. **Test preprocessing:** 5 min
2. **Full preprocessing:** 60 min
3. **Merge datasets:** 10 min
4. **Create splits:** 10 min
5. **Validation:** 15 min
6. **Documentation:** 20 min

---

## Validation Checklist

After preprocessing, verify:

- [ ] `datasets/vision/processed/slp/images/` has 13,000+ images
- [ ] `datasets/vision/processed/slp/labels/` has matching .txt files
- [ ] `datasets/vision/processed/slp/metadata.json` exists
- [ ] `datasets/vision/yolo_merged/train/` has images + labels
- [ ] `datasets/vision/yolo_merged/dataset.yaml` exists
- [ ] Sample a few labels to check format

### Quick Validation Commands
```bash
# Count processed images
python -c "from pathlib import Path; print(len(list(Path('datasets/vision/processed/slp/images').glob('*.png'))))"

# Count merged dataset
python -c "from pathlib import Path; print('Train:', len(list(Path('datasets/vision/yolo_merged/train/images').glob('*'))))"

# Check label format
head datasets/vision/processed/slp/labels/00001_uncover_image_000001.txt
```

---

## Next Steps (After Tomorrow)

1. **Upload to Kaggle:**
   - Compress `yolo_merged/` folder
   - Upload as new dataset
   - Name: "fall-detection-with-slp"

2. **Train Model:**
   - Use existing Kaggle notebook
   - Update dataset path
   - Train with 2 classes (normal, fall)

3. **Update Config:**
   - Replace `fall_detection/weights/best.pt`
   - Update `config/config.yaml`

---

## Summary

**What you're doing:**
- Converting SLP's 13,770 RGB images to YOLO format
- Merging with existing 79,000 fall detection frames
- Creating unified dataset with 92,770+ images

**Why:**
- Improve normal pose classification
- Better distinguish falls from normal lying
- Reduce false positives

**Time required:**
- Quick test: 5 minutes
- Full processing: 30-60 minutes
- Total: ~1 hour

**Run these 3 commands:**
```bash
python scripts/preprocess_slp.py --quick          # Test (5 min)
python scripts/preprocess_slp.py                  # Full (30-60 min)
python scripts/merge_slp_with_falls.py            # Merge (10 min)
```

Done! ✅








