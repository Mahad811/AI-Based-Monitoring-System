"""Quick script to get video metadata for all demo clips."""
import cv2
import numpy as np

clips = [
    ("S37_0_39",  "Normal",  r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S37_0_39.mp4"),
    ("S37_0_170", "Normal",  r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S37_0_170.mp4"),
    ("S15_1_140", "Normal",  r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S15_1_140.mp4"),
    ("S15_2_2",   "Normal",  r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Normal\S15_2_2.mp4"),
    ("S37_0_80",  "Seizure", r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S37_0_80.mp4"),
    ("S37_0_75",  "Seizure", r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S37_0_75.mp4"),
    ("S15_3_88",  "Seizure", r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S15_3_88.mp4"),
    ("S15_3_89",  "Seizure", r"d:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data\Seizure\S15_3_89.mp4"),
    ("B_D_0231",  "Fall",    r"d:\project\FYP\datasets\raw_datasets\raw_datasets\falls\harvard_fall\f_raw_b_1\f_raw_b_1\B_D_0231.mp4"),
    ("B_N_458",   "Fall",    r"d:\project\FYP\datasets\raw_datasets\raw_datasets\falls\harvard_fall\f_raw_b_2\f_raw_b_2\B_N_458.mp4"),
    ("B_M_48",    "NoFall",  r"d:\project\FYP\datasets\raw_datasets\raw_datasets\normal\Harvard_no-fall\nf_raw_b_3\nf_raw_b_3\B_M_48.mp4"),
]

print(f"{'Name':15s} | {'Resolution':10s} | {'FPS':>5s} | {'Frames':>6s} | {'Dur(s)':>6s} | {'Bright':>6s} | Label")
print("-" * 80)
for name, label, path in clips:
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur = fc / fps if fps > 0 else 0
    ret, frame = cap.read()
    brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) if ret else -1
    cap.release()
    print(f"{name:15s} | {w:4d}x{h:<4d} | {fps:5.1f} | {fc:6d} | {dur:6.1f} | {brightness:6.0f} | {label}")
