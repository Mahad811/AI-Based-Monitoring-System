"""
Rank unusual-movement videos for demo selection (CPU-only).

This script analyzes both Normal and Seizure clips and produces:
1) per-video scores (quality, people presence, rhythmic motion)
2) tier labels (A/B/C)
3) shortlist files for fast demo curation

Usage:
    python scripts/analysis/rank_unusual_movement_videos.py
    python scripts/analysis/rank_unusual_movement_videos.py --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def score_gaussian(x: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return clamp01(math.exp(-((x - mean) ** 2) / (2 * (std ** 2))))


def score_linear(x: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp01((x - low) / (high - low))


@dataclass
class VideoMetrics:
    path: str
    filename: str
    label: str
    duration_sec: float
    fps: float
    frame_count: int
    width: int
    height: int
    sampled_frames: int
    person_visible_ratio: float
    single_person_ratio: float
    multi_person_ratio: float
    avg_person_count: float
    blur_score: float
    exposure_score: float
    contrast_score: float
    visual_quality_score: float
    camera_stability_score: float
    motion_mean: float
    motion_std: float
    rhythmicity_score: float
    sustained_motion_score: float
    seizure_visibility_score: float
    crop_reliability_score: float
    needs_crop: bool
    crop_box_rel: str
    context_score: float
    false_alarm_value_score: float
    final_score: float
    tier: str
    normal_profile: str


class CpuVideoRanker:
    def __init__(self, sample_frames: int = 18):
        self.sample_frames = sample_frames
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def _sample_indices(self, frame_count: int) -> List[int]:
        if frame_count <= 0:
            return []
        if frame_count <= self.sample_frames:
            return list(range(frame_count))
        return np.linspace(0, frame_count - 1, self.sample_frames).astype(int).tolist()

    def _detect_people(self, frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
        # Downscale for speed then map back.
        h, w = frame_bgr.shape[:2]
        target_w = 320
        scale = safe_div(target_w, w) if w > target_w else 1.0
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale != 1.0 else frame_bgr
        rects, _ = self.hog.detectMultiScale(
            small,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        boxes = []
        for (x, y, rw, rh) in rects:
            if scale != 1.0:
                x = int(x / scale)
                y = int(y / scale)
                rw = int(rw / scale)
                rh = int(rh / scale)
            x2, y2 = x + rw, y + rh
            x, y = max(0, x), max(0, y)
            x2, y2 = min(w, x2), min(h, y2)
            if (x2 - x) > 8 and (y2 - y) > 8:
                boxes.append((x, y, x2, y2))
        return boxes

    @staticmethod
    def _largest_box(boxes: Sequence[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        if not boxes:
            return None
        return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

    @staticmethod
    def _merge_boxes_rel(
        boxes: Sequence[Tuple[int, int, int, int]], width: int, height: int
    ) -> Optional[Tuple[float, float, float, float]]:
        if not boxes:
            return None
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        x1 = np.median(xs1) / max(1, width)
        y1 = np.median(ys1) / max(1, height)
        x2 = np.median(xs2) / max(1, width)
        y2 = np.median(ys2) / max(1, height)
        return (clamp01(x1), clamp01(y1), clamp01(x2), clamp01(y2))

    def _format_crop_box(self, box_rel: Optional[Tuple[float, float, float, float]]) -> str:
        if box_rel is None:
            return ""
        return ",".join(f"{v:.4f}" for v in box_rel)

    def analyze_video(self, video_path: Path, label: str) -> Optional[VideoMetrics]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_sec = safe_div(frame_count, fps) if fps > 0 else 0.0

        sample_idx = self._sample_indices(frame_count)
        if not sample_idx:
            cap.release()
            return None

        blur_scores = []
        exposure_scores = []
        contrast_scores = []
        person_counts = []
        main_boxes = []
        frame_motion = []
        roi_motion = []
        prev_gray = None
        prev_roi_gray = None

        for idx in sample_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Quality proxies.
            lap = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_scores.append(score_linear(lap, 20.0, 220.0))

            mean_luma = float(gray.mean())
            std_luma = float(gray.std())
            exposure_scores.append(score_gaussian(mean_luma, mean=120.0, std=55.0))
            contrast_scores.append(score_linear(std_luma, 18.0, 65.0))

            # People detection.
            boxes = self._detect_people(frame)
            person_counts.append(float(len(boxes)))
            main_box = self._largest_box(boxes)
            if main_box is not None:
                main_boxes.append(main_box)

            # Global motion.
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                frame_motion.append(float(diff.mean()))
            prev_gray = gray

            # Target ROI motion (if person exists, else whole frame).
            if main_box is not None:
                x1, y1, x2, y2 = main_box
                roi = gray[y1:y2, x1:x2]
            else:
                roi = gray

            if roi.size > 0:
                roi_small = cv2.resize(roi, (128, 128))
                if prev_roi_gray is not None:
                    roi_diff = cv2.absdiff(roi_small, prev_roi_gray)
                    roi_motion.append(float(roi_diff.mean()))
                prev_roi_gray = roi_small

        cap.release()

        sampled = max(
            len(blur_scores),
            len(exposure_scores),
            len(contrast_scores),
            len(person_counts),
        )
        if sampled == 0:
            return None

        person_visible_ratio = safe_div(sum(c > 0 for c in person_counts), sampled)
        single_person_ratio = safe_div(sum(c == 1 for c in person_counts), sampled)
        multi_person_ratio = safe_div(sum(c > 1 for c in person_counts), sampled)
        avg_person_count = float(np.mean(person_counts)) if person_counts else 0.0

        blur_score = float(np.mean(blur_scores)) if blur_scores else 0.0
        exposure_score = float(np.mean(exposure_scores)) if exposure_scores else 0.0
        contrast_score = float(np.mean(contrast_scores)) if contrast_scores else 0.0
        visual_quality_score = clamp01(0.45 * blur_score + 0.30 * exposure_score + 0.25 * contrast_score)

        motion_mean = float(np.mean(frame_motion)) if frame_motion else 0.0
        motion_std = float(np.std(frame_motion)) if frame_motion else 0.0
        # Lower global motion usually means a stable camera.
        camera_stability_score = clamp01(1.0 - score_linear(motion_mean, 10.0, 55.0))

        roi_motion_arr = np.array(roi_motion, dtype=np.float32)
        if roi_motion_arr.size >= 6:
            # Rhythm score from ROI motion signal.
            signal = roi_motion_arr - np.mean(roi_motion_arr)
            spectrum = np.abs(np.fft.rfft(signal))
            freqs = np.fft.rfftfreq(signal.size, d=1.0 / max(1.0, fps))
            total_energy = float(np.sum(spectrum)) + 1e-6
            # Rhythmic movement band (approx) 1 to 6 Hz.
            band_mask = (freqs >= 1.0) & (freqs <= 6.0)
            band_energy = float(np.sum(spectrum[band_mask]))
            rhythmicity_score = clamp01(band_energy / total_energy)
            high_motion_thresh = float(np.percentile(roi_motion_arr, 65))
            sustained_motion_score = safe_div(float(np.sum(roi_motion_arr >= high_motion_thresh)), roi_motion_arr.size)
        else:
            rhythmicity_score = 0.0
            sustained_motion_score = 0.0

        seizure_visibility_score = clamp01(
            0.45 * sustained_motion_score
            + 0.35 * rhythmicity_score
            + 0.20 * person_visible_ratio
        )

        crop_reliability_score = clamp01(
            0.50 * person_visible_ratio
            + 0.35 * single_person_ratio
            + 0.15 * (1.0 - multi_person_ratio)
        )

        needs_crop = multi_person_ratio > 0.10 and person_visible_ratio > 0.50
        crop_box_rel = self._format_crop_box(self._merge_boxes_rel(main_boxes, width, height))

        # 5-second clips are ideal for this dataset; slight tolerance.
        context_score = score_gaussian(duration_sec, mean=5.0, std=1.5)

        false_alarm_value_score = clamp01(
            0.45 * score_linear(motion_mean, 6.0, 30.0)
            + 0.30 * (1.0 - rhythmicity_score)
            + 0.25 * single_person_ratio
        )

        if label.lower() == "seizure":
            final_score = clamp01(
                0.30 * seizure_visibility_score
                + 0.20 * (0.65 * single_person_ratio + 0.35 * person_visible_ratio)
                + 0.15 * crop_reliability_score
                + 0.15 * visual_quality_score
                + 0.10 * camera_stability_score
                + 0.10 * context_score
            )
            normal_profile = ""
        else:
            baseline_score = clamp01(
                0.30 * (1.0 - score_linear(motion_mean, 8.0, 28.0))
                + 0.25 * single_person_ratio
                + 0.20 * visual_quality_score
                + 0.15 * camera_stability_score
                + 0.10 * context_score
            )
            hard_negative_score = clamp01(
                0.30 * false_alarm_value_score
                + 0.25 * visual_quality_score
                + 0.20 * single_person_ratio
                + 0.15 * (1.0 - rhythmicity_score)
                + 0.10 * context_score
            )
            # Keep both useful normal styles: calm baseline and confuser negatives.
            final_score = max(baseline_score, hard_negative_score)
            normal_profile = "baseline" if baseline_score >= hard_negative_score else "hard_negative"

        return VideoMetrics(
            path=str(video_path),
            filename=video_path.name,
            label=label.lower(),
            duration_sec=round(duration_sec, 4),
            fps=round(fps, 4),
            frame_count=frame_count,
            width=width,
            height=height,
            sampled_frames=sampled,
            person_visible_ratio=round(person_visible_ratio, 6),
            single_person_ratio=round(single_person_ratio, 6),
            multi_person_ratio=round(multi_person_ratio, 6),
            avg_person_count=round(avg_person_count, 6),
            blur_score=round(blur_score, 6),
            exposure_score=round(exposure_score, 6),
            contrast_score=round(contrast_score, 6),
            visual_quality_score=round(visual_quality_score, 6),
            camera_stability_score=round(camera_stability_score, 6),
            motion_mean=round(motion_mean, 6),
            motion_std=round(motion_std, 6),
            rhythmicity_score=round(rhythmicity_score, 6),
            sustained_motion_score=round(sustained_motion_score, 6),
            seizure_visibility_score=round(seizure_visibility_score, 6),
            crop_reliability_score=round(crop_reliability_score, 6),
            needs_crop=bool(needs_crop),
            crop_box_rel=crop_box_rel,
            context_score=round(context_score, 6),
            false_alarm_value_score=round(false_alarm_value_score, 6),
            final_score=round(final_score, 6),
            tier="",
            normal_profile=normal_profile,
        )


def assign_tiers(metrics: List[VideoMetrics]) -> None:
    # Assign tiers within each class so seizure and normal both get strong picks.
    by_label = {"normal": [], "seizure": []}
    for idx, m in enumerate(metrics):
        by_label[m.label].append((idx, m))

    for label, items in by_label.items():
        if not items:
            continue
        scores = np.array([m.final_score for _, m in items], dtype=np.float32)
        q_a = float(np.quantile(scores, 0.80))
        q_b = float(np.quantile(scores, 0.50))
        for _, m in items:
            if m.final_score >= q_a:
                m.tier = "A"
            elif m.final_score >= q_b:
                m.tier = "B"
            else:
                m.tier = "C"


def write_csv(path: Path, rows: List[VideoMetrics]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def collect_videos(root: Path, label: str, limit: int = 0) -> List[Path]:
    files = sorted(root.glob("*.mp4"))
    if limit and limit > 0:
        files = files[:limit]
    return files


def progress_iter(items: Sequence[Path], desc: str):
    if tqdm is not None:
        return tqdm(items, desc=desc)
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank unusual movement videos for demo selection (CPU)")
    parser.add_argument(
        "--data-root",
        default=r"D:\project\FYP\datasets\raw_datasets\raw_datasets\unusual_movement\data",
        help="Root folder containing Normal/ and Seizure/ directories",
    )
    parser.add_argument(
        "--output-root",
        default=r"datasets\vision\processed\unusual_movement\ranking",
        help="Output folder for ranking artifacts",
    )
    parser.add_argument("--sample-frames", type=int, default=18, help="Sampled frames per video")
    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=0,
        help="Optional cap per class (0 = all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: process only first 40 videos per class",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    normal_dir = data_root / "Normal"
    seizure_dir = data_root / "Seizure"
    output_root = Path(args.output_root)

    if not normal_dir.exists() or not seizure_dir.exists():
        raise FileNotFoundError(
            f"Expected both folders under {data_root}: Normal/ and Seizure/"
        )

    limit = 40 if args.quick else args.limit_per_class
    normal_videos = collect_videos(normal_dir, "normal", limit)
    seizure_videos = collect_videos(seizure_dir, "seizure", limit)

    print("=" * 72)
    print("UNUSUAL MOVEMENT VIDEO RANKING (CPU)")
    print("=" * 72)
    print(f"Data root         : {data_root}")
    print(f"Normal videos     : {len(normal_videos)}")
    print(f"Seizure videos    : {len(seizure_videos)}")
    print(f"Sample frames     : {args.sample_frames}")
    print(f"Output root       : {output_root}")
    print("=" * 72)

    ranker = CpuVideoRanker(sample_frames=args.sample_frames)
    rows: List[VideoMetrics] = []

    for p in progress_iter(normal_videos, desc="Analyzing normal"):
        m = ranker.analyze_video(p, "normal")
        if m is not None:
            rows.append(m)

    for p in progress_iter(seizure_videos, desc="Analyzing seizure"):
        m = ranker.analyze_video(p, "seizure")
        if m is not None:
            rows.append(m)

    if not rows:
        raise RuntimeError("No videos were successfully analyzed.")

    assign_tiers(rows)
    rows_sorted = sorted(rows, key=lambda x: (x.label, -x.final_score, x.filename))

    write_csv(output_root / "video_ranking_all.csv", rows_sorted)

    normal_sorted = [r for r in rows_sorted if r.label == "normal"]
    seizure_sorted = [r for r in rows_sorted if r.label == "seizure"]

    normal_tier_a = [r for r in normal_sorted if r.tier == "A"]
    seizure_tier_a = [r for r in seizure_sorted if r.tier == "A"]

    # Two normal shortlist flavors.
    normal_baseline = sorted(
        [r for r in normal_tier_a if r.normal_profile == "baseline"],
        key=lambda x: -x.final_score,
    )
    normal_hard_neg = sorted(
        [r for r in normal_tier_a if r.normal_profile == "hard_negative"],
        key=lambda x: -x.final_score,
    )

    write_csv(output_root / "shortlist_seizure_tierA.csv", seizure_tier_a)
    write_csv(output_root / "shortlist_normal_baseline_tierA.csv", normal_baseline)
    write_csv(output_root / "shortlist_normal_hard_negative_tierA.csv", normal_hard_neg)

    summary = {
        "totals": {
            "analyzed": len(rows_sorted),
            "normal": len(normal_sorted),
            "seizure": len(seizure_sorted),
        },
        "tier_counts": {
            "normal": {
                "A": sum(r.tier == "A" for r in normal_sorted),
                "B": sum(r.tier == "B" for r in normal_sorted),
                "C": sum(r.tier == "C" for r in normal_sorted),
            },
            "seizure": {
                "A": sum(r.tier == "A" for r in seizure_sorted),
                "B": sum(r.tier == "B" for r in seizure_sorted),
                "C": sum(r.tier == "C" for r in seizure_sorted),
            },
        },
        "top_examples": {
            "seizure_top10": [asdict(r) for r in seizure_tier_a[:10]],
            "normal_baseline_top10": [asdict(r) for r in normal_baseline[:10]],
            "normal_hard_negative_top10": [asdict(r) for r in normal_hard_neg[:10]],
        },
    }
    write_json(output_root / "ranking_summary.json", summary)

    print("\nRanking complete.")
    print(f"Saved: {output_root / 'video_ranking_all.csv'}")
    print(f"Saved: {output_root / 'shortlist_seizure_tierA.csv'}")
    print(f"Saved: {output_root / 'shortlist_normal_baseline_tierA.csv'}")
    print(f"Saved: {output_root / 'shortlist_normal_hard_negative_tierA.csv'}")
    print(f"Saved: {output_root / 'ranking_summary.json'}")


if __name__ == "__main__":
    main()
