"""
Build a minimal Kaggle demo weights dataset.

Copies ONLY the weights required by `config/config.yaml` for the web demo:
  - fall_detection/fall_v2_ensemble/fold0.pt..fold4.pt
  - seizure_detection/seizure_v3_ensemble/fold0.pt..fold4.pt
  - seizure_detection/seizure_temporal_ensemble/fold0.pt..fold4.pt
  - yolov8n.pt (repo root)

It preserves the same relative paths so the existing config continues to work.

Usage (PowerShell examples):
  python scripts/kaggle/build_demo_weights_dataset.py --out demo_weights --dry-run
  python scripts/kaggle/build_demo_weights_dataset.py --out demo_weights
  python scripts/kaggle/build_demo_weights_dataset.py --out demo_weights --zip
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REQUIRED_RELATIVE_FILES = [
    # YOLO
    Path("yolov8n.pt"),
    # Fall ensemble
    Path("fall_detection/fall_v2_ensemble/fold0.pt"),
    Path("fall_detection/fall_v2_ensemble/fold1.pt"),
    Path("fall_detection/fall_v2_ensemble/fold2.pt"),
    Path("fall_detection/fall_v2_ensemble/fold3.pt"),
    Path("fall_detection/fall_v2_ensemble/fold4.pt"),
    # Seizure motion ensemble
    Path("seizure_detection/seizure_v3_ensemble/fold0.pt"),
    Path("seizure_detection/seizure_v3_ensemble/fold1.pt"),
    Path("seizure_detection/seizure_v3_ensemble/fold2.pt"),
    Path("seizure_detection/seizure_v3_ensemble/fold3.pt"),
    Path("seizure_detection/seizure_v3_ensemble/fold4.pt"),
    # Seizure temporal ensemble
    Path("seizure_detection/seizure_temporal_ensemble/fold0.pt"),
    Path("seizure_detection/seizure_temporal_ensemble/fold1.pt"),
    Path("seizure_detection/seizure_temporal_ensemble/fold2.pt"),
    Path("seizure_detection/seizure_temporal_ensemble/fold3.pt"),
    Path("seizure_detection/seizure_temporal_ensemble/fold4.pt"),
]


def _copy_one(src: Path, dst: Path, *, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    shutil.copy2(src, dst)


def _zip_folder(folder: Path, zip_path: Path) -> None:
    import zipfile

    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in folder.rglob("*"):
            if f.is_file():
                zf.write(f, arcname=str(f.relative_to(folder)))


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build minimal Kaggle demo weights dataset.")
    ap.add_argument(
        "--repo-root",
        default=".",
        help="Path to repo root that contains fall_detection/, seizure_detection/, yolov8n.pt (default: .)",
    )
    ap.add_argument("--out", required=True, help="Output folder to write the minimal weights dataset to.")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be copied; do not copy.")
    ap.add_argument("--zip", action="store_true", help="Also create a zip of the output folder.")
    args = ap.parse_args(argv)

    repo_root = Path(args.repo_root).resolve()
    out_root = Path(args.out).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"repo_root   : {repo_root}")
    print(f"out_root    : {out_root}")
    print(f"files_needed: {len(REQUIRED_RELATIVE_FILES)}")
    if args.dry_run:
        print("mode        : DRY RUN (no files copied)")

    missing: list[Path] = []
    copied = 0
    total_bytes = 0

    for rel in REQUIRED_RELATIVE_FILES:
        src = repo_root / rel
        dst = out_root / rel
        if not src.exists():
            missing.append(src)
            continue
        if src.is_file():
            total_bytes += src.stat().st_size
        _copy_one(src, dst, dry_run=args.dry_run)
        copied += 1
        print(f"[ok] {src}  ->  {dst}")

    if missing:
        print("\n[warn] Missing files (not copied):")
        for p in missing:
            print(f"  - {p}")

    gb = total_bytes / (1024 ** 3)
    print(f"\nDone. Copied {copied}/{len(REQUIRED_RELATIVE_FILES)} files. Size ~= {gb:.2f} GB")

    if args.zip:
        zip_path = out_root.with_suffix(".zip")
        if args.dry_run:
            print(f"[dry-run] Would create zip: {zip_path}")
        else:
            _zip_folder(out_root, zip_path)
            print(f"[ok] Zip created: {zip_path}")

    return 0 if copied == len(REQUIRED_RELATIVE_FILES) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

