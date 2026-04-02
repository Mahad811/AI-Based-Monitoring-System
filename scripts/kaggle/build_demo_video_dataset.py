"""
Build a minimal Kaggle demo video dataset.

Copies ONLY the mp4 files referenced in `scripts/demo/demo_server.py` (SEGMENTS)
into an output directory, preserving the relative folder structure under the
raw dataset root.

Usage (PowerShell examples):
  python scripts/kaggle/build_demo_video_dataset.py --out demo_dataset
  python scripts/kaggle/build_demo_video_dataset.py --out demo_dataset --zip
  python scripts/kaggle/build_demo_video_dataset.py --raw-root "D:/path/to/raw_datasets" --out demo_dataset
  python scripts/kaggle/build_demo_video_dataset.py --out demo_dataset --dry-run

Then upload `demo_dataset/` (or the generated zip) as a Kaggle Dataset and set:
  _R = Path("/kaggle/input/<your-dataset-name>")
in `scripts/demo/demo_server.py`.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


def _extract_raw_root_from_demo_server(demo_server_path: Path) -> Path | None:
    """
    Extract the `_R = Path(r"...")` from demo_server.py if present.
    Returns a Path (may not exist).
    """
    txt = demo_server_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"_R\s*=\s*Path\(\s*r([\"'])(.+?)\1\s*\)", txt)
    if not m:
        return None
    return Path(m.group(2))


def _extract_relative_clip_paths(demo_server_path: Path) -> list[Path]:
    """
    Extract r\"...mp4\" parts from SEGMENTS.

    In demo_server.py clips are expressed like:
      _R / r\"normal\\...\\B_M_48.mp4\"
    We extract the raw string literal and treat it as a relative path.
    """
    txt = demo_server_path.read_text(encoding="utf-8", errors="ignore")

    # Narrow search to SEGMENTS block to avoid picking up unrelated mp4 mentions.
    seg_start = txt.find("SEGMENTS")
    if seg_start == -1:
        raise RuntimeError("Could not find SEGMENTS in demo_server.py")
    seg_txt = txt[seg_start:]

    rels: list[Path] = []
    for m in re.finditer(r"_R\s*/\s*r([\"'])(.+?\.mp4)\1", seg_txt, flags=re.IGNORECASE):
        rels.append(Path(m.group(2)))

    # De-duplicate while preserving order
    seen = set()
    uniq: list[Path] = []
    for p in rels:
        key = str(p).lower()
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    return uniq


def _copy_one(src: Path, dst: Path, *, dry_run: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    shutil.copy2(src, dst)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build minimal Kaggle demo videos dataset.")
    ap.add_argument(
        "--demo-server",
        default=str(Path("scripts/demo/demo_server.py")),
        help="Path to demo_server.py (default: scripts/demo/demo_server.py)",
    )
    ap.add_argument(
        "--raw-root",
        default=None,
        help="Raw dataset root path (folder that contains normal/, falls/, unusual_movement/, etc.). "
             "If omitted, attempts to read `_R = Path(...)` from demo_server.py.",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="Output folder to write the minimal dataset to (will be created).",
    )
    ap.add_argument("--dry-run", action="store_true", help="Print what would be copied; do not copy.")
    ap.add_argument("--zip", action="store_true", help="Also create a zip of the output folder.")
    args = ap.parse_args(argv)

    demo_server_path = Path(args.demo_server)
    if not demo_server_path.exists():
        print(f"[error] demo_server.py not found: {demo_server_path}", file=sys.stderr)
        return 2

    rel_clips = _extract_relative_clip_paths(demo_server_path)
    if not rel_clips:
        print("[error] No mp4 clip paths found in SEGMENTS.", file=sys.stderr)
        return 2

    raw_root = Path(args.raw_root) if args.raw_root else _extract_raw_root_from_demo_server(demo_server_path)
    if raw_root is None:
        print("[error] --raw-root not provided and `_R = Path(...)` not found in demo_server.py", file=sys.stderr)
        return 2

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"demo_server : {demo_server_path}")
    print(f"raw_root    : {raw_root}")
    print(f"out_root    : {out_root}")
    print(f"clips_found : {len(rel_clips)}")
    if args.dry_run:
        print("mode        : DRY RUN (no files copied)")

    missing: list[Path] = []
    copied = 0

    for rel in rel_clips:
        src = raw_root / rel
        dst = out_root / rel
        if not src.exists():
            missing.append(src)
            continue
        _copy_one(src, dst, dry_run=args.dry_run)
        copied += 1
        print(f"[ok] {src}  ->  {dst}")

    if missing:
        print("\n[warn] Missing files (not copied):")
        for p in missing:
            print(f"  - {p}")

    print(f"\nDone. Copied {copied}/{len(rel_clips)} files.")

    if args.zip:
        zip_path = out_root.with_suffix(".zip")
        if args.dry_run:
            print(f"[dry-run] Would create zip: {zip_path}")
        else:
            # make_archive wants base_name without extension
            import zipfile

            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for f in out_root.rglob("*"):
                    if f.is_file():
                        zf.write(f, arcname=str(f.relative_to(out_root)))
            print(f"[ok] Zip created: {zip_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

