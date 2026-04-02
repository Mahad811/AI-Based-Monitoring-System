"""
Standalone sanity check for YOLOv11n OpenVINO GPU inference.

Run (PowerShell) from repo root:
  python scripts/diagnostics/test_yolo11n_openvino_gpu.py --model yolo11n_openvino_model --device intel:gpu

If GPU plugin is not available, you'll get a clear exception. Try --device intel:cpu to compare.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="yolo11n_openvino_model",
        help="Path to OpenVINO-exported YOLO model folder (contains *.xml and *.bin).",
    )
    p.add_argument(
        "--device",
        default="intel:gpu",
        help="Ultralytics/OpenVINO device string (e.g. 'intel:gpu', 'intel:cpu').",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (square).",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of timed inference runs.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup runs excluded from timing.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if args.device.lower() in {"gpu", "cpu"}:
        print(
            "NOTE: For OpenVINO models, use 'intel:gpu' or 'intel:cpu'. "
            "Passing 'GPU' is interpreted as CUDA by Ultralytics."
        )

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("FAIL: ultralytics not importable. Install with: pip install ultralytics")
        print(f"Error: {e}")
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"FAIL: model path not found: {model_path}")
        print("Hint: run `yolo export model=yolo11n.pt format=openvino` first.")
        return 2

    xmls = list(model_path.glob("*.xml"))
    bins = list(model_path.glob("*.bin"))
    if not xmls or not bins:
        print(f"FAIL: {model_path} does not look like an OpenVINO IR folder (missing .xml/.bin).")
        print(f"Found xml={len(xmls)} bin={len(bins)}")
        return 2

    print("== Vital Guardian | YOLO11n OpenVINO sanity check ==")
    print(f"Model folder : {model_path.resolve()}")
    print(f"Device       : {args.device}")
    print(f"imgsz        : {args.imgsz}")
    print(f"warmup/runs  : {args.warmup}/{args.runs}")
    print(f"Python       : {os.sys.version.split()[0]}")

    # Synthetic input (random image). This only validates inference path, not accuracy.
    img = (np.random.rand(args.imgsz, args.imgsz, 3) * 255).astype(np.uint8)

    try:
        yolo = YOLO(str(model_path))
    except Exception as e:
        print("FAIL: could not load YOLO model folder.")
        print(f"Error: {e}")
        return 3

    # Warmup
    try:
        for _ in range(args.warmup):
            _ = yolo.predict(img, verbose=False, classes=[0], device=args.device)
    except Exception as e:
        print("FAIL: inference failed during warmup.")
        print("This usually means the requested device plugin is unavailable (e.g. OpenVINO GPU not set up).")
        print(f"Error: {e}")
        return 4

    # Timed runs
    times_ms: list[float] = []
    last_res = None
    for _ in range(args.runs):
        t0 = time.perf_counter()
        last_res = yolo.predict(img, verbose=False, classes=[0], device=args.device)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)

    p50 = float(np.percentile(times_ms, 50))
    p90 = float(np.percentile(times_ms, 90))
    avg = float(np.mean(times_ms))

    # Ultralytics returns a list of Results; speed info (if present) is per-result.
    speed = None
    try:
        if last_res and hasattr(last_res[0], "speed"):
            speed = last_res[0].speed
    except Exception:
        speed = None

    print("\nPASS: inference completed.")
    print(f"Timing (ms)  : avg={avg:.1f} p50={p50:.1f} p90={p90:.1f} over {args.runs} runs")
    if speed:
        # Example keys: preprocess, inference, postprocess
        print(f"Ultralytics speed (ms): {speed}")
    print("\nNext check:")
    print("- Run the same script with --device CPU and compare timings.")
    print("- If GPU is slower than CPU on your machine, keep CPU for YOLO and rely on frame-skipping.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

