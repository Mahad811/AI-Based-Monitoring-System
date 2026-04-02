"""
Smoke test: loads PersonDetector exactly as VisionPipeline does (via config.yaml),
runs a few inference frames, and prints per-frame latency + device confirmation.
"""
import sys, time, yaml
import numpy as np

sys.path.insert(0, '.')

with open('config/config.yaml') as f:
    full_cfg = yaml.safe_load(f)

cfg = full_cfg['vision']['person_detector']
print(f"Config model  : {cfg['model']}")
print(f"Config device : {cfg['device']}")

import importlib.util, pathlib
_spec = importlib.util.spec_from_file_location(
    "person_detector",
    pathlib.Path("visual_guardian/person_detector.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PersonDetector = _mod.PersonDetector

det = PersonDetector(
    model_path=cfg['model'],
    confidence=cfg['confidence'],
    device=cfg['device'],
)

img = (np.random.rand(480, 640, 3) * 255).astype('uint8')

times = []
for i in range(5):
    t0 = time.perf_counter()
    result = det.detect(img)
    dt = (time.perf_counter() - t0) * 1000
    times.append(dt)
    print(f"  Frame {i+1}: {dt:.1f}ms | detection={result}")

avg = sum(times) / len(times)
print(f"\nAvg latency : {avg:.1f}ms")
print("PASS: Pipeline PersonDetector ran successfully via config.")
