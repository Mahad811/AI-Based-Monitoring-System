"""
One-Time Model Export Script
=============================
Converts .keras -> TF SavedModel for 10x faster loading.

Usage:  python scripts/export_models.py
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['PYTHONWARNINGS'] = 'ignore'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import sys
import time
import threading
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tf_keras
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from visual_guardian.movinet_loader import BinaryMovinet

ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "name":        "Fall MoViNet-A2 (16-frame)",
        "keras_path":  ROOT / "fall_detection" / "fall_model_best.keras",
        "saved_path":  ROOT / "fall_detection" / "fall_savedmodel",
        "clip_frames": 16,
    },
    {
        "name":        "Seizure MoViNet-A2 (32-frame)",
        "keras_path":  ROOT / "seizure_detection" / "seizure_model_best.keras",
        "saved_path":  ROOT / "seizure_detection" / "seizure_savedmodel",
        "clip_frames": 32,
    },
]

# ── Heartbeat ─────────────────────────────────────────────────────────────────
_stop = threading.Event()

def _heartbeat(label, interval=5):
    start = time.time()
    while not _stop.is_set():
        time.sleep(interval)
        if not _stop.is_set():
            e = int(time.time() - start)
            print(f"  ⏳  [{label}] still working... {e//60:02d}:{e%60:02d}", flush=True)

def hb_start(label):
    _stop.clear()
    t = threading.Thread(target=_heartbeat, args=(label,), daemon=True)
    t.start()
    return t

def hb_stop(t):
    _stop.set()
    t.join(timeout=2)


def export_model(cfg):
    print(f"\n{'='*60}")
    print(f"  {cfg['name']}")
    print(f"{'='*60}")

    if not cfg["keras_path"].exists():
        print(f"  ❌ Not found: {cfg['keras_path']}")
        return False

    if cfg["saved_path"].exists():
        print(f"  ✓ Already exported → {cfg['saved_path']}")
        print(f"    (Delete folder to re-export)")
        return True

    # Step 1: Load via tf_keras.models.load_model (correct deserialization)
    print(f"\n  [1/4] Loading .keras model ...")
    print(f"        {cfg['keras_path']}")
    print(f"        This builds the full MoViNet graph — may take 3-8 min on CPU.")
    t = hb_start("loading model")
    t0 = time.time()

    model = tf_keras.models.load_model(
        str(cfg["keras_path"]),
        custom_objects={'BinaryMovinet': BinaryMovinet},
        compile=False
    )

    hb_stop(t)
    print(f"  ✓ Loaded in {time.time()-t0:.0f}s", flush=True)

    # Step 2: Warm-up
    print(f"\n  [2/4] Warm-up inference ...")
    t = hb_start("warm-up")
    t0 = time.time()
    dummy = np.zeros([1, cfg["clip_frames"], 224, 224, 3], dtype="float32")
    _ = model.predict_on_batch(dummy)
    hb_stop(t)
    print(f"  ✓ Warm-up done ({time.time()-t0:.0f}s)", flush=True)

    # Step 3: Export to SavedModel
    print(f"\n  [3/4] Saving compiled graph ...")
    print(f"        → {cfg['saved_path']}")
    t = hb_start("saving")
    t0 = time.time()
    tf.saved_model.save(model, str(cfg["saved_path"]))
    hb_stop(t)
    print(f"  ✓ Saved ({time.time()-t0:.0f}s)", flush=True)

    # Step 4: Quick verify
    print(f"\n  [4/4] Verifying ...")
    t = hb_start("verifying")
    t0 = time.time()
    loaded = tf.saved_model.load(str(cfg["saved_path"]))
    infer  = loaded.signatures["serving_default"]
    result = infer(tf.constant(dummy))
    out    = list(result.values())[0].numpy().flatten()[0]
    hb_stop(t)
    print(f"  ✓ Verified — prob={out:.4f} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n  ✅ {cfg['name']} export complete!")
    return True


if __name__ == "__main__":
    t_total = time.time()
    print("=" * 60)
    print("  Vital Guardian — One-Time Model Export")
    print("=" * 60)
    print("  This runs ONCE. After this, demo loads in ~5 seconds.")
    print("  Do NOT close until you see ✅ All done.\n")

    ok = all(export_model(c) for c in MODELS)

    e = int(time.time() - t_total)
    print(f"\n{'='*60}")
    if ok:
        print(f"  ✅ All done in {e//60}m {e%60}s!")
        print(f"  Run the demo:  python scripts\\demo\\demo_server.py")
    else:
        print(f"  ❌ Some exports failed.")
    print("=" * 60)
