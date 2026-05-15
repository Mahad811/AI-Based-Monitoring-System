"""
One-Time Model Export Script
=============================
Converts .keras -> TF SavedModel for 10x faster loading.

Usage:  python scripts/export_models.py
"""
import os
# ── Must set LD_LIBRARY_PATH BEFORE tensorflow is imported ────────────────────
_venv = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_nvidia = os.path.join(_venv, 'venv', 'lib', 'python3.10', 'site-packages', 'nvidia')
_gpu_libs = ':'.join([
    f'{_nvidia}/cudnn/lib', f'{_nvidia}/cublas/lib',
    f'{_nvidia}/cuda_runtime/lib', f'{_nvidia}/cufft/lib',
    f'{_nvidia}/cusolver/lib', f'{_nvidia}/cusparse/lib',
    f'{_nvidia}/curand/lib', f'{_nvidia}/cuda_cupti/lib',
    f'{_nvidia}/nvjitlink/lib', f'{_nvidia}/nccl/lib',
])
os.environ['LD_LIBRARY_PATH'] = _gpu_libs + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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
from visual_guardian.movinet_loader import _build_binary_movinet_class
BinaryMovinet = _build_binary_movinet_class()

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

    # Steps 2-4: Export via tf.Module wrapper (bypasses tf_keras cuDNN init issues)
    # We wrap the model in a plain tf.Module with an explicit @tf.function signature.
    # This traces the graph without running a forward pass through tf_keras's executor.
    print(f"\n  [2/4] Preparing export wrapper ...")
    dummy = np.zeros([1, cfg["clip_frames"], 224, 224, 3], dtype="float32")
    input_spec = tf.TensorSpec(shape=[None, cfg["clip_frames"], 224, 224, 3],
                               dtype=tf.float32, name="inputs")

    class _ExportModule(tf.Module):
        def __init__(self, keras_model):
            super().__init__()
            self.model = keras_model

        @tf.function(input_signature=[input_spec])
        def serving_default(self, inputs):
            return {"output": self.model(inputs, training=False)}

    module = _ExportModule(model)
    print(f"  ✓ Export wrapper ready")

    # Step 3: Get concrete function (traces graph, no actual inference needed)
    print(f"\n  [3/4] Saving SavedModel → {cfg['saved_path']} ...")
    t = hb_start("saving")
    t0 = time.time()

    signatures = {
        "serving_default": module.serving_default.get_concrete_function(
            tf.TensorSpec(shape=[None, cfg["clip_frames"], 224, 224, 3],
                          dtype=tf.float32)
        )
    }
    tf.saved_model.save(module, str(cfg["saved_path"]), signatures=signatures)

    hb_stop(t)
    print(f"  ✓ Saved ({time.time()-t0:.0f}s)", flush=True)
    print(f"\n  ✅ {cfg['name']} export complete!")
    print(f"     SavedModel will run on RTX 4050 GPU at inference time.")
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
