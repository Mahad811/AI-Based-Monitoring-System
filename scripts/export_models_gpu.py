"""
Export MoViNet .keras → SavedModel for GPU inference.

Approach:
  - Reconstructs BinaryMovinet WITHOUT tf-models-official
    by using tensorflow_hub MoViNet backbone (same as Kaggle training).
  - Loads trained weights from .keras file into the rebuilt graph.
  - Exports to SavedModel format.
  - After this ONE-TIME export, the server loads in ~3s with full GPU.

Usage:
    bash run_export.sh
  OR (with LD_LIBRARY_PATH already set):
    venv/bin/python scripts/export_models_gpu.py
"""
import os, sys, warnings, logging, time
os.environ['TF_CPP_MIN_LOG_LEVEL']    = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']   = '0'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
warnings.filterwarnings('ignore')
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tf_keras
import tensorflow_hub as hub
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ── GPU check ─────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
print(f"\n  GPU: {gpus[0].name if gpus else 'NOT FOUND — will export on CPU'}\n")

# ──────────────────────────────────────────────────────────────────────────────
# BinaryMovinet — pure tf_keras + TF Hub (no official module needed)
# Architecture matches what was trained on Kaggle:
#   TF Hub MoViNet-A2 backbone → Dense(1) → Sigmoid
# ──────────────────────────────────────────────────────────────────────────────
MOVINET_A2_URL = "https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3"

class BinaryMovinet(tf_keras.Model):
    """Binary classifier wrapping TF Hub MoViNet-A2."""

    HUB_URL = MOVINET_A2_URL

    def __init__(self, model_name='fall_movinet_a2', **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        # Load MoViNet backbone from TF Hub (frozen weights — we override with ours)
        self.encoder = hub.KerasLayer(
            self.HUB_URL,
            trainable=True,
            name='movinet_encoder'
        )
        self.classifier_head = tf_keras.layers.Dense(1, activation='sigmoid',
                                                      name='classifier_head')

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        return self.classifier_head(x)

    def get_config(self):
        cfg = super().get_config()
        cfg['model_name'] = self.model_name
        return cfg

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def export_model(keras_path: Path, saved_path: Path, clip_frames: int, name: str):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    if not keras_path.exists():
        print(f"  ❌ Not found: {keras_path}")
        return False

    if saved_path.exists():
        print(f"  ✓ Already exported → {saved_path}")
        print(f"    Delete folder to re-export.")
        return True

    # Step 1 — Build model graph
    print(f"\n  [1/5] Building model graph (downloads MoViNet-A2 from TF Hub)...")
    print(f"        This may take 1-3 minutes on first run.")
    t0 = time.time()
    model = BinaryMovinet(model_name=keras_path.stem.split('_model_')[0] + '_movinet_a2')
    # Build the graph by running a dummy forward pass
    dummy = np.zeros([1, clip_frames, 224, 224, 3], dtype='float32')
    _ = model(dummy, training=False)
    print(f"  ✓ Model graph built ({time.time()-t0:.0f}s)")

    # Step 2 — Load trained weights from .keras file
    print(f"\n  [2/5] Loading trained weights from {keras_path.name} ...")
    t0 = time.time()
    try:
        model.load_weights(str(keras_path))
        print(f"  ✓ Weights loaded ({time.time()-t0:.0f}s)")
    except Exception as e:
        # The .keras file may use a different layer naming scheme
        # Try loading via tf_keras with custom_objects
        print(f"  ⚠  Direct load failed ({e}), trying tf_keras.models.load_model...")
        t0 = time.time()
        try:
            loaded = tf_keras.models.load_model(
                str(keras_path),
                custom_objects={'BinaryMovinet': BinaryMovinet},
                compile=False
            )
            model = loaded
            print(f"  ✓ Loaded via tf_keras ({time.time()-t0:.0f}s)")
        except Exception as e2:
            print(f"  ❌ Failed: {e2}")
            return False

    # Step 3 — Warm-up on GPU
    print(f"\n  [3/5] GPU warm-up inference ...")
    t0 = time.time()
    result = model(dummy, training=False)
    print(f"  ✓ Warm-up done — prob={float(result.numpy().flatten()[0]):.4f} ({time.time()-t0:.2f}s)")

    # Step 4 — Export to SavedModel
    print(f"\n  [4/5] Exporting to SavedModel → {saved_path} ...")
    t0 = time.time()
    tf.saved_model.save(model, str(saved_path))
    print(f"  ✓ Saved ({time.time()-t0:.0f}s)")

    # Step 5 — Verify reload
    print(f"\n  [5/5] Verifying reload ...")
    t0 = time.time()
    loaded = tf.saved_model.load(str(saved_path))
    fn = loaded.signatures.get('serving_default')
    if fn:
        out_key = list(fn.structured_outputs.keys())[0]
        result = fn(tf.constant(dummy))[out_key].numpy().flatten()[0]
        print(f"  ✓ Verified — prob={result:.4f} ({time.time()-t0:.1f}s)")
    else:
        # Try calling directly
        result = loaded(tf.constant(dummy)).numpy().flatten()[0]
        print(f"  ✓ Verified (direct call) — prob={result:.4f} ({time.time()-t0:.1f}s)")

    print(f"\n  ✅ {name} export complete!")
    return True


if __name__ == '__main__':
    t_total = time.time()
    print("=" * 60)
    print("  Vital Guardian — GPU Model Export (Ubuntu)")
    print("  MoViNet-A2 .keras → TF SavedModel")
    print("=" * 60)
    if not gpus:
        print("  ⚠  No GPU detected! Set LD_LIBRARY_PATH correctly.")
        print("     Use:  bash run_export.sh")

    models = [
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

    ok = all(export_model(**{k: v for k, v in m.items()}) for m in models)

    elapsed = int(time.time() - t_total)
    print(f"\n{'='*60}")
    if ok:
        print(f"  ✅ All done in {elapsed//60}m {elapsed%60}s!")
        print(f"  Run the server:  bash run_server.sh")
    else:
        print(f"  ❌ Some exports failed. Check output above.")
    print("=" * 60)
