"""
Shared MoViNet-A2 Loader
=========================
Loads .keras model exactly like the Kaggle server and runs predict_on_batch.

INIT ORDER IS CRITICAL:
  TF GPU context must be initialized BEFORE PyTorch/YOLO loads.
  demo_server.py handles this by running a trivial TF GPU op at the very top,
  before VisionPipeline (YOLO) is created. This file is import-order agnostic.

BinaryMovinet is defined at module level so export_models.py can also import it.

GPU/CPU device logic:
  - Set MOVINET_FORCE_CPU=true in env to skip GPU entirely (recommended for Docker
    when TF/PyTorch share CUDA context and GPU warmup fails).
  - Otherwise attempts GPU; if warmup raises, automatically falls back to CPU
    so the pipeline always works even if GPU setup is incomplete.
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import tf_keras
from pathlib import Path


# ── BinaryMovinet — exact same class as Kaggle server ────────────────────────
def _build_binary_movinet_class():
    """Returns BinaryMovinet using official.projects.movinet (same as Kaggle)."""
    from official.projects.movinet.modeling import movinet as movinet_lib
    from official.projects.movinet.modeling import movinet_model

    class BinaryMovinet(tf_keras.Model):
        def __init__(self, backbone=None, model_id='a2', model_name=None, **kwargs):
            super().__init__(**kwargs)
            if backbone is None:
                backbone = movinet_lib.Movinet(model_id=model_id)
            self._model_id = model_id
            self.classifier = movinet_model.MovinetClassifier(
                backbone=backbone, num_classes=1, dropout_rate=0.5)

        def call(self, inputs, training=False):
            return tf.sigmoid(self.classifier(inputs, training=training))

        def get_config(self):
            cfg = super().get_config()
            cfg['model_id'] = self._model_id
            return cfg

        @classmethod
        def from_config(cls, config):
            config.pop('model_name', None)  # tolerate extra key from old saves
            return cls(**config)

    return BinaryMovinet


def load_movinet(keras_path: str, clip_frames: int):
    """
    Load a BinaryMovinet model — same approach as the Kaggle server.

    Loads the .keras file using tf_keras.models.load_model with BinaryMovinet
    as a custom object, then runs a warmup inference call to prime kernels.

    GPU/CPU selection:
      - MOVINET_FORCE_CPU=true  → always use CPU (set this in Docker)
      - Otherwise: attempt GPU; auto-fall-back to CPU if warmup fails
        (handles TF/PyTorch CUDA context conflict in Docker environments)
    """
    import numpy as np

    keras_path = Path(keras_path)
    if not keras_path.exists():
        raise FileNotFoundError(f"Model not found: {keras_path}")

    size_mb   = keras_path.stat().st_size / (1024 * 1024)
    gpus      = tf.config.list_physical_devices('GPU')
    force_cpu = os.getenv("MOVINET_FORCE_CPU", "false").lower() == "true"

    device = "CPU" if (force_cpu or not gpus) else "GPU:0"
    print(f"  Loading .keras model ({size_mb:.0f} MB) on {device}...")

    BinaryMovinet = _build_binary_movinet_class()

    def _load(dev):
        with tf.device(f"/{dev}"):
            return tf_keras.models.load_model(
                str(keras_path),
                custom_objects={'BinaryMovinet': BinaryMovinet},
                compile=False
            )

    model = _load(device)
    dummy = np.zeros([1, clip_frames, 224, 224, 3], dtype='float32')

    # Warmup: prime cuDNN/oneDNN kernels so first real inference is fast
    print(f"  Warming up ({device} kernel compilation)...")
    try:
        with tf.device(f"/{device}"):
            _ = model.predict_on_batch(dummy)
        print(f"  ✓ Model ready on {device} ({len(gpus)} GPU(s))")
    except Exception as gpu_err:
        if device == "CPU":
            raise  # CPU also failed — real problem, re-raise
        # GPU warmup failed (e.g. libdevice missing, TF/PyTorch CUDA conflict).
        # Automatically reload on CPU so the pipeline still works.
        print(f"  ⚠ GPU warmup failed ({type(gpu_err).__name__}), falling back to CPU...")
        device = "CPU"
        model  = _load(device)
        with tf.device("/CPU"):
            _ = model.predict_on_batch(dummy)
        print(f"  ✓ Model ready on CPU (GPU unavailable in this environment)")

    return model
