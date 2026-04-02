"""
Shared MoViNet-A2 Loader
=========================
FAST PATH (demo): loads pre-exported TF SavedModel — no 'official' module needed.
SLOW PATH (export only): falls back to tf_keras.models.load_model with BinaryMovinet.
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
from pathlib import Path


class _SavedModelWrapper:
    """Wraps a tf.saved_model so it exposes predict_on_batch() like a Keras model."""
    def __init__(self, saved_model):
        self._infer   = saved_model.signatures["serving_default"]
        self._in_key  = list(self._infer.structured_input_signature[1].keys())[0]
        self._out_key = list(self._infer.structured_outputs.keys())[0]

    def predict_on_batch(self, x):
        result = self._infer(tf.constant(x, dtype=tf.float32))
        return result[self._out_key].numpy()


def load_movinet(keras_path: str, clip_frames: int):
    """
    Load a BinaryMovinet model.

    Fast path: pre-exported SavedModel → loads in ~3 sec, no 'official' import needed.
    Slow path: tf_keras.models.load_model → takes minutes, needs 'official' installed.

    SavedModel location is derived automatically from the keras_path:
        fall_detection/fall_model_best.keras  →  fall_detection/fall_savedmodel/
        seizure_detection/seizure_model_best.keras  →  seizure_detection/seizure_savedmodel/
    """
    keras_path = Path(keras_path)
    prefix     = keras_path.stem.split("_model_")[0]   # e.g. "fall" or "seizure"
    saved_path = keras_path.parent / f"{prefix}_savedmodel"

    # ── Fast path ─────────────────────────────────────────────────────────────
    if saved_path.exists():
        print(f"  Loading SavedModel: {saved_path} ...")
        loaded  = tf.saved_model.load(str(saved_path))
        wrapper = _SavedModelWrapper(loaded)
        # One warm-up pass to trigger XLA compilation (fast after this)
        import numpy as np
        wrapper.predict_on_batch(np.zeros([1, clip_frames, 224, 224, 3], dtype='float32'))
        print(f"  ✓ {prefix.capitalize()} model ready")
        return wrapper

    # ── Slow path (needs 'official' module) ───────────────────────────────────
    print(f"  ⚠  No SavedModel at {saved_path}")
    print(f"  ⚠  Falling back to slow .keras load — run scripts/export_models.py first!")

    try:
        import tf_keras
        from official.projects.movinet.modeling import movinet as movinet_lib
        from official.projects.movinet.modeling import movinet_model

        class BinaryMovinet(tf_keras.Model):
            def __init__(self, backbone=None, model_id='a2', **kwargs):
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

        model = tf_keras.models.load_model(
            str(keras_path),
            custom_objects={'BinaryMovinet': BinaryMovinet},
            compile=False
        )
        print(f"  ✓ Model loaded from .keras (slow path)")
        return model

    except Exception as e:
        raise RuntimeError(
            f"Cannot load model. SavedModel missing and .keras fallback failed: {e}\n"
            f"Run:  python scripts/export_models.py"
        )
