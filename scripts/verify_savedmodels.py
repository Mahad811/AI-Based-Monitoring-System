"""
Verify both SavedModels are valid and ready for the demo.
Usage:  python scripts/verify_savedmodels.py
"""
import os, warnings, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)

import tensorflow as tf
tf.get_logger().setLevel('ERROR')
import numpy as np

MODELS = [
    {
        "name":       "Seizure MoViNet-A2",
        "path":       "seizure_detection/seizure_savedmodel",
        "clip_frames": 32,
    },
    {
        "name":       "Fall MoViNet-A2",
        "path":       "fall_detection/fall_savedmodel",
        "clip_frames": 16,
    },
]

all_ok = True

for cfg in MODELS:
    print(f"\n{'='*55}")
    print(f"  Checking: {cfg['name']}")
    print(f"  Path    : {cfg['path']}")
    print(f"{'='*55}")

    if not os.path.exists(cfg["path"]):
        print(f"  ❌ MISSING — folder not found!")
        all_ok = False
        continue

    try:
        print("  [1/4] Loading SavedModel ...", flush=True)
        loaded = tf.saved_model.load(cfg["path"])
        print("  ✅ Loaded")

        print("  [2/4] Checking signatures ...", flush=True)
        sigs       = list(loaded.signatures.keys())
        infer      = loaded.signatures["serving_default"]
        in_key     = list(infer.structured_input_signature[1].keys())[0]
        in_shape   = infer.structured_input_signature[1][in_key].shape
        out_key    = list(infer.structured_outputs.keys())[0]
        print(f"  ✅ Signature : serving_default")
        print(f"     Input key : {in_key}")
        print(f"     Input shape: {in_shape}")
        print(f"     Output key : {out_key}")

        print("  [3/4] Inference on zeros ...", flush=True)
        dummy  = tf.zeros([1, cfg["clip_frames"], 224, 224, 3], dtype=tf.float32)
        result = infer(dummy)
        val    = float(result[out_key].numpy().flatten()[0])
        assert 0.0 <= val <= 1.0, f"Output {val} is outside [0,1]!"
        print(f"  ✅ Output = {val:.6f}  (valid sigmoid probability)")

        print("  [4/4] Inference on random noise ...", flush=True)
        noise  = tf.random.uniform([1, cfg["clip_frames"], 224, 224, 3])
        result = infer(noise)
        val2   = float(result[out_key].numpy().flatten()[0])
        assert 0.0 <= val2 <= 1.0
        print(f"  ✅ Output = {val2:.6f}  (valid)")

        print(f"\n  ✅ {cfg['name']} — READY FOR DEMO")

    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        all_ok = False

print(f"\n{'='*55}")
if all_ok:
    print("  ✅ ALL MODELS VERIFIED — run the demo:")
    print("     python scripts\\demo\\demo_server.py")
else:
    print("  ❌ Some models failed — check above.")
print(f"{'='*55}\n")
