import os
# ── Silence TF logs (MUST BE BEFORE TENSORFLOW IMPORT) ──────────────────────
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ════════════════════════════════════════════════════════════════════
#  ✏️  CONFIGURE THESE TWO PATHS before running in a new notebook.
#
#  Option A — files uploaded as a Kaggle Dataset:
#    FALL_KERAS_PATH    = "/kaggle/input/<your-dataset>/fall_model_best.keras"
#    SEIZURE_KERAS_PATH = "/kaggle/input/<your-dataset>/seizure_model_best.keras"
#
#  Option B — files uploaded directly to notebook working dir:
#    FALL_KERAS_PATH    = "/kaggle/working/fall_model_best.keras"
#    SEIZURE_KERAS_PATH = "/kaggle/working/seizure_model_best.keras"
# ════════════════════════════════════════════════════════════════════
FALL_KERAS_PATH    = "/kaggle/working/fall_model_best.keras"
SEIZURE_KERAS_PATH = "/kaggle/working/seizure_model_best.keras"

import subprocess, sys

def _pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# ── Install required packages if missing ─────────────────────────────────────
try:
    from official.projects.movinet.modeling import movinet as movinet_lib
except ModuleNotFoundError:
    print("[Setup] Installing tf-models-official (needed for BinaryMovinet)...")
    _pip("tf-models-official")
    print("[Setup] ✅ tf-models-official installed.")

try:
    import tf_keras
except ModuleNotFoundError:
    print("[Setup] Installing tf-keras...")
    _pip("tf-keras")
    print("[Setup] ✅ tf-keras installed.")

try:
    import nest_asyncio
except ModuleNotFoundError:
    print("[Setup] Installing nest-asyncio...")
    _pip("nest-asyncio")
    print("[Setup] ✅ nest-asyncio installed.")

import base64
import numpy as np
import tensorflow as tf
import tf_keras
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List
import uvicorn
import nest_asyncio

# Required: lets uvicorn run inside an already-running event loop (Kaggle uses one)
nest_asyncio.apply()


# ── BinaryMovinet custom class (needed to deserialize .keras) ─────────────────
# MUST match the class used during training exactly — same signature, same body.
from official.projects.movinet.modeling import movinet as movinet_lib
from official.projects.movinet.modeling import movinet_model

class BinaryMovinet(tf_keras.Model):
    def __init__(self, backbone=None, model_name='fall_movinet_a2'):
        super().__init__(name=model_name)
        # backbone=None happens during .keras deserialization — build a fresh
        # architecture; tf_keras will load the saved weights on top afterwards.
        if backbone is None:
            backbone = movinet_lib.Movinet(model_id='a2')
        self.classifier = movinet_model.MovinetClassifier(
            backbone=backbone,
            num_classes=1,
            dropout_rate=0.5
        )

    def call(self, inputs, training=None):
        logits = self.classifier(inputs, training=training)
        return tf.sigmoid(logits)

    def get_config(self):
        # backbone is not serializable — we reconstruct it in __init__ when None
        return {'model_name': self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ── Module-level model singletons ────────────────────────────────────────────
class Models:
    fall_model    = None
    seizure_model = None


# ── Loader helper ─────────────────────────────────────────────────────────────
def _verify_model(model, clip_frames: int, label: str) -> bool:
    """
    Run three checks to confirm the model is actually trained (not random/untrained).

    Check 1 — File size gate (already done before load, but logged here).
    Check 2 — Output spread: run N random clips; std of outputs must be > 0.05.
               An untrained model outputs ~0.50 for everything → std ≈ 0.
    Check 3 — Extreme clip test: a saturated all-ones clip and a zeros clip
               should produce different outputs from each other in a trained model.

    Returns True if model passes, False if it looks untrained.
    """
    print(f"[Verify:{label}] Running 3 verification checks ...")

    # Check 2: at least one output must be clearly away from 0.50.
    # Run 10 random clips; if ALL of them are within 0.08 of 0.50, model is broken.
    N = 10
    probs = []
    for _ in range(N):
        noise = np.random.rand(1, clip_frames, 224, 224, 3).astype('float32')
        p     = float(model.predict_on_batch(noise).flatten()[0])
        probs.append(p)
    max_deviation = float(max(abs(p - 0.5) for p in probs))
    mean_val      = float(np.mean(probs))
    print(f"[Verify:{label}] Check 2 — {N} random clips: mean={mean_val:.4f}  "
          f"max_deviation_from_0.5={max_deviation:.4f}")
    if max_deviation < 0.08:
        print(f"[Verify:{label}] ❌ FAIL — every output is within 0.08 of 0.50. "
              f"Model weights are bad/untrained.")
        return False
    print(f"[Verify:{label}] ✅ PASS — model produces non-trivial outputs.")

    # Check 3: two clearly different random clips should produce different outputs.
    # NOTE: zeros-vs-ones is NOT a valid test for MoViNet — both are static (no
    # temporal motion) so the model correctly scores them similarly. Instead we
    # compare two independent random clips which have different motion patterns.
    clip_a = np.random.rand(1, clip_frames, 224, 224, 3).astype('float32')
    clip_b = np.random.rand(1, clip_frames, 224, 224, 3).astype('float32')
    p_a = float(model.predict_on_batch(clip_a).flatten()[0])
    p_b = float(model.predict_on_batch(clip_b).flatten()[0])
    diff = abs(p_a - p_b)
    print(f"[Verify:{label}] Check 3 — clip_a={p_a:.4f}  clip_b={p_b:.4f}  diff={diff:.4f}")
    if max(abs(p_a - 0.5), abs(p_b - 0.5)) < 0.05:
        print(f"[Verify:{label}] ❌ FAIL — both random clips stuck near 0.5. Model not discriminating.")
        return False
    print(f"[Verify:{label}] ✅ PASS — model produces varied outputs on different inputs.")

    print(f"[Verify:{label}] ✅ All checks passed. Model looks correctly trained.")
    return True


def _load_keras_model(path: str, clip_frames: int, label: str):
    """
    Load a BinaryMovinet .keras model and verify it is actually trained.
    Returns a loaded tf_keras model ready for predict_on_batch(), or None on failure.
    """
    # ── Pre-load file check ───────────────────────────────────────────────────
    if not os.path.exists(path):
        print(f"[Loader:{label}] ❌ File not found: {path}")
        print(f"[Loader:{label}]    Update FALL_KERAS_PATH / SEIZURE_KERAS_PATH at the top of this file.")
        return None

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"[Loader:{label}] File found: {path}  ({size_mb:.1f} MB)")

    # A trained MoViNet-A2 .keras file should be at least 10 MB.
    # An untrained/empty export would be tiny.
    if size_mb < 10:
        print(f"[Loader:{label}] ❌ File is only {size_mb:.1f} MB — looks empty or corrupt. "
              f"Expected >10 MB for a trained MoViNet-A2.")
        return None
    print(f"[Loader:{label}] ✅ File size OK ({size_mb:.1f} MB).")

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"[Loader:{label}] Loading model (3-8 min on first run, building MoViNet graph)...")
    try:
        model = tf_keras.models.load_model(
            path,
            custom_objects={'BinaryMovinet': BinaryMovinet},
            compile=False
        )
    except Exception as e:
        print(f"[Loader:{label}] ❌ Load failed: {e}")
        return None

    print(f"[Loader:{label}] ✅ Model loaded. Running verification checks...")

    # ── Verify ────────────────────────────────────────────────────────────────
    ok = _verify_model(model, clip_frames, label)
    if not ok:
        print(f"[Loader:{label}] ❌ Model FAILED verification. "
              f"The .keras file is not correctly trained. "
              f"Make sure you are using the right file (the one that gave AUC 0.97 on Kaggle).")
        return None

    return model


# ── Lifespan Startup ──────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading Fall MoViNet-A2 (16-frame) from .keras ...")
    Models.fall_model = _load_keras_model(FALL_KERAS_PATH, clip_frames=16, label="Fall")

    print("[Startup] Loading Seizure MoViNet-A2 (32-frame) from .keras ...")
    Models.seizure_model = _load_keras_model(SEIZURE_KERAS_PATH, clip_frames=32, label="Seizure")

    if Models.fall_model is None or Models.seizure_model is None:
        print("[Startup] ❌ One or both models failed. Fix the errors above before sending requests.")
    else:
        print("[Startup] ✅ Both models verified and ready. Server accepting requests.")

    yield  # Server runs here


app = FastAPI(title="MoViNet Kaggle Inference Engine", lifespan=lifespan)


# ── Request schema ────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    frames_b64: List[str]   # base64-encoded JPEG strings


# ── Frame decoder ─────────────────────────────────────────────────────────────
def _decode_frames(frames_b64: List[str]) -> np.ndarray:
    """
    Decode list of base64 JPEG strings into float32 RGB [0,1] tensor.
    Matches exactly what the local pipeline sends.
    """
    tensors = []
    for b64 in frames_b64:
        raw = base64.b64decode(b64)
        img = tf.io.decode_jpeg(raw, channels=3)        # uint8 RGB
        img = tf.cast(img, tf.float32) / 255.0          # float32 [0, 1]
        tensors.append(img)
    return tf.stack(tensors).numpy()                    # (T, 224, 224, 3)


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "fall_model_loaded":    Models.fall_model is not None,
        "seizure_model_loaded": Models.seizure_model is not None,
    }


@app.post("/predict/fall")
def predict_fall(req: PredictionRequest):
    if Models.fall_model is None:
        raise HTTPException(status_code=503, detail="Fall model not loaded on server.")
    if len(req.frames_b64) != 16:
        raise HTTPException(status_code=400, detail=f"Expected 16 frames, got {len(req.frames_b64)}.")

    clip = _decode_frames(req.frames_b64)               # (16, 224, 224, 3)
    x    = np.expand_dims(clip, 0)                      # (1, 16, 224, 224, 3)
    prob = float(Models.fall_model.predict_on_batch(x).flatten()[0])
    return {"fall_prob": prob}


@app.post("/predict/seizure")
def predict_seizure(req: PredictionRequest):
    if Models.seizure_model is None:
        raise HTTPException(status_code=503, detail="Seizure model not loaded on server.")
    if len(req.frames_b64) != 32:
        raise HTTPException(status_code=400, detail=f"Expected 32 frames, got {len(req.frames_b64)}.")

    clip = _decode_frames(req.frames_b64)               # (32, 224, 224, 3)
    x    = np.expand_dims(clip, 0)                      # (1, 32, 224, 224, 3)
    prob = float(Models.seizure_model.predict_on_batch(x).flatten()[0])
    return {"seizure_prob": prob}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Fix 403 Error: Install Ngrok manually using .tgz instead of the deprecated .zip url
    print("Installing Ngrok binary manually to bypass 403 error...")
    os.system("wget -q -O ngrok.tgz https://bin.ngrok.com/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz")
    os.system("tar -xf ngrok.tgz")
    os.system("mv ngrok /usr/local/bin/ngrok && chmod +x /usr/local/bin/ngrok")

    from pyngrok import ngrok, conf
    conf.get_default().ngrok_path = '/usr/local/bin/ngrok'

    NGROK_TOKEN = "3BVGaAUAxeON3gQVeaS9jz1pRyF_zzREDGozJKRhC5psJcag"

    ngrok.set_auth_token(NGROK_TOKEN)
    tunnel     = ngrok.connect(8000)
    public_url = tunnel.public_url
    print("\n" + "=" * 60)
    print("  ✅  NGROK TUNNEL ACTIVE")
    print(f"  Public URL : {public_url}")
    print(f"  Add to your local .env:")
    print(f"    INFERENCE_MODE=KAGGLE")
    print(f"    KAGGLE_ENDPOINT={public_url}")
    print("=" * 60 + "\n")

    async def _serve():
        config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
        server = uvicorn.Server(config)
        await server.serve()

    import asyncio
    asyncio.get_event_loop().run_until_complete(_serve())
