# ── Vital Guardian — Docker Image ─────────────────────────────────────────────
# Base: Python 3.11 slim (Debian Bookworm).  Keeps the image smaller than
# the full Ubuntu variant while still providing apt for native deps.
FROM python:3.11-slim-bookworm

# ── System-level native libraries ────────────────────────────────────────────
# libgl1 + libglib2.0-0  → OpenCV (cv2) shared objects
# libgomp1               → TensorFlow / PyTorch OpenMP threading
# portaudio19-dev        → PyAudio compile-time requirement (needed even when
#                          AUDIO_ENABLED=false because pip builds the wheel)
# wget / ca-certificates → optional health-check tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        portaudio19-dev \
        gcc \
        python3-dev \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy only requirements first so Docker's layer cache is preserved across
# code-only changes (the expensive pip install step is skipped on rebuilds
# unless requirements.txt actually changes).
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# .dockerignore excludes venv/, datasets/, model binaries, __pycache__ etc.
COPY . .

# ── Runtime defaults ──────────────────────────────────────────────────────────
# These are safe fallback values; docker-compose.yml and .env override them.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    AUDIO_ENABLED=false \
    OPENVINO_DEVICE=intel:cpu

EXPOSE 8000

# demo_server.py uses ROOT = Path(__file__).resolve().parent.parent.parent
# which resolves correctly when run from /app.
CMD ["python", "scripts/demo/demo_server.py"]
