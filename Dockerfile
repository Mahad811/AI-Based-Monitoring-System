# ── Vital Guardian — Docker Image (GPU-enabled) ────────────────────────────────
# Base: NVIDIA CUDA 12.1.1 + cuDNN 8 runtime on Ubuntu 22.04.
# Provides libcuda, libcudnn, libcublas — required by TensorFlow and PyTorch
# for GPU inference inside the container.
#
# ⚠ NOTE: This image requires the NVIDIA Container Toolkit on the host:
#     sudo apt-get install -y nvidia-container-toolkit
#     sudo nvidia-ctk runtime configure --runtime=docker
#     sudo systemctl restart docker
# And docker-compose.yml must declare a 'deploy.resources.reservations.devices'
# GPU block for the app service.
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System-level native libraries ─────────────────────────────────────────────
# python3.11          → application runtime
# libgl1 + libglib2.0 → OpenCV (cv2) shared objects
# libgomp1            → TensorFlow / PyTorch OpenMP threading
# portaudio19-dev     → PyAudio compile-time requirement
# gcc / python3-dev   → build native Python extensions
# wget / ca-certs     → optional health-check tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-dev \
        python3-pip \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        portaudio19-dev \
        gcc \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 / pip3 the system default
RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /app

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first — Docker layer-cache skips the expensive pip install
# on rebuilds when only application source code has changed.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Application source ─────────────────────────────────────────────────────────
# .dockerignore excludes venv/, model binaries, __pycache__, demo_dataset/, etc.
COPY . .

# ── Runtime defaults ───────────────────────────────────────────────────────────
# These are safe fallback values. docker-compose.yml and .env override them.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    AUDIO_ENABLED=false

EXPOSE 8000

# demo_server.py uses ROOT = Path(__file__).resolve().parent.parent.parent
# which resolves correctly when run from /app.
CMD ["python", "scripts/demo/demo_server.py"]
