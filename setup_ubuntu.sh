#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Vital Guardian — Ubuntu Native Setup Script
# Run this from the project root: bash setup_ubuntu.sh
# ═══════════════════════════════════════════════════════════════════════════════

set -e  # Exit on any error
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║        Vital Guardian — Ubuntu Setup Script         ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: System dependencies ──────────────────────────────────────────────
echo "▶ [1/5] Installing system dependencies..."
sudo apt update -qq
sudo apt install -y --no-install-recommends \
    portaudio19-dev \
    ffmpeg \
    python3-dev \
    build-essential \
    libsm6 \
    libxext6 \
    libasound2-dev \
    curl \
    ca-certificates
echo "  ✓ System dependencies installed"

# ── Step 2: Docker (if not already installed) ────────────────────────────────
echo ""
echo "▶ [2/5] Checking Docker installation..."
if ! command -v docker &>/dev/null; then
    echo "  Docker not found. Installing Docker Engine..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker "$USER"
    echo "  ✓ Docker installed."
    echo "  ⚠  NOTE: You must LOGOUT and LOGIN again for docker group to take effect."
    echo "     After re-login, re-run: bash setup_ubuntu.sh --skip-docker"
    DOCKER_NEEDS_RELOGIN=true
else
    echo "  ✓ Docker already installed: $(docker --version)"
    DOCKER_NEEDS_RELOGIN=false
fi

# ── Step 3: Python packages into existing venv ────────────────────────────────
echo ""
echo "▶ [3/5] Installing Python packages into venv..."
if [ ! -f "venv/bin/python" ]; then
    echo "  venv not found, creating..."
    python3 -m venv venv
fi

echo "  Upgrading pip..."
venv/bin/pip install --upgrade pip --quiet

echo "  Installing project requirements..."
# Install in explicit order to avoid conflicts
# torch/torchvision/tensorflow already installed — skip GPU variants to avoid overwrite
venv/bin/pip install --quiet \
    "opencv-python>=4.8.0" \
    "ultralytics>=8.0.0" \
    "openvino>=2024.0.0" \
    "mediapipe>=0.10.0" \
    "tf_keras>=2.14.0" \
    "tensorflow-hub>=0.12.0"

venv/bin/pip install --quiet \
    "librosa>=0.10.0" \
    "soundfile>=0.12.0" \
    "faster-whisper>=1.0.0" \
    "silero-vad" \
    "pyaudio"

venv/bin/pip install --quiet \
    "google-genai>=1.51.0" \
    "pydantic>=2.0.0" \
    "python-dotenv>=1.0.0"

venv/bin/pip install --quiet \
    "fastapi>=0.110.0" \
    "uvicorn>=0.29.0" \
    "httpx>=0.27.0" \
    "flask>=2.3.0" \
    "sqlalchemy>=2.0.0" \
    "psycopg2-binary>=2.9.0"

venv/bin/pip install --quiet \
    "pyyaml>=6.0" \
    "tqdm>=4.65.0" \
    "pandas>=2.0.0" \
    "matplotlib>=3.7.0" \
    "seaborn>=0.12.0" \
    "pillow>=10.0.0"

# Downgrade numpy to < 2.0 if needed for tf-models-official compatibility
# (tensorflow 2.21 may work with numpy 2.x, skip for now unless errors occur)
# venv/bin/pip install "numpy<2.0" --quiet

echo "  ✓ Python packages installed"

# ── Step 4: Start PostgreSQL via Docker ───────────────────────────────────────
echo ""
echo "▶ [4/5] Starting PostgreSQL database via Docker..."
if [ "$DOCKER_NEEDS_RELOGIN" = true ]; then
    echo "  ⚠  Skipping Docker DB start — need re-login first (see Step 2 note above)."
    echo "     After re-login, run: docker compose up -d db"
else
    docker compose up -d db
    echo "  Waiting for PostgreSQL to be healthy..."
    for i in $(seq 1 20); do
        if docker compose ps db | grep -q "healthy"; then
            echo "  ✓ PostgreSQL is healthy and ready"
            break
        fi
        echo "    Waiting... ($i/20)"
        sleep 3
    done
fi

# ── Step 5: Verify setup ─────────────────────────────────────────────────────
echo ""
echo "▶ [5/5] Verifying setup..."

echo -n "  Python venv: "
venv/bin/python --version

echo -n "  PyTorch CUDA: "
venv/bin/python -c "import torch; print(f'torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "PyTorch not available"

echo -n "  TensorFlow: "
venv/bin/python -c "import tensorflow as tf; print(f'tf {tf.__version__} | GPUs: {len(tf.config.list_physical_devices(\"GPU\"))}')" 2>/dev/null || echo "TensorFlow not available"

echo -n "  FastAPI: "
venv/bin/python -c "import fastapi; print(fastapi.__version__)" 2>/dev/null || echo "FastAPI not installed"

echo -n "  Ultralytics (YOLO): "
venv/bin/python -c "import ultralytics; print(ultralytics.__version__)" 2>/dev/null || echo "Ultralytics not installed"

echo -n "  OpenCV: "
venv/bin/python -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "OpenCV not installed"

echo -n "  SQLAlchemy: "
venv/bin/python -c "import sqlalchemy; print(sqlalchemy.__version__)" 2>/dev/null || echo "SQLAlchemy not installed"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║              Setup Complete!                        ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  ⚠  IMPORTANT — Before running the server:"
echo "     1. Set KAGGLE_ENDPOINT in .env if you have a Kaggle notebook running."
echo "        OR change INFERENCE_MODE=LOCAL and place model files in:"
echo "           ./fall_detection/fall_model_best.keras"
echo "           ./seizure_detection/seizure_model_best.keras"
echo "     2. Place demo video clips in ./demo_dataset/"
echo ""
echo "  To run the server:"
echo "     venv/bin/python scripts/demo/demo_server.py"
echo ""
echo "  Dashboard URL: http://localhost:8000"
echo "  Default login: admin / admin"
echo ""
