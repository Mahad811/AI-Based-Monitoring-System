#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Vital Guardian — Launch Script (Ubuntu Native, RTX 4050)
# Usage: bash run_server.sh
# ═══════════════════════════════════════════════════════════════════════════════

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

VENV="$PROJ_DIR/venv"
NVIDIA_SITE="$VENV/lib/python3.10/site-packages/nvidia"

# ── Set LD_LIBRARY_PATH so TF 2.21 can find cuDNN, cuBLAS, etc. ───────────────
export LD_LIBRARY_PATH="\
$NVIDIA_SITE/cudnn/lib:\
$NVIDIA_SITE/cublas/lib:\
$NVIDIA_SITE/cuda_runtime/lib:\
$NVIDIA_SITE/cufft/lib:\
$NVIDIA_SITE/cusolver/lib:\
$NVIDIA_SITE/cusparse/lib:\
$NVIDIA_SITE/curand/lib:\
$NVIDIA_SITE/cuda_cupti/lib:\
$NVIDIA_SITE/nvjitlink/lib:\
$NVIDIA_SITE/nccl/lib:\
${LD_LIBRARY_PATH:-}"

# ── YOLO config dir (avoid /root/.config warning) ─────────────────────────────
export YOLO_CONFIG_DIR="/tmp/.ultralytics"

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║           Vital Guardian — Starting Server          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  Dashboard: http://localhost:8000"
echo "  Login:     admin / admin  |  nurse1 / securepassword"
echo ""

exec "$VENV/bin/python" scripts/demo/demo_server.py
