#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Vital Guardian — One-Time Model Export (GPU)
# Converts .keras → TF SavedModel so the server loads in ~3s with RTX 4050
# Run ONCE:  bash run_export.sh
# ═══════════════════════════════════════════════════════════════════════════════

PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

VENV="$PROJ_DIR/venv"
NVIDIA_SITE="$VENV/lib/python3.10/site-packages/nvidia"

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

export TF_FORCE_GPU_ALLOW_GROWTH=true
export YOLO_CONFIG_DIR=/tmp/.ultralytics

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     Vital Guardian — Model Export (One-Time)        ║"
echo "║     MoViNet .keras → TF SavedModel (RTX 4050)       ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "  This runs ONCE. After this, the server loads models"
echo "  in ~3 seconds with full GPU acceleration."
echo ""

exec "$VENV/bin/python" scripts/export_models.py
