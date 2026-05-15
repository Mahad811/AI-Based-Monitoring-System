#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# Vital Guardian — Docker Launch Script (GPU-accelerated)
# Usage: bash run_docker.sh [--build] [--logs] [--stop] [--status]
#
#   (no args)  Start all services (db + app). Build only if image missing.
#   --build    Force a full image rebuild before starting.
#   --logs     Tail live logs after services start (Ctrl+C safe — won't stop containers).
#   --stop     Gracefully stop and remove all containers.
#   --status   Show running container status + GPU utilisation.
# ═══════════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Resolve project root (script can be called from any directory) ─────────────
PROJ_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJ_DIR"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Parse flags ───────────────────────────────────────────────────────────────
DO_BUILD=false
DO_LOGS=false
DO_STOP=false
DO_STATUS=false

for arg in "$@"; do
    case "$arg" in
        --build)  DO_BUILD=true  ;;
        --logs)   DO_LOGS=true   ;;
        --stop)   DO_STOP=true   ;;
        --status) DO_STATUS=true ;;
        *)
            echo -e "${RED}Unknown flag: $arg${RESET}"
            echo "Usage: bash run_docker.sh [--build] [--logs] [--stop] [--status]"
            exit 1
            ;;
    esac
done

# ─────────────────────────────────────────────────────────────────────────────
# STOP
# ─────────────────────────────────────────────────────────────────────────────
if $DO_STOP; then
    echo -e "\n${YELLOW}Stopping Vital Guardian containers...${RESET}"
    docker compose down
    echo -e "${GREEN}All containers stopped.${RESET}\n"
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# STATUS
# ─────────────────────────────────────────────────────────────────────────────
if $DO_STATUS; then
    echo -e "\n${BOLD}── Container Status ─────────────────────────────────────${RESET}"
    docker compose ps
    echo -e "\n${BOLD}── GPU Utilisation ──────────────────────────────────────${RESET}"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu \
               --format=csv,noheader,nounits 2>/dev/null \
        | awk -F',' '{printf "  GPU: %-30s  Util: %s%%  VRAM: %s/%s MiB  Temp: %s°C\n", $1,$2,$3,$4,$5}' \
        || echo "  nvidia-smi not available"
    echo ""
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║         Vital Guardian — Docker GPU Server          ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# Check Docker daemon
if ! docker info &>/dev/null; then
    echo -e "${RED}✗ Docker daemon is not running. Start it with:${RESET}"
    echo "    sudo systemctl start docker"
    exit 1
fi
echo -e "${GREEN}✓ Docker daemon running${RESET}"

# Check nvidia-container-toolkit
if ! docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 \
        nvidia-smi --query-gpu=name --format=csv,noheader &>/dev/null; then
    echo -e "${RED}✗ GPU passthrough not working. Run once:${RESET}"
    echo "    sudo apt-get install -y nvidia-container-toolkit"
    echo "    sudo nvidia-ctk runtime configure --runtime=docker"
    echo "    sudo systemctl restart docker"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "N/A")
echo -e "${GREEN}✓ GPU passthrough OK — ${GPU_NAME}${RESET}"

# Check .env exists
if [[ ! -f "$PROJ_DIR/.env" ]]; then
    echo -e "${YELLOW}⚠ No .env file found. Copying from .env.example if available...${RESET}"
    [[ -f "$PROJ_DIR/.env.example" ]] && cp "$PROJ_DIR/.env.example" "$PROJ_DIR/.env" \
        || { echo -e "${RED}✗ .env missing and no .env.example found. Aborting.${RESET}"; exit 1; }
fi
echo -e "${GREEN}✓ .env present${RESET}"

echo ""
echo -e "  ${CYAN}GPU:       ${GPU_NAME}${RESET}"
echo -e "  ${CYAN}Dashboard: http://localhost:8000${RESET}"
echo -e "  ${CYAN}Login:     admin / admin  |  nurse1 / securepassword${RESET}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# BUILD (smart: auto-detects when Dockerfile or requirements.txt changed)
# ─────────────────────────────────────────────────────────────────────────────
HASH_FILE="$PROJ_DIR/.docker_build_hash"
CURRENT_HASH=$(sha256sum "$PROJ_DIR/Dockerfile" "$PROJ_DIR/requirements.txt" 2>/dev/null | sha256sum | awk '{print $1}')
STORED_HASH=$(cat "$HASH_FILE" 2>/dev/null || echo "")

if $DO_BUILD; then
    echo -e "${YELLOW}Building Docker image (--build flag set)...${RESET}"
    echo -e "${YELLOW}Note: First build pulls ~5 GB CUDA base layer. Subsequent builds are fast.${RESET}\n"
    docker compose build
    echo "$CURRENT_HASH" > "$HASH_FILE"
elif [[ "$CURRENT_HASH" != "$STORED_HASH" ]]; then
    echo -e "${YELLOW}Dockerfile or requirements.txt changed — rebuilding image...${RESET}\n"
    docker compose build
    echo "$CURRENT_HASH" > "$HASH_FILE"
fi

# ─────────────────────────────────────────────────────────────────────────────
# START
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}Starting services...${RESET}"
docker compose up -d

# Wait for the app container to be healthy / reachable
echo -e "${YELLOW}Waiting for app to be ready (up to 120s)...${RESET}"
MAX_WAIT=120
WAITED=0
until curl -sf http://localhost:8000/api/patients &>/dev/null; do
    sleep 2
    WAITED=$((WAITED + 2))
    if (( WAITED >= MAX_WAIT )); then
        echo -e "\n${RED}✗ App did not become ready in ${MAX_WAIT}s.${RESET}"
        echo -e "  Check logs with:  ${CYAN}docker compose logs app${RESET}"
        exit 1
    fi
    printf "."
done
echo ""
echo -e "\n${GREEN}✓ Vital Guardian is running!${RESET}"
echo ""
echo -e "  ${BOLD}Dashboard:${RESET}  http://localhost:8000"
echo -e "  ${BOLD}View logs:${RESET}  docker compose logs -f app"
echo -e "  ${BOLD}GPU watch:${RESET}  watch -n1 nvidia-smi"
echo -e "  ${BOLD}Stop:${RESET}       bash run_docker.sh --stop"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL: TAIL LOGS
# ─────────────────────────────────────────────────────────────────────────────
if $DO_LOGS; then
    echo -e "${CYAN}Tailing logs (Ctrl+C stops log view only — containers keep running)...${RESET}\n"
    # Trap Ctrl+C so we exit gracefully without stopping containers
    trap 'echo -e "\n${YELLOW}Log view stopped. Containers still running.${RESET}\n"; exit 0' INT
    docker compose logs -f app
fi
