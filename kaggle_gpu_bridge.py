"""
Vital Guardian — Kaggle GPU Bridge
================================================================================
This script runs demo_server.py on Kaggle GPU with ngrok tunneling.

Usage:
    1. Create a new Kaggle notebook
    2. In the first cell, paste the entire contents of this script
    3. Run the cell
    4. Copy the ngrok URL and open it in your browser
    
Requirements:
    - Kaggle notebook GPU enabled (T4 or P100)
    - .env file with GEMINI_API_KEY must be uploaded to Kaggle Datasets
    - Video dataset must be in Kaggle Datasets or uploaded
================================================================================
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

print("=" * 80)
print("VITAL GUARDIAN — KAGGLE GPU BRIDGE")
print("=" * 80)

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Install minimal dependencies for GPU + FastAPI + ngrok
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Installing minimal dependencies...")

packages = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "pngrok>=7.0.0",  # Python wrapper for ngrok
]

for pkg in packages:
    print(f"  Installing {pkg}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("  ✓ Dependencies installed")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Set up paths and environment
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Setting up paths...")

# Kaggle's working directory
KAGGLE_WORK = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")

# If your repo is uploaded as a Kaggle Dataset, mount it:
# For now, assume it's in /kaggle/input/vital-guardian
REPO_ROOT = KAGGLE_INPUT / "vital-guardian"  # Change this to match your dataset name

if not REPO_ROOT.exists():
    print(f"  WARNING: {REPO_ROOT} not found.")
    print("  Please ensure you've uploaded your repo as a Kaggle Dataset and enabled it.")
    REPO_ROOT = KAGGLE_WORK / "vital-guardian"
    REPO_ROOT.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

print(f"  Working directory: {os.getcwd()}")
print(f"  Repo root: {REPO_ROOT}")
print("  ✓ Paths configured")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: Load environment variables (GEMINI_API_KEY from .env)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Loading environment...")

env_path = REPO_ROOT / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print(f"  ✓ Loaded .env from {env_path}")
else:
    print(f"  WARNING: .env not found at {env_path}")
    print("  Make sure GEMINI_API_KEY is set in your environment or .env file")

if not os.getenv("GEMINI_API_KEY"):
    print("  ERROR: GEMINI_API_KEY not set. Gemini verification will fail.")
    print("  Set it as: os.environ['GEMINI_API_KEY'] = 'your-key'")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: Verify GPU availability
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Verifying GPU...")

try:
    import torch
    if torch.cuda.is_available():
        print(f"  ✓ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    else:
        print("  WARNING: No GPU detected. Falling back to CPU (slow).")
except Exception as e:
    print(f"  ERROR: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: Set up ngrok and run the server
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Starting FastAPI server with ngrok...")

# You need to provide your ngrok auth token
# Get it from: https://dashboard.ngrok.com/auth/your-authtoken
NGROK_AUTH_TOKEN = os.getenv("NGROK_AUTH_TOKEN")

if not NGROK_AUTH_TOKEN:
    print("  ERROR: NGROK_AUTH_TOKEN not set.")
    print("  Please add it to your .env or set as environment variable.")
    print("  Get your token from: https://dashboard.ngrok.com/auth/your-authtoken")
    sys.exit(1)

from pyngrok import ngrok

# Authenticate with ngrok
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Import and run the FastAPI app
try:
    from scripts.demo.demo_server import app, service_instance
    print("  ✓ Imported demo_server successfully")
except ImportError as e:
    print(f"  ERROR: Failed to import demo_server: {e}")
    sys.exit(1)

# Start ngrok tunnel on port 8000
print("\n  Starting ngrok tunnel...")
try:
    public_url = ngrok.connect(8000, "http")
    print(f"  ✓ ngrok tunnel active: {public_url}")
    print("\n" + "=" * 80)
    print(f"  PUBLIC URL: {public_url}")
    print("=" * 80)
    print("\n  Open this URL in your browser to access the dashboard:")
    print(f"    {public_url}")
    print("\n  Press Ctrl+C to stop the server.\n")
except Exception as e:
    print(f"  ERROR: Failed to start ngrok: {e}")
    sys.exit(1)

# Run FastAPI server
try:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
except KeyboardInterrupt:
    print("\n\nShutting down...")
    ngrok.kill()
    sys.exit(0)
except Exception as e:
    print(f"ERROR: {e}")
    sys.exit(1)
