# Vital Guardian on Kaggle GPU + ngrok Bridge

## Overview
This guide shows how to run **Vital Guardian's GPU-accelerated backend** on Kaggle and expose it to your local browser via **ngrok**.

### Architecture
```
Your Laptop Browser
        ↓
    ngrok URL
        ↓
    ngrok Tunnel
        ↓
    Kaggle GPU (FastAPI + demo_server.py)
        ↓
    Gemini API (verification)
```

---

## Prerequisites

1. **Kaggle Account** (free tier is fine, but GPU runtime required)
2. **Your FYP repo uploaded as a Kaggle Dataset**
3. **ngrok account** (free tier is fine)
4. **Your `.env` file** with `GEMINI_API_KEY`

---

## Step 1: Create a Kaggle Dataset (Your Code)

### Option A: Upload repo as dataset (simplest)
1. Go to [Kaggle.com](https://kaggle.com)
2. Click **Create → Dataset**
3. Name it: `vital-guardian`
4. Upload your entire FYP folder (excluding `venv/`, `__pycache__/`, `.git/`)
5. Make it **private** or **public** (your choice)
6. Copy the **Dataset slug** (appears in the URL: `username/vital-guardian`)

### Option B: Clone from GitHub
If your repo is on GitHub, you can clone it directly in the Kaggle notebook.

---

## Step 2: Get ngrok Auth Token

1. Go to [ngrok.com](https://ngrok.com) and sign up (free)
2. Go to **Auth → Your Authtoken**
3. Copy your token (looks like: `2_abc123xyz...`)
4. Store it in your `.env` file:
   ```
   NGROK_AUTH_TOKEN=2_abc123xyz...
   ```

---

## Step 3: Create Kaggle Notebook

1. Go to **Kaggle → Create → Notebook**
2. Select **Python** language
3. Enable **GPU** (Settings → GPU Accelerator → On)
4. Add your datasets:
   - Add **vital-guardian** dataset (or wherever your repo is)
   - Add any dataset containing your `.env` file (or create one)

---

## Step 4: Set Up Notebook Cell

In the **first cell** of your notebook, paste this (minimal setup):

```python
# Cell 1: Setup environment and run server
import os
import sys
import subprocess
from pathlib import Path

# Mount your dataset
REPO_ROOT = Path("/kaggle/input/vital-guardian")  # Change if different name
ENV_PATH = Path("/kaggle/input/env-secrets/.env")  # If in separate dataset

# Add to path
sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

# Load .env
if ENV_PATH.exists():
    from dotenv import load_dotenv
    load_dotenv(ENV_PATH)
else:
    # Or manually set it:
    os.environ["GEMINI_API_KEY"] = "your-key-here"  # NEVER commit this!

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

In the **second cell**, paste this (install + run):

```python
# Cell 2: Install dependencies and run server
import subprocess
import sys

# Install minimal packages
packages = [
    "fastapi>=0.104.0",
    "uvicorn[standard]>=0.24.0",
    "pydantic>=2.0.0",
    "python-dotenv>=1.0.0",
    "pyyaml>=6.0",
    "pngrok>=7.0.0",
    "google-genai>=1.51.0",  # Gemini API
]

for pkg in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

print("✓ Dependencies installed\n")
```

In the **third cell**, paste this (run server):

```python
# Cell 3: Start ngrok + run server
from pyngrok import ngrok
import os
from dotenv import load_dotenv

# Auth ngrok
NGROK_TOKEN = os.getenv("NGROK_AUTH_TOKEN")
if not NGROK_TOKEN:
    raise ValueError("NGROK_AUTH_TOKEN not found in environment")

ngrok.set_auth_token(NGROK_TOKEN)

# Import app
from scripts.demo.demo_server import app

# Start tunnel
public_url = ngrok.connect(8000, "http")
print("=" * 80)
print(f"PUBLIC URL: {public_url}")
print("=" * 80)
print(f"\nOpen in browser: {public_url}\n")

# Run server
import uvicorn
uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
```

---

## Step 5: Run It

1. Click **Run All** or run cells sequentially
2. **Cell 3** will output your public URL: `https://abc-123-def.ngrok.app`
3. **Open that URL in your browser** on your laptop
4. Watch the dashboard run on Kaggle GPU!

---

## Troubleshooting

### "Module not found: visual_guardian"
Make sure your dataset includes the entire `/visual_guardian` folder. Check that your REPO_ROOT path is correct.

### "GEMINI_API_KEY not found"
Either:
- Upload `.env` as a separate Kaggle Dataset and set `ENV_PATH` correctly, OR
- Paste your key directly (but never commit it):
  ```python
  os.environ["GEMINI_API_KEY"] = "your-actual-key"
  ```

### "ngrok tunnel failed"
- Make sure `NGROK_AUTH_TOKEN` is set
- Check that the token is valid: https://dashboard.ngrok.com/auth/your-authtoken

### Server is slow or crashes
- Check Kaggle notebook logs for GPU memory issues
- If GPU is unavailable, you'll fall back to CPU (slow)
- Reduce frame resolution or FPS in `demo_server.py` if needed

### "ConnectionRefused" or "Connection lost"
- ngrok tunnel times out after ~2 hours of inactivity (free tier)
- Just re-run the notebook cell to get a new URL

---

## Performance Expectations

**On Kaggle GPU (T4 or P100):**
- YOLOv8: ~5-10ms (vs ~20ms CPU)
- 5 EfficientNet models: ~15ms (vs ~150ms CPU)
- 10 seizure models: ~30ms (vs ~300ms CPU)
- **Overall: 30-40 FPS** (vs 5 FPS on CPU)

Frame encoding + WebSocket overhead may still slow this down slightly, but it should be **much faster** than local CPU.

---

## Optional: Clean Up

To stop the server and close the tunnel:
- In Kaggle: Press **Stop** button (or Ctrl+C in notebook)
- ngrok tunnel will automatically close

---

## Security Notes

- **Never commit `.env` with your actual keys** to GitHub
- Use Kaggle Datasets to store secrets (private)
- Or pass them as environment variables at runtime
- ngrok URLs are public but long/unguessable (free tier security by obscurity)

---

## Next Steps

Once this works:
1. Optimize frame resolution/FPS if still slow
2. Test with live camera input (instead of video files)
3. Deploy to a permanent GPU VM (e.g., RunPod) if you need 24/7 hosting
