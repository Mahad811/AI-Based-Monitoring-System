# Quick Start Guide

This guide reflects the current runtime used by `scripts/demo/demo_server.py`.

## 1) Local Python Run (fastest iteration)

```bash
cd d:\project\FYP_new
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/demo/demo_server.py
```

Requirements:

- PostgreSQL reachable at `DATABASE_URL`
- `.env` with `GEMINI_API_KEY`
- for remote MoViNet mode:
  - `INFERENCE_MODE=KAGGLE`
  - `KAGGLE_ENDPOINT=<your-url>`

Dashboard URL:

- `http://localhost:8000`

## 2) Docker Run (CPU-first)

```bash
cd d:\project\FYP_new
docker compose up -d db
docker compose up -d app
docker compose logs -f app
```

Current compose defaults:

- `OPENVINO_DEVICE=intel:cpu`
- `PERSON_DETECTOR_PROCESS_EVERY=4`
- `MIC_ENABLED=false`
- `AUDIO_ANALYTICS_ENABLED=true`

## 3) Common Runtime Modes

### Mode A: Kaggle fall/seizure inference

In `.env`:

```env
INFERENCE_MODE=KAGGLE
KAGGLE_ENDPOINT=https://...
```

### Mode B: Local fall/seizure inference

In `.env`:

```env
INFERENCE_MODE=LOCAL
```

Ensure local model files exist at paths configured in `config/config.yaml`.

## 4) Useful Commands

```bash
# restart app after .env changes
docker compose up -d --force-recreate app

# watch app logs
docker compose logs -f app

# check status
docker compose ps
```

## 5) Troubleshooting

- `ModuleNotFoundError`: rebuild image if dependency list changed
  - `docker compose build app`
- low Docker FPS:
  - increase `PERSON_DETECTOR_PROCESS_EVERY` (e.g., `4` -> `6`)
- Kaggle mode not responding:
  - verify `KAGGLE_ENDPOINT` is active and reachable
