# Vital Guardian: ICU Patient Monitoring System

Vital Guardian is a real-time multimodal monitoring system for ICU scenarios. It combines:

- vision inference (fall, seizure, person detection),
- audio distress analysis,
- Gemini-based verification and clinical enrichment,
- a FastAPI/WebSocket dashboard with PostgreSQL persistence.

## Current Runtime Model

- **Vision person detection:** YOLO11n (OpenVINO backend).
- **Fall and seizure classifiers:** MoViNet-based models, either:
  - `INFERENCE_MODE=LOCAL` (local model files), or
  - `INFERENCE_MODE=KAGGLE` (remote endpoint via `KAGGLE_ENDPOINT`).
- **Audio pipeline:** distress + keyword analysis with support for injected audio from demo clips.
- **Cognitive pipeline:** Tier-2 confirm/suppress and Tier-3 enriched report.

## Quick Start (Local Python Runtime)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/demo/demo_server.py
```

Requirements:

- PostgreSQL reachable at `DATABASE_URL` (or fallback defaults in `scripts/demo/database.py`)
- `.env` with `GEMINI_API_KEY`
- if `INFERENCE_MODE=KAGGLE`, set `KAGGLE_ENDPOINT`

Dashboard:

- `http://localhost:8000`

## Quick Start (Docker Runtime)

```bash
docker compose up -d db
docker compose up -d app
docker compose logs -f app
```

Notes:

- Docker setup is CPU-first (`OPENVINO_DEVICE=intel:cpu`).
- `PERSON_DETECTOR_PROCESS_EVERY=4` is configured in compose for stable FPS.
- `MIC_ENABLED=false` and `AUDIO_ANALYTICS_ENABLED=true` allow clip-audio analytics without live microphone access.

## Repository Layout

| Path | Purpose |
|:---|:---|
| `visual_guardian/` | Vision pipeline, person detector, temporal encoders, fall/seizure classifiers |
| `auditory_watchdog/` | Audio capture/injection, privacy shield, distress + keyword models |
| `cognitive_core/` | Gemini verification and incident enrichment |
| `scripts/demo/` | FastAPI app, WebSocket orchestration, DB models |
| `scripts/demo/public/` | Dashboard frontend static assets |
| `config/` | Runtime configuration (`config.yaml`) |
| `docs/` | Architecture notes, guides, analysis, and deprecated references |

## Documentation

- [Technical Documentation](technical_documentation.md)
- [Quick Start Guide](docs/QUICK_START.md)
- [Architecture Overview](docs/guides/ARCHITECTURE_OVERVIEW.md)
- [Architecture Evolution](docs/Architecture_Evolution.md)

## Status

Actively maintained.
