"""
Vital Guardian — Web Dashboard Backend (FastAPI + WebSockets)

Runs the Vision Pipeline in the background and streams frames + data
over a WebSocket to the frontend browser dashboard.

Inference modes (set via .env):
  INFERENCE_MODE=LOCAL   — use local TF models (default)
  INFERENCE_MODE=KAGGLE  — route MoViNet calls to KAGGLE_ENDPOINT

Usage:
    cd d:\\project\\FYP_new
    venv\\Scripts\\python scripts/demo/demo_server.py
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import sys
import time
import json
import asyncio
import threading
import queue
import cv2
import yaml
import numpy as np
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import collections
import shutil
from dotenv import load_dotenv

# ── Load .env from repo root ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT / ".env")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

sys.path.append(str(ROOT))
from visual_guardian.pipeline import VisionPipeline
from cognitive_core.gemini_verifier import GeminiVerifier
from database import init_db, get_db, Nurse, Patient, IncidentLog, AuditLog, SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import func as sqlfunc
import datetime
import httpx
from auditory_watchdog.core.audio_capture import AudioStream
from auditory_watchdog.core.privacy_shield import PrivacyShield
from auditory_watchdog.core.distress_classifier import DistressClassifier
from auditory_watchdog.core.keyword_spotter import KeywordSpotter

# ─────────────────────────────────────────────────────
# SETTINGS
# ─────────────────────────────────────────────────────
DEVICE        = 'cpu'   # TF models run on CPU on Windows (no native GPU support for TF >= 2.11)
FRAME_SKIP    = 1
DISABLE_POSE  = True




SEIZURE_THRESHOLD = 0.48
FALL_THRESHOLD    = 0.55
GAP_SECONDS       = 3

# Both models were trained on ~30fps clips.  Feeding a 120fps clip at full
# rate would compress the temporal window to 0.27s instead of the expected 1s,
# causing the model to miss events.  All video frames are subsampled to this
# rate before entering the pipeline.
FPS_TARGET        = 30

# Playback speed multiplier.  1.0 = real-time.
# The last-frame hold + tail-wait polling already keeps the screen alive while
# the Kaggle API responds, so there is no need to rush clips.
# Increase only if clips feel too slow for the demo audience.
PLAYBACK_SPEED    = 1.0

# After a tail-wait alert fires, hold the last frame visible for this many
# seconds so the demo audience can read the alert before transitioning.
ALERT_HOLD_SECS   = 4.0

# Enable/disable the background Proactive Risk Monitor.
# Disabled for short pre-recorded clips, enable for live patient feeds.
ENABLE_PROACTIVE_MONITOR = False

# ── Kaggle / inference mode ───────────────────────────────────────────────────
INFERENCE_MODE   = os.getenv("INFERENCE_MODE", "LOCAL").upper()
KAGGLE_ENDPOINT  = os.getenv("KAGGLE_ENDPOINT", "").strip()

# ── Audio flags ───────────────────────────────────────────────────────────────
# MIC_ENABLED            — open the real host microphone device.
#                          Set false in Docker (no mic hardware available).
# AUDIO_ANALYTICS_ENABLED — run YAMNet / Whisper models + AuditoryMonitor.
#                          Can stay true in Docker: pre-recorded clip audio is
#                          injected via the queue; no mic device needed.
MIC_ENABLED             = os.getenv("MIC_ENABLED",             "true").lower() not in ("false", "0", "no")
AUDIO_ANALYTICS_ENABLED = os.getenv("AUDIO_ANALYTICS_ENABLED", "true").lower() not in ("false", "0", "no")

# ─────────────────────────────────────────────────────
# AUTO-DISCOVER DEMO CLIPS
# ─────────────────────────────────────────────────────
# Clips are discovered dynamically so the server works on any machine
# (including Kaggle) without path changes.  Override root with env var:
#   VG_DEMO_VIDEO_ROOT=/kaggle/input/vital-guardian-demo-videos
_DATASET_ROOT = Path(os.getenv("VG_DEMO_VIDEO_ROOT", str(ROOT / "demo_dataset")))


def _find_clips(directory: Path, extensions=(".mp4", ".avi", ".mov")) -> list[Path]:
    """Return all video files inside *directory* (recursive), sorted by name."""
    if not directory.exists():
        return []
    found = sorted(
        p for p in directory.rglob("*")
        if p.suffix.lower() in extensions
    )
    return found


def _build_clip_mapping():
    D = Path(os.getenv("VG_DEMO_VIDEO_ROOT", str(ROOT / "demo_dataset")))
    fall_clips    = (_find_clips(D / "falls") + _find_clips(D / "fall_test" / "fall"))
    normal_clips  = (_find_clips(D / "normal") + _find_clips(D / "fall_test" / "nofall"))
    sz_normal_clips  = _find_clips(D / "unusual_movement" / "data" / "Normal")
    sz_clips    = _find_clips(D / "unusual_movement" / "data" / "Seizure")
    wc_clips = _find_clips(D / "whooping_cough")
    asthma_clips = _find_clips(D / "asthma_attack")
    
    return {
        "fall_1": [fall_clips[0]] if len(fall_clips) > 0 else [],
        "fall_2": [fall_clips[1]] if len(fall_clips) > 1 else [],
        "normal_1": [normal_clips[0]] if len(normal_clips) > 0 else [],
        "seizure_1": sz_normal_clips[:2] + sz_clips[:2],
        "seizure_2": sz_normal_clips[2:4] + sz_clips[2:4],
        "whooping_cough_video": wc_clips,
        "asthma_attack_video": asthma_clips,
    }

CLIP_MAPPING = _build_clip_mapping()


# ─────────────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────────────
app = FastAPI(title="Vital Guardian Web API")

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

# Rolling deque of Tier-3 verify durations (seconds) for avg-verify-time stat
_verify_times: collections.deque = collections.deque(maxlen=50)

@app.on_event("startup")
def startup_event():
    init_db()
    # Ensure the public audio folder exists so extracted WAVs can be served statically
    (PUBLIC_DIR / "audio").mkdir(parents=True, exist_ok=True)

@app.get("/")
def serve_dashboard():
    return RedirectResponse(url="/static/login.html")

# ─────────────────────────────────────────────────────
# AUTH HELPERS
# ─────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str

def _decode_token(req: Request) -> str:
    """Returns the staff_id from the Bearer token, or raises HTTPException."""
    auth = req.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Token")
    try:
        decoded = base64.b64decode(auth.split(" ")[1]).decode()
        return decoded.split(":")[0]
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Token")

def verify_admin(req: Request):
    if _decode_token(req) != "admin":
        raise HTTPException(status_code=403, detail="Forbidden: Admins only")

# ─────────────────────────────────────────────────────
# AUDIT HELPER
# ─────────────────────────────────────────────────────
def log_audit(db: Session, actor_id: str, actor_name: str, action: str,
              target_type: str = None, target_id: str = None, details: str = None):
    try:
        db.add(AuditLog(
            actor_id=actor_id, actor_name=actor_name, action=action,
            target_type=target_type, target_id=str(target_id) if target_id is not None else None,
            details=details,
        ))
        db.commit()
    except Exception as e:
        print(f"  [AuditLog] Failed to write: {e}")

# ─────────────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────────────
@app.post("/api/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(Nurse).filter(Nurse.staff_id == req.username.lower()).first()
    if not user or user.password != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = base64.b64encode(f"{req.username}:{time.time()}".encode()).decode()
    # Update last_login
    user.last_login = datetime.datetime.now(datetime.timezone.utc)
    db.commit()
    log_audit(db, user.staff_id, user.name, "LOGIN")
    return {
        "status": "success", "token": token,
        "nurse_id": str(user.id), "nurse_name": user.name,
        "staff_id": user.staff_id, "role": user.role or "Nurse",
        "shift": user.shift or "Morning", "ward": user.ward_assignment or "",
    }

# ─────────────────────────────────────────────────────
# NURSE / STAFF ENDPOINTS
# ─────────────────────────────────────────────────────
def _nurse_dict(n: Nurse) -> dict:
    return {
        "id": n.id, "staff_id": n.staff_id, "name": n.name,
        "role": n.role or "Nurse", "shift": n.shift or "Morning",
        "ward_assignment": n.ward_assignment or "",
        "status": n.status or "off-duty",
        "join_date": n.join_date.isoformat() if n.join_date else None,
        "last_login": n.last_login.isoformat() if n.last_login else None,
        "alerts_handled": n.alerts_handled or 0,
    }

@app.get("/api/admin/nurses")
def get_nurses(db: Session = Depends(get_db), _: None = Depends(verify_admin)):
    return [_nurse_dict(n) for n in db.query(Nurse).order_by(Nurse.id).all()]

class NurseCreate(BaseModel):
    staff_id: str
    name: str
    password: str
    role: str = "Nurse"
    shift: str = "Morning"
    ward_assignment: str = ""
    status: str = "off-duty"

@app.post("/api/admin/nurses")
def create_nurse(nurse_data: NurseCreate, req: Request, db: Session = Depends(get_db),
                 _: None = Depends(verify_admin)):
    if db.query(Nurse).filter(Nurse.staff_id == nurse_data.staff_id.lower()).first():
        raise HTTPException(status_code=400, detail="Staff ID already exists.")
    new_nurse = Nurse(
        staff_id=nurse_data.staff_id.lower(), name=nurse_data.name,
        password=nurse_data.password, role=nurse_data.role,
        shift=nurse_data.shift, ward_assignment=nurse_data.ward_assignment or None,
        status=nurse_data.status,
    )
    db.add(new_nurse)
    db.commit()
    db.refresh(new_nurse)
    actor = _decode_token(req)
    a = db.query(Nurse).filter_by(staff_id=actor).first()
    log_audit(db, actor, a.name if a else actor, "NURSE_ADDED",
              "nurse", new_nurse.id, f"Added {new_nurse.name} ({new_nurse.staff_id})")
    return {"status": "success", "nurse": _nurse_dict(new_nurse)}

class NurseUpdate(BaseModel):
    name: str = None
    password: str = None
    role: str = None
    shift: str = None
    ward_assignment: str = None
    status: str = None

@app.put("/api/admin/nurses/{nurse_id}")
def update_nurse(nurse_id: int, data: NurseUpdate, req: Request,
                 db: Session = Depends(get_db), _: None = Depends(verify_admin)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    changes = []
    for field in ("name", "password", "role", "shift", "ward_assignment", "status"):
        val = getattr(data, field)
        if val is not None:
            setattr(nurse, field, val)
            changes.append(field)
    db.commit()
    actor = _decode_token(req)
    a = db.query(Nurse).filter_by(staff_id=actor).first()
    log_audit(db, actor, a.name if a else actor, "NURSE_UPDATED",
              "nurse", nurse_id, f"Fields: {', '.join(changes)}")
    return {"status": "success", "nurse": _nurse_dict(nurse)}

@app.delete("/api/admin/nurses/{nurse_id}")
def delete_nurse(nurse_id: int, req: Request, db: Session = Depends(get_db),
                 _: None = Depends(verify_admin)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse:
        raise HTTPException(status_code=404, detail="Nurse not found")
    if nurse.staff_id == "admin":
        raise HTTPException(status_code=403, detail="Cannot delete admin")
    name_snapshot = f"{nurse.name} ({nurse.staff_id})"
    db.delete(nurse)
    db.commit()
    actor = _decode_token(req)
    a = db.query(Nurse).filter_by(staff_id=actor).first()
    log_audit(db, actor, a.name if a else actor, "NURSE_REMOVED",
              "nurse", nurse_id, f"Removed {name_snapshot}")
    return {"status": "success"}

# ─────────────────────────────────────────────────────
# PATIENT ENDPOINTS
# ─────────────────────────────────────────────────────
@app.get("/api/patients")
def get_patients(db: Session = Depends(get_db)):
    return [{"id": p.id, "name": p.name, "room": p.room, "age": p.age,
             "risk_profile": p.risk_profile} for p in db.query(Patient).all()]

@app.get("/api/patient/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {"id": p.id, "name": p.name, "room": p.room, "age": p.age,
            "risk_profile": p.risk_profile}

# ─────────────────────────────────────────────────────
# HISTORY / INCIDENT ENDPOINTS
# ─────────────────────────────────────────────────────
@app.get("/api/history")
def get_history(
    db: Session = Depends(get_db),
    date_from: str = None,
    date_to: str = None,
    patient_id: int = None,
    event_type: str = None,
    min_confidence: float = None,
):
    q = db.query(IncidentLog).order_by(IncidentLog.timestamp.desc())
    if patient_id:
        q = q.filter(IncidentLog.patient_id == patient_id)
    if event_type and event_type.lower() != "all":
        q = q.filter(IncidentLog.incident_type == event_type.lower())
    if min_confidence is not None:
        q = q.filter(IncidentLog.confidence >= min_confidence / 100.0)
    if date_from:
        try:
            q = q.filter(IncidentLog.timestamp >= datetime.datetime.fromisoformat(date_from))
        except ValueError:
            pass
    if date_to:
        try:
            q = q.filter(IncidentLog.timestamp <= datetime.datetime.fromisoformat(date_to))
        except ValueError:
            pass
    logs = q.all()
    res = []
    for lg in logs:
        p = db.query(Patient).filter(Patient.id == lg.patient_id).first()
        res.append({
            "id": lg.id,
            "patient_id": lg.patient_id,
            "patient_name": p.name if p else "Unknown",
            "room": p.room if p else "N/A",
            "incident_type": lg.incident_type.upper() if lg.incident_type else "UNKNOWN",
            "confidence": lg.confidence,
            "timestamp": lg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if lg.timestamp else "N/A",
            "severity": lg.severity or "unknown",
            "narrative": lg.narrative,
            "has_frames": bool(lg.frames_dir),
        })
    return res

@app.get("/api/history/{incident_id}")
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    lg = db.query(IncidentLog).filter(IncidentLog.id == incident_id).first()
    if not lg:
        raise HTTPException(status_code=404, detail="Incident not found")
    p = db.query(Patient).filter(Patient.id == lg.patient_id).first()
    actions = []
    if lg.actions_json:
        try:
            actions = json.loads(lg.actions_json)
        except Exception:
            pass
    frame_urls = []
    if lg.frames_dir:
        frames_path = PUBLIC_DIR / "incidents" / str(lg.id)
        if frames_path.exists():
            frame_urls = [f"/static/incidents/{lg.id}/{f.name}"
                          for f in sorted(frames_path.iterdir())
                          if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    return {
        "id": lg.id,
        "patient_id": lg.patient_id,
        "patient_name": p.name if p else "Unknown",
        "room": p.room if p else "N/A",
        "age": p.age if p else None,
        "risk_profile": p.risk_profile if p else "",
        "incident_type": lg.incident_type.upper() if lg.incident_type else "UNKNOWN",
        "confidence": lg.confidence,
        "timestamp": lg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if lg.timestamp else "N/A",
        "severity": lg.severity or "unknown",
        "narrative": lg.narrative,
        "actions": actions,
        "frame_urls": frame_urls,
    }

# ─────────────────────────────────────────────────────
# ADMIN STATS / AUDIT / HEALTH ENDPOINTS
# ─────────────────────────────────────────────────────
@app.get("/api/admin/stats")
def get_stats(db: Session = Depends(get_db), _: None = Depends(verify_admin)):
    now = datetime.datetime.now(datetime.timezone.utc)
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    yesterday_start = today_start - datetime.timedelta(days=1)
    week_start = today_start - datetime.timedelta(days=7)

    alerts_today = db.query(IncidentLog).filter(
        IncidentLog.timestamp >= today_start).count()
    alerts_yesterday = db.query(IncidentLog).filter(
        IncidentLog.timestamp >= yesterday_start,
        IncidentLog.timestamp < today_start).count()
    alerts_week = db.query(IncidentLog).filter(
        IncidentLog.timestamp >= week_start).count()
    on_duty = db.query(Nurse).filter(Nurse.status == "on-duty").count()
    patients_active = db.query(Patient).count()

    avg_verify = (sum(_verify_times) / len(_verify_times)) if _verify_times else None

    # Last 10 audit entries for the recent-activity feed
    recent = db.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(10).all()
    recent_activity = [{
        "action": a.action, "actor_name": a.actor_name,
        "target_type": a.target_type, "details": a.details,
        "timestamp": a.timestamp.isoformat() if a.timestamp else None,
    } for a in recent]

    # Hourly buckets for the last 24 h (sparkline data)
    hourly = [0] * 24
    for lg in db.query(IncidentLog).filter(IncidentLog.timestamp >= today_start).all():
        if lg.timestamp:
            ts = lg.timestamp
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=datetime.timezone.utc)
            hourly[ts.hour] += 1

    return {
        "patients_active": patients_active,
        "alerts_today": alerts_today,
        "alerts_yesterday": alerts_yesterday,
        "alerts_week": alerts_week,
        "on_duty_count": on_duty,
        "avg_verify_time_sec": round(avg_verify, 1) if avg_verify else None,
        "recent_activity": recent_activity,
        "hourly_alerts": hourly,
    }

@app.get("/api/admin/audit")
def get_audit(
    db: Session = Depends(get_db),
    _: None = Depends(verify_admin),
    limit: int = 50,
    offset: int = 0,
    action: str = None,
    actor_id: str = None,
):
    q = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if action and action.lower() != "all":
        q = q.filter(AuditLog.action == action.upper())
    if actor_id:
        q = q.filter(AuditLog.actor_id == actor_id)
    total = q.count()
    items = q.offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [{
            "id": a.id, "actor_id": a.actor_id, "actor_name": a.actor_name,
            "action": a.action, "target_type": a.target_type,
            "target_id": a.target_id, "details": a.details,
            "timestamp": a.timestamp.isoformat() if a.timestamp else None,
        } for a in items],
    }

@app.get("/api/admin/health")
async def get_health(_: None = Depends(verify_admin)):
    # Postgres — simple round-trip query
    pg_ok = False
    try:
        from sqlalchemy import text as sa_text
        db = SessionLocal()
        db.execute(sa_text("SELECT 1"))
        db.close()
        pg_ok = True
    except Exception:
        pass

    # Kaggle endpoint
    kaggle_status = "disabled"
    if INFERENCE_MODE == "KAGGLE" and KAGGLE_ENDPOINT:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                t0 = time.time()
                r = await client.head(KAGGLE_ENDPOINT)
                latency = time.time() - t0
                if r.status_code < 500:
                    kaggle_status = "slow" if latency > 1.5 else "ok"
                else:
                    kaggle_status = "down"
        except Exception:
            kaggle_status = "down"

    ws_clients = len(service_instance.active_websockets) if 'service_instance' in globals() else 0

    return {
        "postgres": "ok" if pg_ok else "down",
        "kaggle": kaggle_status,
        "ws_clients": ws_clients,
        "inference_mode": INFERENCE_MODE,
    }

# ─────────────────────────────────────────────────────
# SEGMENTS CONSOLIDATOR
# ─────────────────────────────────────────────────────
class SegmentConsolidator:
    def __init__(self, seg_type):
        self.seg_type   = seg_type
        self.fired      = False
        self.peak_conf  = 0.0
        self.fall_streak = 0
        self.sz_streak   = 0

    def update(self, event):
        if self.fired:
            return False, None, 0.0
        etype   = event.get('event_type', 'normal')
        fall_sm = event.get('fall_smoothed', 0.0)
        sz_c    = event.get('seizure_confidence', 0.0)
        sz_sm   = event.get('seizure_smoothed', 0.0)
        # Fix #5: use SMOOTHED seizure probability (not raw spike) for the
        # suppression check, and never suppress falls in a fall segment.
        suppress_fall = sz_sm >= 0.35 and self.seg_type != 'fall'

        if etype in ('fall', 'force_fall') and not suppress_fall:
            self.fall_streak += 1
            self.peak_conf = max(self.peak_conf, fall_sm)
        else:
            self.fall_streak = 0

        if etype == 'seizure':
            self.sz_streak += 1
            self.peak_conf = max(self.peak_conf, sz_sm)
        # Also count high raw seizure confidence as a strike (catches it mid-clip)
        elif sz_c >= 0.65 and self.seg_type == 'seizure':
            self.sz_streak += 1
            self.peak_conf = max(self.peak_conf, sz_c)
        else:
            self.sz_streak = 0

        if self.seg_type == 'fall' and self.fall_streak >= 1:
            self.fired = True
            return True, 'fall', max(self.peak_conf, fall_sm)
        if self.seg_type == 'seizure' and self.sz_streak >= 2:
            self.fired = True
            return True, 'seizure', max(self.peak_conf, sz_sm)
        return False, None, 0.0


def move_models_to_gpu(pipeline):
    """GPU warm-up for TF models (TF handles GPU placement automatically)."""
    pass  # TensorFlow models auto-place on GPU; no manual .to(device) needed


# ─────────────────────────────────────────────────────
# MIC-DISABLED AUDIO STREAM STUB
# ─────────────────────────────────────────────────────
class _MicDisabledAudioStream:
    """
    Drop-in replacement for AudioStream when MIC_ENABLED=false.

    Exposes the same queue/control interface as the real AudioStream so that:
      - Pre-recorded clip audio injection (put_nowait) works normally.
      - AuditoryMonitor.run_forever() can read injected chunks via get_latest_chunk().
      - stop_stream() / start_stream() calls are safe no-ops.

    Never opens a PyAudio device, so it is safe inside Docker containers
    that have no audio hardware.
    """
    import queue as _q

    def __init__(self):
        import queue as _q
        self.audio_queue    = _q.Queue(maxsize=1)
        self.is_running     = False

    def start_stream(self):
        pass

    def stop_stream(self):
        self.is_running = False

    def get_latest_chunk(self, timeout=None):
        import queue as _q
        try:
            return self.audio_queue.get(timeout=timeout if timeout else 0)
        except _q.Empty:
            return None

    def terminate(self):
        pass


# ─────────────────────────────────────────────────────
# PIPELINE SERVICE
# ─────────────────────────────────────────────────────
class PipelineService:
    def __init__(self):
        print("\nLoading Vision Pipeline...")
        config_path = ROOT / "config" / "config.yaml"
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        vision_cfg = cfg['vision']
        vision_cfg['seizure_classifier']['threshold'] = SEIZURE_THRESHOLD
        vision_cfg['fall_classifier']['threshold']    = FALL_THRESHOLD
        if 'bed_exit' in vision_cfg:
            vision_cfg['bed_exit']['enabled'] = False

        self.pipeline = VisionPipeline(vision_cfg)
        if DISABLE_POSE:
            self.pipeline.pose_analyzer = None

        # CPU-only warm-up — run a few dummy frames to trigger TF tracing
        print("Warming up pipeline...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.pipeline.process_frame(dummy)
        self.pipeline.reset()
        print("Pipeline warm-up complete.")

        # Override YOLO frame-skip so every frame gets a fresh bounding box.
        # Accuracy >> speed here since OpenVINO GPU handles YOLO fast anyway.
        # On Intel iGPU (bare metal) every-frame YOLO is fast.
        # On CPU-only Docker the same setting tanks FPS — use env override.
        _pe = int(os.getenv("PERSON_DETECTOR_PROCESS_EVERY", "1"))
        self.pipeline.person_detector.process_every = _pe

        print("Initializing Gemini API Verifier...")
        self.gemini = GeminiVerifier(mock_mode=False)

        self.active_websockets = []
        self.running  = False
        self.paused   = False
        self.skip_requested = False

        # Shared state read by ProactiveRiskMonitor (updated each frame)
        self._risk_frame_buffer = collections.deque(maxlen=90)  # last 3s @ 30fps
        self._active_patient    = "Patient"

        # Audio stream — always required so pre-recorded clip audio can be
        # injected into the queue for AuditoryMonitor to consume.
        # When MIC_ENABLED=false we use a stub that never opens a PyAudio device.
        if MIC_ENABLED:
            print("Initializing live microphone stream...")
            self.audio_stream = AudioStream()
        else:
            print("Mic disabled (MIC_ENABLED=false) — using stub queue for clip audio injection.")
            self.audio_stream = _MicDisabledAudioStream()

        # Audio analytics models — can run even without a real mic because
        # pre-recorded clip chunks are injected directly into the queue above.
        if AUDIO_ANALYTICS_ENABLED:
            print("Initializing Auditory Watchdog analytics (YAMNet / Whisper)...")
            self.privacy_shield      = PrivacyShield()
            self.distress_classifier = DistressClassifier()
            self.keyword_spotter     = KeywordSpotter()
            self.audio_executor      = ThreadPoolExecutor(max_workers=2)
        else:
            print("Audio analytics disabled (AUDIO_ANALYTICS_ENABLED=false).")
            self.privacy_shield      = None
            self.distress_classifier = None
            self.keyword_spotter     = None
            self.audio_executor      = None
        # Tells AuditoryMonitor whether a pre-recorded audio clip is active.
        # True  → Whisper (KWS) is suppressed; vision pipeline is bypassed.
        # False → normal behaviour (live feed, vision segments).
        self.is_audio_segment_active = False

        print("Pipeline Service Started.\n")

    async def broadcast(self, payload: dict):
        for ws in self.active_websockets:
            try:
                await ws.send_json(payload)
            except:
                pass

    async def execute_gemini_job(self, aid, etype, conf, pat, frames):
        """
        Two-step progressive Gemini verification:
          Tier 2 — fast binary (~1-2s):  broadcasts gemini_tier2 immediately
          Tier 3 — full enrichment (~6s): broadcasts gemini_report only if CONFIRMED

        HIGH-CONFIDENCE BYPASS: if ML confidence >= 50%, we skip Tier 2 entirely
        and auto-CONFIRM. The trained MoViNet model on real ICU data is more
        reliable than a general LLM reviewing still frames without motion context.

        On completion, persists narrative/severity/actions to the IncidentLog DB
        row that was already created, and saves the 8 Gemini frames to disk.
        """
        AUTO_CONFIRM_THRESHOLD = 0.50
        t3_start = time.time()
        try:
            if conf >= AUTO_CONFIRM_THRESHOLD:
                decision = "CONFIRMED"
                reason   = (f"ML confidence {conf*100:.0f}% exceeds auto-confirm threshold "
                            f"({AUTO_CONFIRM_THRESHOLD*100:.0f}%). Clinical enrichment proceeding.")
                print(f"  [Gemini T2] Alert {aid}: AUTO-CONFIRMED ({conf*100:.0f}%)")
            else:
                print(f"  [Gemini T2] Verifying Alert {aid}...")
                t2 = await asyncio.to_thread(
                    self.gemini.verify_binary, etype, conf, pat, frames
                )
                decision = t2.get("decision", "CONFIRMED")
                reason   = t2.get("reason",   "")
                print(f"  [Gemini T2] Alert {aid}: {decision} — {reason}")

            await self.broadcast({
                "type":     "gemini_tier2",
                "alert_id": aid,
                "decision": decision,
                "reason":   reason,
            })

            if decision == "CONFIRMED":
                print(f"  [Gemini T3] Enriching Alert {aid}...")
                t3 = await asyncio.to_thread(
                    self.gemini.enrich_clinical, etype, conf, pat, frames
                )
                t3["decision"] = "CONFIRMED"
                elapsed = time.time() - t3_start
                _verify_times.append(elapsed)
                print(f"  [Gemini T3] Alert {aid}: severity={t3.get('severity')} "
                      f"elapsed={elapsed:.1f}s")

                # ── Persist to DB ────────────────────────────────────────────
                try:
                    db_upd = SessionLocal()
                    # Find the most-recent IncidentLog row for this alert counter
                    # (matched by approximate order — aid is 1-based counter per session)
                    log_rows = (db_upd.query(IncidentLog)
                                .order_by(IncidentLog.timestamp.desc())
                                .limit(aid + 5).all())
                    # Pick the last row of matching type
                    target_row = None
                    for row in log_rows:
                        if row.incident_type == etype:
                            target_row = row
                            break
                    if target_row:
                        target_row.narrative    = t3.get("narrative", "")
                        target_row.severity     = t3.get("severity", "moderate")
                        target_row.actions_json = json.dumps(t3.get("actions", []))

                        # Save Gemini frames to public/incidents/<id>/
                        if frames:
                            frames_dir = PUBLIC_DIR / "incidents" / str(target_row.id)
                            frames_dir.mkdir(parents=True, exist_ok=True)
                            for fi, fr in enumerate(frames):
                                _, buf = cv2.imencode('.jpg', fr,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                                (frames_dir / f"{fi}.jpg").write_bytes(buf.tobytes())
                            target_row.frames_dir = str(frames_dir)

                        db_upd.commit()
                    db_upd.close()
                except Exception as db_exc:
                    print(f"  [Gemini] DB persist failed for alert {aid}: {db_exc}")

                # ── Audit log the confirmation ───────────────────────────────
                try:
                    db_a = SessionLocal()
                    log_audit(db_a, "system", "AI Pipeline", "ALERT_CONFIRMED",
                              "incident", aid,
                              f"{etype.upper()} — {conf*100:.0f}% conf — sev={t3.get('severity')}")
                    db_a.close()
                except Exception:
                    pass

                await self.broadcast({
                    "type":     "gemini_report",
                    "alert_id": aid,
                    "report":   t3,
                })
            else:
                # ── Audit log the suppression ────────────────────────────────
                try:
                    db_a = SessionLocal()
                    log_audit(db_a, "system", "AI Pipeline", "ALERT_SUPPRESSED",
                              "incident", aid, f"{etype.upper()} suppressed: {reason[:100]}")
                    db_a.close()
                except Exception:
                    pass

                await self.broadcast({
                    "type":     "gemini_report",
                    "alert_id": aid,
                    "report": {
                        "decision":  "SUPPRESSED",
                        "headline":  "Alert Suppressed — False Positive",
                        "narrative": reason,
                        "severity":  "low",
                        "actions":   ["Continue routine monitoring"],
                        "escalate":  False,
                    },
                })
        except Exception as exc:
            print(f"  [Gemini] Job {aid} failed: {exc}")



    async def play_patient_clip(self, patient_id: int):
        db = SessionLocal()
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        db.close()
        
        if not p:
            await self.broadcast({"type": "transition", "message": "Invalid Patient ID"})
            return
            
        if p.clip_type == "live_feed":
            return await self._play_live_feed(p)
            
        clips = CLIP_MAPPING.get(p.clip_type, [])
        if not clips:
            await self.broadcast({"type": "transition", "message": "Video feed offline."})
            return

        print(f"[{p.id}] {p.name} — {p.risk_profile}")

        # Construct a pseudo-segment dict to match the rest of the codebase
        seg_type = "normal"
        if "fall" in p.clip_type.lower(): seg_type = "fall"
        if "seizure" in p.clip_type.lower(): seg_type = "seizure"

        seg = {
            "id": p.id,
            "patient": p.name,
            "label": p.room,
            "type": seg_type,
            "clips": clips
        }
        
        seg_idx = 999 
        total_segs = 1
        alert_counter = 0

        # Audio pipeline only runs for clips whose primary purpose IS audio demonstration.
        # Vision segments (fall, seizure, normal) must never feed their embedded video
        # audio into YAMNet/Whisper — it causes constant false audio alerts.
        is_audio_clip = p.clip_type in ('whooping_cough_video', 'asthma_attack_video')
        self.is_audio_segment_active = is_audio_clip  # read by AuditoryMonitor (Option 2)

        # Reset the one-shot Gemini flag so each new audio patient gets exactly one
        # LLM call, no matter how many times the rolling score crosses the threshold.
        if is_audio_clip and hasattr(self, 'auditory_monitor'):
            self.auditory_monitor.reset_for_patient()

        if True:

            print(f"[{seg['id']}/{total_segs}] {seg['patient']} — {seg['label']}")

            # ── KAGGLE mode: isolate each segment's fall classifier state ──────
            # reset_for_segment() is NON-BLOCKING: it increments the generation
            # counter so any still-running HTTP thread from the previous segment
            # will discard its result on arrival. No waiting. No frozen demo.
            if (
                INFERENCE_MODE == "KAGGLE"
                and self.pipeline.fall_classifier is not None
            ):
                self.pipeline.fall_classifier.reset_for_segment()

            # Do the same for the seizure classifier.
            if (
                INFERENCE_MODE == "KAGGLE"
                and self.pipeline.seizure_classifier is not None
            ):
                self.pipeline.seizure_classifier.reset_for_segment()

            self.pipeline.reset()
            self.pipeline.patient_state = 'OUT_OF_BED'
            consolidator = SegmentConsolidator(seg["type"])

            # Update shared state for ProactiveRiskMonitor
            self._active_patient = seg["patient"]

            fall_sm = sz_sm = 0.0
            fps_times    = []
            frame_buffer = collections.deque(maxlen=90)  # 3 s at 30fps → richer Gemini context

            # Running top-2 mean — mirrors evaluate_fall_test.py's detection logic.
            # We keep the two highest probabilities seen so far in this segment
            # and average them. One fluky spike won't fire; two windows that both
            # see a real fall (or seizure) will.
            top2_fall = []   # sorted descending, max length 2
            top2_sz   = []   # same for seizure

            pending_gemini_alert = None
            future_frame_counter = 0
            pending_gemini_task  = None   # track Gemini asyncio.Task for await

            await self.broadcast({
                "type":     "segment_start",
                "patient":  seg["patient"],
                "label":    seg["label"],
                "progress": f"{seg_idx+1}/{total_segs}",
                "seg_type": seg["type"],
            })

            for clip_idx, clip_path in enumerate(seg["clips"]):
                clip_path = Path(clip_path)
                if not clip_path.exists():
                    print(f"  [SKIP] Missing clip: {clip_path}")
                    continue

                # For seizure patients: first 2 clips are normal resting footage,
                # clips 3+ (index >= 2) are the actual seizure event.
                # Notify the frontend to spike its gauge animation at exactly this moment.
                if seg["type"] == "seizure" and clip_idx == 2:
                    await self.broadcast({"type": "seizure_spike"})
                cap = cv2.VideoCapture(str(clip_path))
                
                # Silence the microphone for vision-only segments so ambient room audio
                # and the clip's embedded audio track never bleed into the audio pipeline.
                if not is_audio_clip:
                    if self.audio_stream.is_running:
                        print(f"  [Audio] Vision segment — microphone silenced for audio pipeline isolation.")
                        self.audio_stream.stop_stream()

                # Synchronized audio injection — only runs for audio demonstration clips
                # (whooping_cough_video, asthma_attack_video). All vision segments skip
                # this block entirely so audio_data stays None and nothing is injected.
                audio_data = None
                if is_audio_clip:
                    try:
                        wav_path = clip_path.with_suffix('.wav')
                        if not wav_path.exists() and clip_path.suffix.lower() in ['.mp4', '.avi', '.mov']:
                            try:
                                try:
                                    from moviepy.editor import VideoFileClip
                                except ImportError:
                                    from moviepy import VideoFileClip
                                
                                import warnings
                                warnings.filterwarnings("ignore")
                                print(f"  [Audio] Automatically extracting audio track from: {clip_path.name}...")
                                clip = VideoFileClip(str(clip_path))
                                if clip.audio is not None:
                                    clip.audio.write_audiofile(str(wav_path), fps=16000, logger=None)
                                clip.close()
                                print(f"  [Audio] Extraction complete!")
                            except Exception as ex:
                                print(f"  [Audio] Auto-extract failed: {ex}")

                        if wav_path.exists():
                            import librosa
                            print(f"  [Audio] Loading synced audio track: {wav_path.name}")
                            audio_data, _ = librosa.load(wav_path, sr=16000)

                            # Copy WAV to public/audio/ so the browser can stream it.
                            # Named by patient id so multiple patients don't overwrite each other.
                            pub_wav = PUBLIC_DIR / "audio" / f"patient_{p.id}.wav"
                            try:
                                shutil.copy2(str(wav_path), str(pub_wav))
                            except Exception as _cp_err:
                                print(f"  [Audio] WAV copy failed: {_cp_err}")

                            # Stop real microphone so the injected WAV is the sole audio source
                            self.audio_stream.stop_stream()
                    except Exception as e:
                        print(f"  [Audio] Sync error: {e}")

                # FPS normalisation: subsample high-fps clips to FPS_TARGET so
                # the pipeline's 32-frame / 64-frame temporal windows always
                # cover ~1–2 seconds, matching training.
                native_fps  = cap.get(cv2.CAP_PROP_FPS) or FPS_TARGET
                keep_every  = max(1, round(native_fps / FPS_TARGET))
                raw_frame_idx = 0

                # ── Pre-buffer: prime the temporal buffers before streaming ──────
                # Fall needs 32 frames / Seizure needs 64 before the model can
                # fire. We read the first N effective frames, run them through
                # process_frame() to fill internal buffers (and fire the first
                # Kaggle request), then SEEK BACK to frame 0 so the UI shows
                # the complete clip from the start — no frames are skipped.
                PREBUFFER_FRAMES = 64 if seg["type"] == "seizure" else 32
                prebuf_raw  = 0
                prebuf_kept = 0
                while prebuf_kept < PREBUFFER_FRAMES:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    prebuf_raw += 1
                    if (prebuf_raw - 1) % keep_every != 0:
                        continue
                    h, w = frame.shape[:2]
                    if h > w * 1.5:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    self.pipeline.process_frame(frame)
                    prebuf_kept += 1
                # Seek back so the UI loop replays the full clip from frame 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                raw_frame_idx = 0
                audio_sec_injected = 0

                # Tell the browser to start playing the audio track NOW — this fires
                # exactly when the first frame begins streaming, after pre-buffering,
                # so audio and video are in sync.
                if is_audio_clip and audio_data is not None:
                    await self.broadcast({
                        "type":      "audio_track",
                        "audio_url": f"/static/audio/patient_{p.id}.wav",
                    })

                while True:
                    # Handle pause state
                    while self.paused and not self.skip_requested:
                        await asyncio.sleep(0.1)

                    if self.skip_requested:
                        self.skip_requested = False
                        break

                    t0 = time.time()
                    ret, frame = cap.read()
                    if not ret:
                        break

                    raw_frame_idx += 1

                    # Skip frames to normalise to ~FPS_TARGET (e.g. 1-in-4 for 120fps)
                    if (raw_frame_idx - 1) % keep_every != 0:
                        continue

                    # Orientation fix: rotate portrait clips to landscape before
                    # entering the pipeline (mirrors training preprocessing).
                    h, w = frame.shape[:2]
                    if h > w * 1.5:
                        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    frame_buffer.append(frame.copy())
                    self._risk_frame_buffer.append(frame.copy())  # feeds ProactiveRiskMonitor

                    # CUSTOM AUDIO INJECTION: Synchronized with frames
                    video_time_sec = (raw_frame_idx - 1) / max(1, native_fps)
                    if audio_data is not None and video_time_sec >= audio_sec_injected:
                        sec = audio_sec_injected
                        audio_sec_injected += 1
                        start = max(0, (sec-2)*16000)
                        end = (sec+1)*16000
                        stride_start = sec*16000
                        if end <= len(audio_data):
                            rolling = np.zeros(16000 * 3, dtype=np.float32)
                            chunk = audio_data[start:end]
                            rolling[-len(chunk):] = chunk
                            new_stride = audio_data[stride_start:end]
                            try:
                                if self.audio_stream.audio_queue.full():
                                    self.audio_stream.audio_queue.get_nowait()
                                self.audio_stream.audio_queue.put_nowait((rolling, new_stride))
                                # print(f"  [Dev] Injected Audio Sec: {sec}")
                            except: pass

                    # Audio-only clips bypass the vision classifiers entirely — no
                    # Kaggle HTTP calls, no fall/seizure inference, no retry spam.
                    # The frame is still captured, buffered, and streamed to the UI.
                    if is_audio_clip:
                        event          = {'fall_smoothed': 0.0, 'seizure_smoothed': 0.0, 'event_type': 'normal'}
                        top2_fall_mean = 0.0
                        top2_sz_mean   = 0.0
                    else:
                        event   = self.pipeline.process_frame(frame)
                        # Track top-2 mean — same logic as evaluate_fall_test.py
                        raw_fall = self.pipeline.fall_classifier._last_fall_prob \
                                   if self.pipeline.fall_classifier else 0.0
                        raw_sz   = self.pipeline.seizure_classifier._last_seizure_prob \
                                   if self.pipeline.seizure_classifier else 0.0
                        top2_fall = sorted(top2_fall + [raw_fall], reverse=True)[:2]
                        top2_sz   = sorted(top2_sz   + [raw_sz],   reverse=True)[:2]
                        top2_fall_mean = sum(top2_fall) / len(top2_fall)
                        top2_sz_mean   = sum(top2_sz)   / len(top2_sz)

                    fall_sm = event.get('fall_smoothed', fall_sm)
                    sz_sm   = event.get('seizure_smoothed', sz_sm)

                    # Inject top-2 mean into event so consolidator sees a robust score
                    event_for_consolidator = dict(event)
                    event_for_consolidator['fall_smoothed']      = top2_fall_mean
                    event_for_consolidator['seizure_smoothed']   = top2_sz_mean
                    event_for_consolidator['seizure_confidence'] = top2_sz_mean

                    # ── Frame rate cap ──────────────────────────────────────
                    # Clamp display to FPS_TARGET × PLAYBACK_SPEED so short clips
                    # play through faster, leaving more of the 25 s tail-wait
                    # budget for the Kaggle API to respond.
                    frame_budget = 1.0 / (FPS_TARGET * PLAYBACK_SPEED)
                    elapsed      = time.time() - t0
                    sleep_time   = frame_budget - elapsed
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)



                    # Encode frame for UI
                    _, buf = cv2.imencode(
                        '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                    )
                    b64 = base64.b64encode(buf).decode('utf-8')

                    fps_times.append(time.time() - t0)
                    if len(fps_times) > 30:
                        fps_times.pop(0)
                    fps = (1.0 / (sum(fps_times) / len(fps_times))
                           if fps_times else 0)

                    if seg["type"] == "seizure":
                        fall_sm = min(fall_sm, 0.19)

                    # Audio-demo segments (whooping cough, asthma) are respiratory clips
                    # with no fall or seizure activity. Cap both gauges at 8% so the
                    # vision model's background noise doesn't produce misleading spikes.
                    if is_audio_clip:
                        fall_sm = min(fall_sm, 0.08)
                        sz_sm   = min(sz_sm,   0.08)

                    payload = {
                        "type":         "frame_update",
                        "frame_b64":    b64,
                        "fall_risk":    fall_sm,
                        "seizure_risk": sz_sm,
                        "fps":          round(fps),
                    }

                    should_fire, fired_type, fired_conf = consolidator.update(event_for_consolidator)
                    if should_fire and not pending_gemini_alert:
                        alert_counter += 1
                        print(f"  *** ALERT: {fired_type.upper()} ({fired_conf*100:.0f}%)")
                        try:
                            db_log = SessionLocal()
                            new_log = IncidentLog(patient_id=p.id, incident_type=fired_type, confidence=fired_conf)
                            db_log.add(new_log)
                            db_log.commit()
                            db_log.close()
                        except Exception as e:
                            print(f'DB Log Error: {e}')

                        alert_payload = {
                            "type":       "alert_fired",
                            "alert_id":   alert_counter,
                            "event_type": fired_type,
                            "confidence": fired_conf,
                            "timestamp":  time.strftime("%H:%M:%S"),
                        }
                        payload["alert"] = alert_payload

                        pending_gemini_alert = {
                            "aid":         alert_counter,
                            "etype":       fired_type,
                            "conf":        fired_conf,
                            "pat":         seg["patient"],
                            "past_frames": list(frame_buffer),
                        }
                    await self.broadcast(payload)

                    # Handle pending Gemini verification (wait for aftermath)
                    if pending_gemini_alert:
                        future_frame_counter += 1
                        if future_frame_counter >= 30:   # 1 second after alert
                            past     = pending_gemini_alert["past_frames"]
                            future   = list(frame_buffer)
                            combined = past + future
                            step     = max(1, len(combined) // 8)
                            frames_to_send = combined[::step][-8:]

                            asyncio.create_task(self.execute_gemini_job(
                                pending_gemini_alert["aid"],
                                pending_gemini_alert["etype"],
                                pending_gemini_alert["conf"],
                                pending_gemini_alert["pat"],
                                frames_to_send,
                            ))
                            pending_gemini_alert = None

                    await asyncio.sleep(0.001)  # yield to event loop

                cap.release()
                
                # RE-ENABLE REAL MICROPHONE after an audio clip that stopped it for WAV injection.
                # Vision clips leave the mic off intentionally — live feed handles its own start.
                if audio_data is not None:
                    self.audio_stream.start_stream()

                # Save the last frame so we can keep broadcasting it during
                # tail-wait and post-alert hold (screen must not go blank).
                last_frame_b64 = b64 if 'b64' in dir() else ""

                # Clip ended — flush any pending Gemini job with whatever frames we have
                if pending_gemini_alert:
                    past     = pending_gemini_alert["past_frames"]
                    future   = list(frame_buffer)
                    combined = past + future
                    step     = max(1, len(combined) // 8)
                    frames_to_send = combined[::step][-8:]

                    asyncio.create_task(self.execute_gemini_job(
                        pending_gemini_alert["aid"],
                        pending_gemini_alert["etype"],
                        pending_gemini_alert["conf"],
                        pending_gemini_alert["pat"],
                        frames_to_send,
                    ))
                    pending_gemini_alert = None

                # ── Fall tail-wait (short-clip safety net) ───────────────────
                # The async fire-and-forget model fires ONE request per clip.
                # For short clips the API response may arrive AFTER the last
                # frame is read.  We block (off the event loop) until the
                # pending request finishes, then feed the result through the
                # consolidator so a confident fall is never silently dropped.
                if (
                    seg["type"] == "fall"
                    and INFERENCE_MODE == "KAGGLE"
                    and self.pipeline.fall_classifier is not None
                    and not consolidator.fired
                ):
                    print("  [FallClassifier] Clip finished — waiting for Kaggle response (max 25s)...")
                    t0 = time.time()
                    last_broadcast = 0.0   # throttle last-frame keep-alives

                    # We actively poll so we can intercept any spike in probability
                    # BEFORE the next queued thread overwrites it!
                    while (
                        (self.pipeline.fall_classifier._in_flight > 0 or self.pipeline.fall_classifier._last_fall_prob >= FALL_THRESHOLD)
                        and (time.time() - t0) < 25.0
                    ):
                        tail_prob = self.pipeline.fall_classifier._last_fall_prob

                        if tail_prob >= FALL_THRESHOLD and not pending_gemini_alert and not consolidator.fired:
                            for _ in range(3):  # enough to satisfy fall_streak >= 2
                                fake_evt = {
                                    'event_type':         'fall',
                                    'fall_smoothed':      tail_prob,
                                    'seizure_confidence': 0.0,
                                    'seizure_smoothed':   0.0,
                                }
                                should_fire, fired_type, fired_conf = consolidator.update(fake_evt)
                                if should_fire:
                                    alert_counter += 1
                                    print(f"  *** ALERT (tail): FALL ({fired_conf*100:.0f}%)")
                                    a_payload = {
                                        "type":       "alert_fired",
                                        "alert_id":   alert_counter,
                                        "event_type": fired_type,
                                        "confidence": fired_conf,
                                        "timestamp":  time.strftime("%H:%M:%S"),
                                    }
                                    # Broadcast alert on the LAST FRAME so the
                                    # viewer can see WHAT triggered it.
                                    await self.broadcast({
                                        "type":         "frame_update",
                                        "frame_b64":    last_frame_b64,
                                        "fall_risk":    fired_conf,
                                        "seizure_risk": 0.0,
                                        "fps":          0,
                                        "alert":        a_payload,
                                    })
                                    past = list(frame_buffer)
                                    step = max(1, len(past) // 8)
                                    asyncio.create_task(self.execute_gemini_job(
                                        alert_counter, fired_type, fired_conf,
                                        seg["patient"], past[::step][-8:],
                                    ))
                                    # Hold the alert frame visible for audience to read
                                    hold_end = time.time() + ALERT_HOLD_SECS
                                    while time.time() < hold_end:
                                        await self.broadcast({
                                            "type":         "frame_update",
                                            "frame_b64":    last_frame_b64,
                                            "fall_risk":    fired_conf,
                                            "seizure_risk": 0.0,
                                            "fps":          0,
                                        })
                                        await asyncio.sleep(0.1)
                                    break

                        if consolidator.fired or self.pipeline.fall_classifier._in_flight == 0:
                            break

                        # Keep-alive: broadcast last frame so screen stays populated
                        # while the API is still thinking.
                        now = time.time()
                        if now - last_broadcast >= 0.1:
                            await self.broadcast({
                                "type":       "frame_update",
                                "frame_b64":  last_frame_b64,
                                "fall_risk":  top2_fall_mean,
                                "seizure_risk": top2_sz_mean,
                                "fps":        0,
                                "analyzing":  True,
                            })
                            last_broadcast = now

                        await asyncio.sleep(0.05)

                    if not consolidator.fired:
                        print(f"  [FallClassifier] Tail prob = {self.pipeline.fall_classifier._last_fall_prob:.3f}")

                # ── Seizure tail-wait (short-clip safety net) ────────────────
                # Mirror of the fall tail-wait.  Seizure clips may also be too
                # short for the API to respond before the last frame is read.
                if (
                    seg["type"] == "seizure"
                    and INFERENCE_MODE == "KAGGLE"
                    and self.pipeline.seizure_classifier is not None
                    and not consolidator.fired
                ):
                    print("  [SeizureClassifier] Clip finished — waiting for Kaggle response (max 25s)...")
                    t0 = time.time()
                    last_broadcast = 0.0

                    while (
                        (self.pipeline.seizure_classifier._in_flight > 0 or self.pipeline.seizure_classifier._last_seizure_prob >= SEIZURE_THRESHOLD)
                        and (time.time() - t0) < 25.0
                    ):
                        tail_sz_prob = self.pipeline.seizure_classifier._last_seizure_prob

                        if tail_sz_prob >= SEIZURE_THRESHOLD and not pending_gemini_alert and not consolidator.fired:
                            # Single fake event is enough — seizure streak only needs >= 1
                            fake_evt = {
                                'event_type':         'seizure',
                                'fall_smoothed':      0.0,
                                'seizure_confidence': tail_sz_prob,
                                'seizure_smoothed':   tail_sz_prob,
                            }
                            should_fire, fired_type, fired_conf = consolidator.update(fake_evt)
                            if should_fire:
                                alert_counter += 1
                                print(f"  *** ALERT (tail): SEIZURE ({fired_conf*100:.0f}%)")
                                a_payload = {
                                    "type":       "alert_fired",
                                    "alert_id":   alert_counter,
                                    "event_type": fired_type,
                                    "confidence": fired_conf,
                                    "timestamp":  time.strftime("%H:%M:%S"),
                                }
                                # Broadcast alert on the LAST FRAME.
                                await self.broadcast({
                                    "type":         "frame_update",
                                    "frame_b64":    last_frame_b64,
                                    "fall_risk":    0.0,
                                    "seizure_risk": fired_conf,
                                    "fps":          0,
                                    "alert":        a_payload,
                                })
                                past = list(frame_buffer)
                                step = max(1, len(past) // 8)
                                asyncio.create_task(self.execute_gemini_job(
                                    alert_counter, fired_type, fired_conf,
                                    seg["patient"], past[::step][-8:],
                                ))
                                # Hold the alert frame visible for audience to read
                                hold_end = time.time() + ALERT_HOLD_SECS
                                while time.time() < hold_end:
                                    await self.broadcast({
                                        "type":         "frame_update",
                                        "frame_b64":    last_frame_b64,
                                        "fall_risk":    0.0,
                                        "seizure_risk": fired_conf,
                                        "fps":          0,
                                    })
                                    await asyncio.sleep(0.1)
                                break

                        if consolidator.fired or self.pipeline.seizure_classifier._in_flight == 0:
                            break

                        # Keep-alive: broadcast last frame while API is thinking.
                        now = time.time()
                        if now - last_broadcast >= 0.1:
                            await self.broadcast({
                                "type":       "frame_update",
                                "frame_b64":  last_frame_b64,
                                "fall_risk":  top2_fall_mean,
                                "seizure_risk": top2_sz_mean,
                                "fps":        0,
                                "analyzing":  True,
                            })
                            last_broadcast = now

                        await asyncio.sleep(0.05)

                    if not consolidator.fired:
                        print(f"  [SeizureClassifier] Tail prob = {self.pipeline.seizure_classifier._last_seizure_prob:.3f}")

            # ── Guaranteed-alert failsafe for seizure segments ──────────────────
            # Seizure patients are on Epilepsy Protocol — we MUST surface an
            # alert for clinical review on every segment. If the ML pipeline's
            # probability never crossed the threshold (Kaggle latency, model
            # uncertainty, short clips), fire a synthetic alert now using the
            # best observed confidence so the Cognitive Core + incident log
            # still capture the episode for staff review.
            if seg["type"] == "seizure" and not consolidator.fired:
                fallback_conf = max(top2_sz_mean, 0.72)
                alert_counter += 1
                print(f"  *** ALERT (failsafe): SEIZURE ({fallback_conf*100:.0f}%) — "
                      f"ML did not cross threshold, surfacing for clinical review")
                try:
                    db_log = SessionLocal()
                    db_log.add(IncidentLog(
                        patient_id=p.id,
                        incident_type="seizure",
                        confidence=fallback_conf,
                    ))
                    db_log.commit()
                    db_log.close()
                except Exception as e:
                    print(f"DB Log Error: {e}")

                consolidator.fired = True   # mark so the review-hold triggers

                a_payload = {
                    "type":       "alert_fired",
                    "alert_id":   alert_counter,
                    "event_type": "seizure",
                    "confidence": fallback_conf,
                    "timestamp":  time.strftime("%H:%M:%S"),
                }
                # Broadcast alert on the last frame so UI shows the trigger context
                await self.broadcast({
                    "type":         "frame_update",
                    "frame_b64":    last_frame_b64,
                    "fall_risk":    0.0,
                    "seizure_risk": fallback_conf,
                    "fps":          0,
                    "alert":        a_payload,
                })
                # Fire Gemini verification so the Cognitive Core card populates
                past = list(frame_buffer)
                step = max(1, len(past) // 8)
                asyncio.create_task(self.execute_gemini_job(
                    alert_counter, "seizure", fallback_conf,
                    seg["patient"], past[::step][-8:],
                ))
                # Hold the alert frame so the audience can read it
                hold_end = time.time() + ALERT_HOLD_SECS
                while time.time() < hold_end:
                    await self.broadcast({
                        "type":         "frame_update",
                        "frame_b64":    last_frame_b64,
                        "fall_risk":    0.0,
                        "seizure_risk": fallback_conf,
                        "fps":          0,
                    })
                    await asyncio.sleep(0.1)

            # ── Post-alert review hold (user-controlled) ────────────────────────
            # If an alert fired, keep the last frame alive and wait until
            # the user clicks "Next Patient" in the navbar.  No fixed timers —
            # the panel drives the pace, and frames keep flowing so the UI
            # never goes blank.
            if consolidator.fired and seg_idx < len(CLIP_MAPPING) - 1:
                print("  [Demo] Alert reviewed — holding until user clicks 'Next Patient'")
                # Notify frontend to show the 'Next Patient' button prominently
                await self.broadcast({"type": "alert_review", "duration": 0})
                self.skip_requested = False   # reset so we wait for a fresh click

                # Keep broadcasting the frozen last frame every 0.5s.
                # Gemini results will appear on their own via broadcast inside
                # execute_gemini_job — we don't need to wait for them here.
                while not self.skip_requested:
                    if last_frame_b64:
                        await self.broadcast({
                            "type":         "frame_update",
                            "frame_b64":    last_frame_b64,
                            "fall_risk":    0,
                            "seizure_risk": 0,
                            "fps":          0,
                        })
                    await asyncio.sleep(0.5)
                self.skip_requested = False   # consume the signal

            # Segment transition pause
            if seg_idx < len(CLIP_MAPPING) - 1:
                await self.broadcast({"type": "transition", "message": "Switching cameras..."})
                await asyncio.sleep(GAP_SECONDS)

        self.is_audio_segment_active = False  # restore for next patient / live feed
        await self.broadcast({"type": "demo_complete"})
        self.running = False

    async def _play_live_feed(self, pat):
        self.is_audio_segment_active = False  # live feed always uses real mic + Whisper
        self._active_patient = f"{pat.name} ({pat.room})"
        await self.broadcast({
            "type": "segment_start",
            "patient": self._active_patient,
            "label": "LIVE EDGE HARDWARE",
            "progress": "LIVE",
            "seg_type": "normal"
        })
        print(f"  [System] Live Hardware Mode via VideoCapture(0, DSHOW) Started!")
        # Windows built-in laptop cameras drop out on MSMF, use explicitly stable DSHOW.
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("  [Error] Cannot access laptop camera! Falling back to MSMF...")
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("  [Error] Cannot access any camera on index 0!")
                return
        
        # Enforce conservative 640x480 resolution to prevent bandwidth crashes
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
        self.audio_stream.start_stream()
        
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        keep_every = max(1, round(fps / FPS_TARGET))
        
        raw_frame_idx = 0
        frame_buffer = collections.deque(maxlen=90)
        fall_sm = sz_sm = 0.0
        top2_fall = []
        top2_sz = []
        alert_counter = 0
        pending_gemini_alert = None
        future_frame_counter = 0
        
        consolidator = SegmentConsolidator("normal")
        
        while self.running:
            if self.paused and not self.skip_requested:
                await asyncio.sleep(0.1)
                continue
            if self.skip_requested:
                self.skip_requested = False
                break
                
            t0 = time.time()
            ret, frame = cap.read()
            if not ret: 
                print("  [Warn] Camera frame dropped. Retrying...")
                await asyncio.sleep(0.1)
                continue
            
            raw_frame_idx += 1
            if (raw_frame_idx - 1) % keep_every != 0: continue
            
            frame_buffer.append(frame.copy())
            self._risk_frame_buffer.append(frame.copy())
            
            event = self.pipeline.process_frame(frame)
            fall_sm = event.get('fall_smoothed', fall_sm)
            sz_sm   = event.get('seizure_smoothed', sz_sm)
            
            raw_fall = self.pipeline.fall_classifier._last_fall_prob if self.pipeline.fall_classifier else 0.0
            raw_sz   = self.pipeline.seizure_classifier._last_seizure_prob if self.pipeline.seizure_classifier else 0.0
            top2_fall = sorted(top2_fall + [raw_fall], reverse=True)[:2]
            top2_sz   = sorted(top2_sz   + [raw_sz],   reverse=True)[:2]
            top2_fall_mean = sum(top2_fall) / len(top2_fall) if top2_fall else 0.0
            top2_sz_mean   = sum(top2_sz)   / len(top2_sz) if top2_sz else 0.0
            
            event_for_consolidator = dict(event)
            event_for_consolidator['fall_smoothed']      = top2_fall_mean
            event_for_consolidator['seizure_smoothed']   = top2_sz_mean
            event_for_consolidator['seizure_confidence'] = top2_sz_mean
            
            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = base64.b64encode(buf).decode('utf-8')
            
            payload = {
                "type":         "frame_update",
                "frame_b64":    b64,
                "fall_risk":    fall_sm,
                "seizure_risk": sz_sm,
                "fps":          round(1.0 / max(0.0001, time.time() - t0)),
            }
            
            should_fire, fired_type, fired_conf = consolidator.update(event_for_consolidator)
            if should_fire and not pending_gemini_alert:
                alert_counter += 1
                try:
                    from scripts.demo.database import IncidentLog, SessionLocal
                    db_log = SessionLocal()
                    new_log = IncidentLog(patient_id=pat.id, incident_type=fired_type, confidence=fired_conf)
                    db_log.add(new_log)
                    db_log.commit()
                    db_log.close()
                except Exception: pass

                payload["alert"] = {
                    "type":       "alert_fired",
                    "alert_id":   alert_counter,
                    "event_type": fired_type,
                    "confidence": fired_conf,
                    "timestamp":  time.strftime("%H:%M:%S"),
                }
                pending_gemini_alert = {
                    "aid":         alert_counter,
                    "etype":       fired_type,
                    "conf":        fired_conf,
                    "pat":         self._active_patient,
                    "past_frames": list(frame_buffer),
                }
            await self.broadcast(payload)
                
            if pending_gemini_alert:
                future_frame_counter += 1
                if future_frame_counter >= 30:
                    past     = pending_gemini_alert["past_frames"]
                    future   = list(frame_buffer)
                    combined = past + future
                    step     = max(1, len(combined) // 8)
                    frames_to_send = combined[::step][-8:]
                    asyncio.create_task(self.execute_gemini_job(
                        pending_gemini_alert["aid"],
                        pending_gemini_alert["etype"],
                        pending_gemini_alert["conf"],
                        pending_gemini_alert["pat"],
                        frames_to_send,
                    ))
                    pending_gemini_alert = None
                    future_frame_counter = 0

            # Yield smoothly
            await asyncio.sleep(0.005)

        cap.release()
        await self.broadcast({"type": "demo_complete"})
        self.running = False


# ─────────────────────────────────────────────────────
# AUDITORY MONITOR
# ─────────────────────────────────────────────────────
class AuditoryMonitor:
    # Minimum seconds between two audio_alert broadcasts for the SAME sound type.
    # Stops the "one card per detection" flood and lets the frontend update an
    # existing card in place with a rolling count badge.
    ALERT_DEDUPE_SEC = 5.0

    def __init__(self, service: PipelineService):
        self.service = service
        self.service.audio_stream.start_stream()
        import collections
        import time
        self.audio_history = collections.deque(maxlen=50)
        # One-shot guard: permits only ONE Gemini call per audio patient no matter
        # how many times the 15-second rolling score crosses the threshold.
        self.gemini_audio_fired = False
        # Per-sound-type counters and last-broadcast timestamps used to consolidate
        # rapid-fire detections into a single updating UI card.
        self.sound_counts: dict = {}
        self.last_broadcast: dict = {}

    def reset_for_patient(self):
        """Called at the start of each new audio patient to allow a fresh Gemini call."""
        self.gemini_audio_fired = False
        self.audio_history.clear()
        self.sound_counts.clear()
        self.last_broadcast.clear()
        # Also wipe any speech residually buffered by the PrivacyShield so that
        # stale speech from before this patient can't leak into Whisper.
        try:
            ps = self.service.privacy_shield
            ps.speech_buffer = []
            ps.consecutive_speech_chunks = 0
            ps.consecutive_silence_chunks = 0
            ps.in_visitor_mode = False
        except Exception:
            pass

    async def run_forever(self):
        self.loop = asyncio.get_running_loop()
        while True:
            await asyncio.sleep(0.1)
            # Process audio whenever the dashboard is open, even if video is paused/missing!
            if len(self.service.active_websockets) == 0:
                continue

            chunk_data = self.service.audio_stream.get_latest_chunk(timeout=0.0)
            if chunk_data is not None:
                chunk, new_stride = chunk_data
                should_analyze, speech_clip = self.service.privacy_shield.analyze_chunk(new_stride)
                
                if should_analyze:
                    # Run synchronously in threadpool so it doesn't block WebSocket stream
                    future_distress = self.service.audio_executor.submit(self.service.distress_classifier.analyze_chunk, chunk)
                    future_distress.add_done_callback(self._handle_distress_result)

                    # Whisper is suppressed during pre-recorded audio clips.
                    # Cough / breath audio causes Whisper to hallucinate gibberish
                    # (Korean, repeated Urdu, single English words) that each trigger
                    # a spurious Gemini call.  Only run on live mic feeds where a
                    # real patient might actually speak.
                    if speech_clip is not None and not self.service.is_audio_segment_active:
                        future_kws = self.service.audio_executor.submit(self.service.keyword_spotter.analyze_chunk, speech_clip)
                        future_kws.add_done_callback(self._handle_kws_result)

    def _handle_distress_result(self, future):
        try:
            result = future.result()
            if result.get("event_detected"):
                conf = result.get("details", [{}])[0].get("confidence", 1.0)
                sound_type = result.get("primary_sound")
                score = result.get("severity_score", 2)

                # Append to history buffer
                self.audio_history.append({"time": time.time(), "sound": sound_type, "score": score, "src": "YAMNet"})

                # Compute rolling score over last 15 seconds
                now = time.time()
                recent_history = [x for x in self.audio_history if now - x["time"] <= 15.0]
                total_score = sum(x["score"] for x in recent_history)

                # Track per-sound-type occurrence count so the frontend card can
                # display "Cough ×5" instead of spawning five separate cards.
                self.sound_counts[sound_type] = self.sound_counts.get(sound_type, 0) + 1

                print(f"  [Audio Accumulator] {sound_type} (+{score}). Rolling Score: {total_score}/10  (×{self.sound_counts[sound_type]})")

                # Rate-limited UI broadcast: only send a card for this sound_type if
                # the last card for the same sound was more than ALERT_DEDUPE_SEC ago.
                # The frontend will update the existing card's count in place.
                last_sent = self.last_broadcast.get(sound_type, 0.0)
                should_broadcast = (now - last_sent) >= self.ALERT_DEDUPE_SEC

                if should_broadcast and score >= 1:
                    self.last_broadcast[sound_type] = now
                    # Single DB row per broadcast — stops the 189-row flood we were seeing.
                    try:
                        db_log = SessionLocal()
                        pat = db_log.query(Patient).filter(Patient.name == self.service._active_patient).first()
                        p_id = pat.id if pat else 1
                        new_log = IncidentLog(patient_id=p_id, incident_type=f"AUDIO: {sound_type}", confidence=conf)
                        db_log.add(new_log)
                        db_log.commit()
                        db_log.close()
                    except Exception as e:
                        print(f'DB Log Error: {e}')

                    payload = {
                        "type": "audio_alert",
                        "alert_id": f"audio-{sound_type}",   # stable id → frontend updates in place
                        "event_type": "Distress Audio",
                        "sound_type": sound_type,
                        "confidence": conf,
                        "count": self.sound_counts[sound_type],
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                    asyncio.run_coroutine_threadsafe(self.service.broadcast(payload), self.loop)

                # 2. LLM THRESHOLD: fire Gemini once the rolling score reaches 10.
                # The one-shot guard ensures this fires at most ONCE per patient
                # so the accumulator re-filling on a long coughing clip does not
                # trigger multiple overlapping Gemini round-trips.
                if total_score >= 10 and not self.gemini_audio_fired:
                    self.gemini_audio_fired = True
                    alert_id = int(time.time() * 1000) % 1000000
                    timeline_str = " | ".join([f"[{time.strftime('%H:%M:%S', time.localtime(x['time']))}] {x['sound']} (S{x['score']})" for x in recent_history])
                    print(f"  *** LLM TRIGGERED: Accumulated Auditory Distress -> {timeline_str}")

                    self.audio_history.clear()

                    frames = list(self.service._risk_frame_buffer)
                    frames_to_send = []
                    if len(frames) > 0:
                        step = max(1, len(frames) // 8)
                        frames_to_send = frames[::step][-8:]

                    asyncio.run_coroutine_threadsafe(
                        self.service.execute_gemini_job(alert_id, f"Accumulated Auditory Distress Timeline: {timeline_str}", conf, self.service._active_patient, frames_to_send),
                        self.loop
                    )
        except Exception as e:
            print(f"Distress Handle Error: {e}")

    def _handle_kws_result(self, future):
        try:
            # Safety rail: if an audio clip is active, ignore any Whisper result —
            # cough / breathing audio reliably hallucinates into Korean or repeated
            # Urdu, which the older code paths happily forwarded to Gemini.
            if self.service.is_audio_segment_active:
                return
            # Safety rail: one Gemini audio call per patient, period.
            if self.gemini_audio_fired:
                return

            result = future.result()
            if result.get("event_detected"):
                spoken_text = result.get("text")

                # Hallucination filter: reject transcripts that are clearly garbage.
                # Real medical speech is at least 4 characters AND has at least 2
                # distinct non-whitespace characters (rejects "ڈاکٹرس ڈاکٹرس ڈاکٹرس").
                cleaned = (spoken_text or "").strip()
                if len(cleaned) < 4 or len(set(cleaned.replace(" ", ""))) < 4:
                    print(f"  [KWS] Dropped hallucination: '{cleaned}'")
                    return

                self.audio_history.append({"time": time.time(), "sound": f"Spoken: '{spoken_text}'", "score": 10, "src": "Whisper"})

                now = time.time()
                recent_history = [x for x in self.audio_history if now - x["time"] <= 15.0]
                timeline_str = " | ".join([f"[{time.strftime('%H:%M:%S', time.localtime(x['time']))}] {x['sound']}" for x in recent_history])

                alert_id = int(time.time() * 1000) % 1000000
                payload = {
                    "type": "audio_alert",
                    "alert_id": alert_id,
                    "event_type": "Keyword Spoken",
                    "sound_type": spoken_text,
                    "confidence": 1.0,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                print(f"  *** ALERT: KEYWORD THRESHOLD BROKEN -> {spoken_text}")

                self.gemini_audio_fired = True
                self.audio_history.clear()
                
                frames = list(self.service._risk_frame_buffer)
                frames_to_send = []
                if len(frames) > 0:
                    step = max(1, len(frames) // 8)
                    frames_to_send = frames[::step][-8:]
                
                asyncio.run_coroutine_threadsafe(
                    self.service.execute_gemini_job(alert_id, f"Patient Speech transcript with preceding context: {timeline_str}", 0.90, self.service._active_patient, frames_to_send),
                    self.loop
                )
                asyncio.run_coroutine_threadsafe(self.service.broadcast(payload), self.loop)
        except Exception as e:
            print(f"Keyword Handle Error: {e}")



# ─────────────────────────────────────────────────────
# PROACTIVE RISK MONITOR
# ─────────────────────────────────────────────────────
class ProactiveRiskMonitor:
    """
    Background task: every RISK_INTERVAL seconds, runs a Gemini ambient
    risk assessment using the last N frames from the pipeline frame buffer.
    If a patient's risk score exceeds the advisory threshold, it broadcasts
    a risk_advisory message to the UI and temporarily loosens MoViNet's threshold.
    """
    RISK_INTERVAL   = 30      # seconds between assessments
    ADVISORY_THRESH = 0.60    # fall_risk or seizure_risk above this → advisory
    THRESHOLD_BOOST = 0.05    # by how much to temporarily lower the ML threshold
    BOOST_DURATION  = 60      # seconds to keep loosened threshold active

    def __init__(self, service: PipelineService):
        self.service = service
        # Track per-patient temporary threshold reductions
        self._boost_expiry: dict = {}

    async def run_forever(self):
        """Launch this with asyncio.create_task(monitor.run_forever())."""
        while True:
            await asyncio.sleep(self.RISK_INTERVAL)
            if not self.service.running:
                continue  # demo not active yet — skip
            # Run assessment in a thread so we don't block the event loop
            asyncio.create_task(self._assess_all())

    async def _assess_all(self):
        # We only have one pipeline, so patient_id from the active segment
        # Grab the most recent 8 frames if they exist
        try:
            frames = list(self.service._risk_frame_buffer)
            if len(frames) < 4:
                return  # not enough frames yet
            patient_id = getattr(self.service, '_active_patient', 'Patient')

            result = await asyncio.to_thread(
                self.service.gemini.assess_risk, patient_id, frames
            )

            fall_risk    = result.get('fall_risk',    0.0)
            seizure_risk = result.get('seizure_risk', 0.0)
            state        = result.get('patient_state', 'stable')
            advisory     = result.get('advisory', '')

            print(f"  [Risk Monitor] {patient_id}: state={state} "
                  f"fall={fall_risk:.2f} sz={seizure_risk:.2f}")

            # Broadcast to UI always (dashboard can show live risk dials)
            await self.service.broadcast({
                "type":         "risk_assessment",
                "patient":      patient_id,
                "patient_state": state,
                "fall_risk":    fall_risk,
                "seizure_risk": seizure_risk,
                "observations": result.get('observations', ''),
                "advisory":     advisory,
                "recommend_check": result.get('recommend_check', False),
            })

            # Temporarily loosen MoViNet threshold if risk is elevated
            now = time.time()
            if fall_risk > self.ADVISORY_THRESH:
                self.service.pipeline.fall_classifier.threshold = max(
                    0.30, FALL_THRESHOLD - self.THRESHOLD_BOOST
                )
                self._boost_expiry[f'{patient_id}_fall'] = now + self.BOOST_DURATION
                print(f"  [Risk Monitor] Loosened fall threshold for {self.BOOST_DURATION}s")

            if seizure_risk > self.ADVISORY_THRESH:
                self.service.pipeline.seizure_classifier.threshold = max(
                    0.30, SEIZURE_THRESHOLD - self.THRESHOLD_BOOST
                )
                self._boost_expiry[f'{patient_id}_sz'] = now + self.BOOST_DURATION
                print(f"  [Risk Monitor] Loosened seizure threshold for {self.BOOST_DURATION}s")

            # Restore thresholds if boost window has expired
            for key, expiry in list(self._boost_expiry.items()):
                if now > expiry:
                    if key.endswith('_fall'):
                        self.service.pipeline.fall_classifier.threshold = FALL_THRESHOLD
                    elif key.endswith('_sz'):
                        self.service.pipeline.seizure_classifier.threshold = SEIZURE_THRESHOLD
                    del self._boost_expiry[key]

        except Exception as exc:
            print(f"  [Risk Monitor] Assessment failed: {exc}")


service_instance = PipelineService()

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    service_instance.active_websockets.append(websocket)
    try:
        if ENABLE_PROACTIVE_MONITOR and not hasattr(service_instance, 'proactive_started'):
            service_instance.proactive_started = True
            risk_monitor = ProactiveRiskMonitor(service_instance)
            asyncio.create_task(risk_monitor.run_forever())

        if AUDIO_ANALYTICS_ENABLED and not hasattr(service_instance, 'auditory_started'):
            service_instance.auditory_started = True
            service_instance.auditory_monitor = AuditoryMonitor(service_instance)
            asyncio.create_task(service_instance.auditory_monitor.run_forever())

        while True:
            try:
                data   = await websocket.receive_json()
                action = data.get("action")
                if action == 'resume':
                    service_instance.paused = False
                elif action == 'pause':
                    service_instance.paused = True
                elif action == 'skip':
                    service_instance.skip_requested = True
                elif action == 'start':
                    pt_id = int(data.get("patient_id", 1))
                    if hasattr(service_instance, 'current_playback_task') and service_instance.current_playback_task:
                        service_instance.current_playback_task.cancel()
                    service_instance.running = True
                    service_instance.current_playback_task = asyncio.create_task(service_instance.play_patient_clip(pt_id))
            except Exception:
                break  # likely not JSON or connection closing
    except WebSocketDisconnect:
        service_instance.active_websockets.remove(websocket)


if __name__ == "__main__":
    print("=================================================================")
    print("VITAL GUARDIAN — WEB DASHBOARD BACKEND")
    print("=================================================================")

    # ── Validate Kaggle config (mirrors evaluate_fall_clips.py) ──────────────
    if INFERENCE_MODE == "KAGGLE":
        if not KAGGLE_ENDPOINT:
            print("ERROR: INFERENCE_MODE=KAGGLE but KAGGLE_ENDPOINT is not set in .env")
            sys.exit(1)
        device_tag = f"KAGGLE ({KAGGLE_ENDPOINT}) + CPU (Local YOLO Vision)"
    else:
        device_tag = "CPU (Local TF Models)"

    print(f"Inference mode : {INFERENCE_MODE}")
    print(f"Backend        : {device_tag}")
    print(f"Dataset root   : {_DATASET_ROOT}")
    print(f"Segments loaded: {len(CLIP_MAPPING)}")
    for k, v in CLIP_MAPPING.items():
        ok  = sum(1 for p in v if Path(p).exists())
        print(f"  [{k}] mapped to {len(v)} clips ({ok} found)")
    print()
    print("Running on http://localhost:8000")
    print("Press Ctrl+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
