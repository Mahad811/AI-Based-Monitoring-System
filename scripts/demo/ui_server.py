import os
import sys
import time
import asyncio
import cv2
import base64
from pathlib import Path
from pydantic import BaseModel
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np

from database import init_db, get_db, Nurse, Patient, IncidentLog, SessionLocal
from sqlalchemy.orm import Session

ROOT = Path(__file__).resolve().parent.parent.parent

# Video mapping logic
D = Path(os.getenv("VG_DEMO_VIDEO_ROOT", str(ROOT / "demo_dataset")))
def _find_clips(directory: Path, extensions=(".mp4", ".avi", ".mov")):
    if not directory.exists(): return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in extensions)

fall_clips = _find_clips(D / "falls") + _find_clips(D / "fall_test" / "fall")
normal_clips = _find_clips(D / "normal") + _find_clips(D / "fall_test" / "nofall")
sz_clips = _find_clips(D / "unusual_movement" / "data" / "Seizure")

CLIP_MAPPING = {
    "fall_1": fall_clips[0] if len(fall_clips) > 0 else None,
    "fall_2": fall_clips[1] if len(fall_clips) > 1 else None,
    "normal_1": normal_clips[0] if len(normal_clips) > 0 else None,
    "seizure_1": sz_clips[0] if len(sz_clips) > 0 else None,
    "seizure_2": sz_clips[1] if len(sz_clips) > 1 else None,
}

app = FastAPI(title="Vital Guardian UI Dev Server")

PUBLIC_DIR = Path(__file__).resolve().parent / "public"
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")

@app.get("/")
def serve_dashboard():
    return FileResponse(PUBLIC_DIR / "index.html")

class LoginRequest(BaseModel):
    username: str
    password: str

@app.on_event("startup")
def startup_event():
    init_db()

def verify_admin(req: Request):
    auth = req.headers.get("Authorization")
    if not auth or "Bearer " not in auth:
        raise HTTPException(status_code=401, detail="Missing Token")
    token = auth.split(" ")[1]
    try:
        decoded = base64.b64decode(token).decode()
        username = decoded.split(":")[0]
        if username != "admin":
            raise HTTPException(status_code=403, detail="Forbidden: Admins only")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Token")

@app.post("/api/login")
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(Nurse).filter(Nurse.staff_id == req.username.lower()).first()
    if not user or user.password != req.password:
        raise HTTPException(status_code=401, detail="Invalid credentials.")
    token = base64.b64encode(f"{req.username}:{time.time()}".encode()).decode()
    return {
        "status": "success",
        "token": token,
        "nurse_id": str(user.id),
        "nurse_name": user.name,
        "staff_id": user.staff_id
    }

class NurseCreate(BaseModel):
    staff_id: str
    name: str
    password: str

@app.get("/api/admin/nurses")
def get_nurses(db: Session = Depends(get_db), _: None = Depends(verify_admin)):
    nurses = db.query(Nurse).all()
    return [{"id": n.id, "staff_id": n.staff_id, "name": n.name} for n in nurses]

@app.post("/api/admin/nurses")
def create_nurse(nurse_data: NurseCreate, db: Session = Depends(get_db), _: None = Depends(verify_admin)):
    if db.query(Nurse).filter(Nurse.staff_id == nurse_data.staff_id.lower()).first():
        raise HTTPException(status_code=400, detail="Staff ID already exists.")
    new_nurse = Nurse(staff_id=nurse_data.staff_id.lower(), name=nurse_data.name, password=nurse_data.password)
    db.add(new_nurse)
    db.commit()
    return {"status": "success"}

@app.delete("/api/admin/nurses/{nurse_id}")
def delete_nurse(nurse_id: int, db: Session = Depends(get_db), _: None = Depends(verify_admin)):
    nurse = db.query(Nurse).filter(Nurse.id == nurse_id).first()
    if not nurse: raise HTTPException(status_code=404, detail="Nurse not found")
    if nurse.staff_id == 'admin': raise HTTPException(status_code=403, detail="Cannot delete super admin")
    db.delete(nurse)
    db.commit()
    return {"status": "success"}

# --- Hub Data API ---
@app.get("/api/patients")
def get_patients(db: Session = Depends(get_db)):
    pts = db.query(Patient).all()
    return [{"id": p.id, "name": p.name, "room": p.room, "age": p.age, "risk_profile": p.risk_profile} for p in pts]

@app.get("/api/history")
def get_history(db: Session = Depends(get_db)):
    logs = db.query(IncidentLog).order_by(IncidentLog.timestamp.desc()).all()
    res = []
    for lg in logs:
        p = db.query(Patient).filter(Patient.id == lg.patient_id).first()
        res.append({
            "id": lg.id,
            "patient_name": p.name if p else "Unknown",
            "room": p.room if p else "N/A",
            "incident_type": lg.incident_type.upper(),
            "confidence": lg.confidence,
            "timestamp": lg.timestamp.strftime("%Y-%m-%d %H:%M:%S") if lg.timestamp else "N/A"
        })
    return res

@app.get("/api/patient/{patient_id}")
def get_patient(patient_id: int, db: Session = Depends(get_db)):
    p = db.query(Patient).filter(Patient.id == patient_id).first()
    if not p: raise HTTPException(status_code=404, detail="Patient not found")
    return {"id": p.id, "name": p.name, "room": p.room, "age": p.age, "risk_profile": p.risk_profile}

# --- Event-Driven WebSocket ---
class UIPipelineService:
    def __init__(self):
        self.active_websockets = []
        self.current_playback_task = None
        self.paused = False

    async def broadcast(self, payload: dict):
        for ws in self.active_websockets:
            try: await ws.send_json(payload)
            except: pass

    async def play_patient(self, patient_id: int):
        db = SessionLocal()
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        db.close()
        
        if not p:
            await self.broadcast({"type": "transition", "message": "Invalid Patient ID"})
            return
            
        clip_path = CLIP_MAPPING.get(p.clip_type)
        if not clip_path or not clip_path.exists():
            await self.broadcast({"type": "transition", "message": "Video feed offline."})
            return

        print(f"UI Dev Playing: Patient {p.name} ({p.clip_type})")
        
        await self.broadcast({
            "type": "segment_start",
            "patient": p.name,
            "label": f"Live Feed — {p.room}",
            "progress": f"Bed {p.id}"
        })

        # Base event logic depending on filename mapping
        is_fall = "fall" in p.clip_type.lower()
        is_seizure = "seizure" in p.clip_type.lower()
        event_str = "fall" if is_fall else ("seizure" if is_seizure else "normal")

        cap = cv2.VideoCapture(str(clip_path))
        frame_idx = 0
        alert_fired = False
        future_frame_counter = 0

        while True:
            while self.paused:
                await asyncio.sleep(0.1)

            ret, frame = cap.read()
            if not ret: break

            frame_idx += 1
            h, w = frame.shape[:2]
            if h > w * 1.5:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            _, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            b64 = base64.b64encode(buf).decode('utf-8')

            fall_sm = 0.0
            sz_sm = 0.0
            if is_fall and frame_idx > 30 and frame_idx < 150:
                fall_sm = min((frame_idx - 30) / 100.0, 0.95)
            elif is_seizure and frame_idx > 30 and frame_idx < 150:
                sz_sm = min((frame_idx - 30) / 100.0, 0.95)

            payload = {
                "type": "frame_update",
                "frame_b64": b64,
                "fall_risk": fall_sm,
                "seizure_risk": sz_sm,
                "fps": 30
            }

            if event_str != "normal" and not alert_fired and frame_idx > 100:
                alert_fired = True
                print(f" -> Mock Alert Fired! {event_str} for Patient {p.id}")
                
                # Automatically Log Incident to Database
                db = SessionLocal()
                new_log = IncidentLog(patient_id=p.id, incident_type=event_str, confidence=0.88)
                db.add(new_log)
                db.commit()
                db.close()

                payload["alert"] = {
                    "type": "alert_fired",
                    "alert_id": p.id * 100 + frame_idx,
                    "event_type": event_str,
                    "confidence": 0.88,
                    "timestamp": time.strftime("%H:%M:%S")
                }
                future_frame_counter = 1

            await self.broadcast(payload)
            await asyncio.sleep(1/30)

            if future_frame_counter > 0:
                future_frame_counter += 1
                if future_frame_counter == 15:
                    await self.broadcast({
                        "type": "gemini_tier2",
                        "alert_id": p.id * 100 + frame_idx,
                        "decision": "CONFIRMED",
                        "reason": "Virtual UI Check OK"
                    })
                elif future_frame_counter == 60:
                    await self.broadcast({
                        "type": "gemini_report",
                        "alert_id": p.id * 100 + frame_idx,
                        "report": {
                            "decision": "CONFIRMED",
                            "headline": "UI Development Mock",
                            "narrative": f"PostgreSQL automatically saved this {event_str.upper()} incident to the logs.",
                            "severity": "high",
                            "actions": ["Dispatch assistance", "Review logs"]
                        }
                    })
                    future_frame_counter = 0

        cap.release()
        await self.broadcast({"type": "demo_complete"})

ui_pipeline = UIPipelineService()

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ui_pipeline.active_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("action") == "pause": 
                ui_pipeline.paused = True
            elif data.get("action") == "resume": 
                ui_pipeline.paused = False
            elif data.get("action") == "start":
                pt_id = int(data.get("patient_id", 1))
                if ui_pipeline.current_playback_task:
                    ui_pipeline.current_playback_task.cancel()
                ui_pipeline.current_playback_task = asyncio.create_task(ui_pipeline.play_patient(pt_id))
    except WebSocketDisconnect:
        ui_pipeline.active_websockets.remove(websocket)

if __name__ == "__main__":
    print("\n[SUCCESS] UI DEV PIPELINE LOADED WITH 0% ML OVERHEAD")
    print("Navgiate to: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
