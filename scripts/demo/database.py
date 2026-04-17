from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = "postgresql://postgres:1234@localhost:5432/postgres"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Nurse(Base):
    __tablename__ = "nurses"
    id = Column(Integer, primary_key=True, index=True)
    staff_id = Column(String, unique=True, index=True)
    name = Column(String)
    password = Column(String)

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    room = Column(String)
    age = Column(Integer)
    risk_profile = Column(String)
    clip_type = Column(String)

class IncidentLog(Base):
    __tablename__ = "incident_logs"
    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer)
    incident_type = Column(String)
    confidence = Column(Float)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

def init_db():
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    # Seed Admin / Nurses
    if not db.query(Nurse).filter_by(staff_id="admin").first():
        db.add(Nurse(staff_id="admin", name="Admin Manager", password="admin"))
    if not db.query(Nurse).filter_by(staff_id="nurse1").first():
        db.add(Nurse(staff_id="nurse1", name="Sarah Jenkins", password="securepassword"))
        
    # Seed 5 Patients
    if not db.query(Patient).first():
        patients = [
            Patient(name="Richard Davis", room="ICU-107", age=72, risk_profile="High-Risk Mobility / Frequent Falls", clip_type="fall_1"),
            Patient(name="Martha Kent", room="ICU-112", age=68, risk_profile="Rehab / Monitored Rest", clip_type="normal_1"),
            Patient(name="John Smith", room="ICU-105", age=81, risk_profile="Post-Op Orthopedic Care", clip_type="fall_2"),
            Patient(name="Sarah Connor", room="ICU-204B", age=45, risk_profile="Epilepsy Protocol", clip_type="seizure_1"),
            Patient(name="David Miller", room="ICU-206C", age=50, risk_profile="Epilepsy Protocol", clip_type="seizure_2"),
            Patient(name="Aisha Malik", room="PED-WC01", age=8, risk_profile="Pediatric - Whooping Cough", clip_type="whooping_cough_video"),
            Patient(name="Zainab Tariq", room="ISO-AS02", age=25, risk_profile="Respiratory - Asthma Attack", clip_type="asthma_attack_video")
        ]
        db.bulk_save_objects(patients)
        
        # Seed 1 Admin
        db.add(Nurse(staff_id="admin", name="System Admin", password="admin"))
        db.add(Nurse(staff_id="nurse1", name="Alice Wonderland", password="securepassword"))
        db.commit()
    else:
        # Check if custom patient exists dynamically to fix missing cards
        custom_wc = db.query(Patient).filter_by(clip_type="whooping_cough_video").first()
        if not custom_wc:
            db.add(Patient(name="Aisha Malik", room="PED-WC01", age=8, risk_profile="Pediatric - Whooping Cough", clip_type="whooping_cough_video"))
            
        custom_as = db.query(Patient).filter_by(clip_type="asthma_attack_video").first()
        if not custom_as:
            db.add(Patient(name="Zainab Tariq", room="ISO-AS02", age=25, risk_profile="Respiratory - Asthma Attack", clip_type="asthma_attack_video"))

        live = db.query(Patient).filter_by(clip_type="live_feed").first()
        if not live:
            db.add(Patient(name="LIVE HARDWARE FEED", room="ICU-LIVE", age=0, risk_profile="Edge Mode: WebCam + Mic", clip_type="live_feed"))
        
        db.commit()
    db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
