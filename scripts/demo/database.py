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
            Patient(name="Richard Hendricks", room="ICU-302A", age=79, risk_profile="High Fall Risk", clip_type="fall_1"),
            Patient(name="Martha Stewart", room="ICU-304B", age=82, risk_profile="High Fall Risk", clip_type="fall_2"),
            Patient(name="John Watson", room="ICU-101A", age=48, risk_profile="Stable", clip_type="normal_1"),
            Patient(name="Sarah Palmer", room="ICU-205C", age=35, risk_profile="Epilepsy Protocol", clip_type="seizure_1"),
            Patient(name="David Miller", room="ICU-206C", age=42, risk_profile="Epilepsy Protocol", clip_type="seizure_2")
        ]
        db.bulk_save_objects(patients)
        
    db.commit()
    db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
