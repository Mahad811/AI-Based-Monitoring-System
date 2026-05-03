import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, inspect as sa_inspect
from sqlalchemy.sql import func, text
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:1234@localhost:5432/postgres"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Nurse(Base):
    __tablename__ = "nurses"
    id              = Column(Integer, primary_key=True, index=True)
    staff_id        = Column(String, unique=True, index=True)
    name            = Column(String)
    password        = Column(String)
    role            = Column(String, default="Nurse")
    shift           = Column(String, default="Morning")
    ward_assignment = Column(String, nullable=True)
    status          = Column(String, default="off-duty")
    join_date       = Column(DateTime(timezone=True), server_default=func.now())
    last_login      = Column(DateTime(timezone=True), nullable=True)
    alerts_handled  = Column(Integer, default=0)


class Patient(Base):
    __tablename__ = "patients"
    id           = Column(Integer, primary_key=True, index=True)
    name         = Column(String)
    room         = Column(String)
    age          = Column(Integer)
    risk_profile = Column(String)
    clip_type    = Column(String)


class IncidentLog(Base):
    __tablename__ = "incident_logs"
    id            = Column(Integer, primary_key=True, index=True)
    patient_id    = Column(Integer)
    incident_type = Column(String)
    confidence    = Column(Float)
    timestamp     = Column(DateTime(timezone=True), server_default=func.now())
    narrative     = Column(Text, nullable=True)
    severity      = Column(String, nullable=True)
    actions_json  = Column(Text, nullable=True)
    frames_dir    = Column(String, nullable=True)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id          = Column(Integer, primary_key=True, index=True)
    actor_id    = Column(String)
    actor_name  = Column(String)
    action      = Column(String)
    target_type = Column(String, nullable=True)
    target_id   = Column(String, nullable=True)
    details     = Column(Text, nullable=True)
    timestamp   = Column(DateTime(timezone=True), server_default=func.now())


def _migrate_columns(eng):
    """Idempotently add any ORM columns missing from the live DB."""
    inspector = sa_inspect(eng)
    for model in (Nurse, Patient, IncidentLog, AuditLog):
        tname = model.__tablename__
        if not inspector.has_table(tname):
            continue
        existing = {c["name"] for c in inspector.get_columns(tname)}
        for col in model.__table__.columns:
            if col.name in existing:
                continue
            # Build a raw SQL ALTER TABLE … ADD COLUMN statement.
            col_type = col.type.compile(dialect=eng.dialect)
            default_clause = ""
            if col.server_default is not None:
                default_clause = f" DEFAULT {col.server_default.arg}"
            elif col.default is not None and col.default.is_scalar:
                val = col.default.arg
                if isinstance(val, str):
                    default_clause = f" DEFAULT '{val}'"
                else:
                    default_clause = f" DEFAULT {val}"
            nullable = "" if col.nullable else " NOT NULL"
            ddl = (f'ALTER TABLE {tname} ADD COLUMN IF NOT EXISTS '
                   f'"{col.name}" {col_type}{default_clause}{nullable}')
            with eng.connect() as conn:
                conn.execute(text(ddl))
                conn.commit()
            print(f"  [DB migration] Added column {tname}.{col.name}")


def init_db():
    # Migrate first (adds missing columns to existing tables)
    _migrate_columns(engine)
    # Then create any brand-new tables
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()

    # Seed Admin / Nurses
    if not db.query(Nurse).filter_by(staff_id="admin").first():
        db.add(Nurse(
            staff_id="admin", name="Admin Manager", password="admin",
            role="Admin", shift="Morning", ward_assignment="All Wards",
            status="on-duty",
        ))
    if not db.query(Nurse).filter_by(staff_id="nurse1").first():
        db.add(Nurse(
            staff_id="nurse1", name="Sarah Jenkins", password="securepassword",
            role="Nurse", shift="Morning", ward_assignment="ICU East",
            status="on-duty",
        ))

    # Seed 5 Patients
    if not db.query(Patient).first():
        patients = [
            Patient(name="Richard Hendricks", room="ICU 1", age=79,
                    risk_profile="High Fall Risk",     clip_type="fall_1"),
            Patient(name="Martha Stewart",    room="ICU 2", age=82,
                    risk_profile="High Fall Risk",     clip_type="fall_2"),
            Patient(name="John Watson",       room="ICU 3", age=48,
                    risk_profile="Stable",             clip_type="normal_1"),
            Patient(name="Sarah Palmer",      room="ICU 4", age=35,
                    risk_profile="Epilepsy Protocol",  clip_type="seizure_1"),
            Patient(name="David Miller",      room="ICU 5", age=42,
                    risk_profile="Epilepsy Protocol",  clip_type="seizure_2"),
            Patient(name="Aisha Malik",       room="PED-WC01", age=8, 
                    risk_profile="Pediatric - Whooping Cough", clip_type="whooping_cough_video"),
            Patient(name="Zainab Tariq",      room="ISO-AS02", age=25, 
                    risk_profile="Respiratory - Asthma Attack", clip_type="asthma_attack_video")
        ]
        db.bulk_save_objects(patients)
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

    # Always enforce clean room labels (fixes existing seeded records)
    _clean_rooms = {"fall_1": "ICU 1", "fall_2": "ICU 2", "normal_1": "ICU 3",
                    "seizure_1": "ICU 4", "seizure_2": "ICU 5"}
    for clip, room in _clean_rooms.items():
        db.query(Patient).filter(Patient.clip_type == clip).update({"room": room})

    db.commit()
    db.close()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
