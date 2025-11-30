from sqlalchemy import create_engine, Column, String, JSON, DateTime, Integer, Float, Boolean, ForeignKey, Enum as SQLEnum, Text
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime
from nexa_compute.api.config import get_settings
from nexa_compute.api.models import JobStatus, JobType, WorkerStatus, BillingResourceType

settings = get_settings()

engine = create_engine(
    settings.DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class UserDB(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    api_keys = relationship("ApiKeyDB", back_populates="user")

class ApiKeyDB(Base):
    __tablename__ = "api_keys"
    
    key_id = Column(String, primary_key=True, index=True)
    key_hash = Column(String, index=True) # Store hashed key
    key_prefix = Column(String) # Store first few chars for display
    name = Column(String)
    user_id = Column(String, ForeignKey("users.user_id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)
    is_active = Column(Boolean, default=True)
    
    user = relationship("UserDB", back_populates="api_keys")

class JobDB(Base):
    __tablename__ = "jobs"

    job_id = Column(String, primary_key=True, index=True)
    job_type = Column(SQLEnum(JobType), index=True)
    user_id = Column(String, index=True) # In real app, ForeignKey to users
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    payload = Column(JSON)
    worker_id = Column(String, index=True, nullable=True)
    result = Column(JSON, nullable=True)
    error = Column(String, nullable=True)
    logs = Column(Text, nullable=True)  # Store job execution logs

class WorkerDB(Base):
    __tablename__ = "workers"

    worker_id = Column(String, primary_key=True, index=True)
    hostname = Column(String)
    ip_address = Column(String, nullable=True)
    status = Column(SQLEnum(WorkerStatus), default=WorkerStatus.IDLE)
    gpu_type = Column(String, nullable=True)
    gpu_count = Column(Integer, default=0)
    last_heartbeat = Column(DateTime, default=datetime.utcnow)
    current_job_id = Column(String, nullable=True)

class BillingRecordDB(Base):
    __tablename__ = "billing_records"

    record_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    job_id = Column(String, index=True, nullable=True)
    resource_type = Column(SQLEnum(BillingResourceType), index=True)
    quantity = Column(Float)
    unit_price = Column(Float)
    total_cost = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
