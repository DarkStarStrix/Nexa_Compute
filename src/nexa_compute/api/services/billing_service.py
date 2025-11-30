import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from nexa_compute.api.models import BillingResourceType, BillingSummary
from nexa_compute.api.database import BillingRecordDB

# Pricing Model (Pay-as-you-go)
# Base costs: $6/month VPS, $2/hour compute
# Target: $3/hour GPU (covers $2 cost + $1 profit margin)
# Additional services priced to cover API costs + small margin

RATES = {
    # GPU Compute: $3/hour (covers $2 compute cost + $1 profit)
    BillingResourceType.GPU_HOUR: 3.00,
    
    # Training: $3/hour per GPU (same as GPU_HOUR, but tracked separately)
    # Fine-tuning: $3/hour per GPU
    # Inference: $1/hour per GPU (lower overhead, shared resources)
    
    # Token-based pricing (for API calls)
    BillingResourceType.TOKEN_INPUT: 0.00001,  # $10/1M tokens
    BillingResourceType.TOKEN_OUTPUT: 0.00003,  # $30/1M tokens
    
    # Storage: $0.05/GB/month (covers S3/Wasabi costs + margin)
    BillingResourceType.STORAGE_GB_MONTH: 0.05
}

# Job type multipliers (some jobs use more resources)
JOB_TYPE_MULTIPLIERS = {
    "train": 1.0,      # Full GPU hour rate
    "distill": 0.8,    # Slightly less intensive
    "evaluate": 0.6,   # Evaluation is lighter
    "generate": 0.5,   # Data generation is CPU-bound
    "audit": 0.3,      # Audit is mostly I/O
    "deploy": 0.0,     # Deployment is one-time, no ongoing cost
}

class BillingService:
    def __init__(self, db: Session):
        self.db = db

    def record_usage(
        self, 
        user_id: str, 
        resource_type: BillingResourceType, 
        quantity: float, 
        job_id: Optional[str] = None,
        job_type: Optional[str] = None
    ) -> BillingRecordDB:
        unit_price = RATES.get(resource_type, 0.0)
        
        # Apply job type multiplier for GPU hours
        if resource_type == BillingResourceType.GPU_HOUR and job_type:
            multiplier = JOB_TYPE_MULTIPLIERS.get(job_type, 1.0)
            unit_price *= multiplier
        
        total_cost = quantity * unit_price
        
        record = BillingRecordDB(
            record_id=f"bill_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            job_id=job_id,
            resource_type=resource_type,
            quantity=quantity,
            unit_price=unit_price,
            total_cost=total_cost,
            timestamp=datetime.utcnow()
        )
        
        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def record_job_completion(
        self,
        user_id: str,
        job_id: str,
        job_type: str,
        gpu_hours: float,
        gpu_count: int = 1
    ) -> BillingRecordDB:
        """Record billing for a completed job."""
        # GPU hours are billed per GPU
        total_gpu_hours = gpu_hours * gpu_count
        return self.record_usage(
            user_id=user_id,
            resource_type=BillingResourceType.GPU_HOUR,
            quantity=total_gpu_hours,
            job_id=job_id,
            job_type=job_type
        )

    def get_summary(
        self, 
        user_id: str, 
        start_date: Optional[datetime] = None, 
        end_date: Optional[datetime] = None
    ) -> BillingSummary:
        if not start_date:
            start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if not end_date:
            end_date = datetime.utcnow()
            
        query = self.db.query(BillingRecordDB).filter(
            BillingRecordDB.user_id == user_id,
            BillingRecordDB.timestamp >= start_date,
            BillingRecordDB.timestamp <= end_date
        )
        
        records = query.all()
        
        total_cost = sum(r.total_cost for r in records)
        usage_by_type = {}
        cost_by_type = {}
        
        for r in records:
            rtype = r.resource_type.value
            usage_by_type[rtype] = usage_by_type.get(rtype, 0.0) + r.quantity
            cost_by_type[rtype] = cost_by_type.get(rtype, 0.0) + r.total_cost
            
        return BillingSummary(
            total_cost=total_cost,
            currency="USD",
            period_start=start_date,
            period_end=end_date,
            usage_by_type=usage_by_type,
            cost_by_type=cost_by_type
        )
