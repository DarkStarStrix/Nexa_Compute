import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func
from nexa_compute.api.models import BillingResourceType, BillingSummary
from nexa_compute.api.database import BillingRecordDB

# Mock rates
RATES = {
    BillingResourceType.GPU_HOUR: 2.50,  # $2.50/hr for A100
    BillingResourceType.TOKEN_INPUT: 0.00001, # $10/1M tokens
    BillingResourceType.TOKEN_OUTPUT: 0.00003, # $30/1M tokens
    BillingResourceType.STORAGE_GB_MONTH: 0.02 # $0.02/GB/month
}

class BillingService:
    def __init__(self, db: Session):
        self.db = db

    def record_usage(
        self, 
        user_id: str, 
        resource_type: BillingResourceType, 
        quantity: float, 
        job_id: Optional[str] = None
    ) -> BillingRecordDB:
        unit_price = RATES.get(resource_type, 0.0)
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
        
    def generate_mock_data(self, user_id: str):
        """Generate some mock billing data for demonstration."""
        # Check if we already have data
        if self.db.query(BillingRecordDB).filter(BillingRecordDB.user_id == user_id).first():
            return

        # Generate last 30 days of data
        base_time = datetime.utcnow() - timedelta(days=30)
        
        import random
        
        for i in range(30):
            day = base_time + timedelta(days=i)
            
            # Random GPU usage
            if random.random() > 0.3:
                self.record_usage(
                    user_id=user_id,
                    resource_type=BillingResourceType.GPU_HOUR,
                    quantity=random.uniform(0.5, 8.0),
                    job_id=f"job_mock_{i}"
                ).timestamp = day # Hack to set past date
                
            # Random Token usage
            if random.random() > 0.2:
                self.record_usage(
                    user_id=user_id,
                    resource_type=BillingResourceType.TOKEN_INPUT,
                    quantity=random.randint(1000, 100000),
                    job_id=f"job_mock_{i}"
                ).timestamp = day
                
                self.record_usage(
                    user_id=user_id,
                    resource_type=BillingResourceType.TOKEN_OUTPUT,
                    quantity=random.randint(500, 50000),
                    job_id=f"job_mock_{i}"
                ).timestamp = day
        
        # Fix timestamps (since record_usage sets to utcnow)
        # This is just for the mock generator, in real app we wouldn't do this
        pass
