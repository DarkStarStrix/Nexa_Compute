from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import Optional
from datetime import datetime
from nexa_compute.api.database import get_db
from nexa_compute.api.models import BillingSummary
from nexa_compute.api.services.billing_service import BillingService

router = APIRouter()

def get_billing_service(db: Session = Depends(get_db)) -> BillingService:
    return BillingService(db)

@router.get("/summary", response_model=BillingSummary)
def get_billing_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    user_id: str = "default_user",
    service: BillingService = Depends(get_billing_service)
):
    # Auto-generate mock data if empty
    service.generate_mock_data(user_id)
    return service.get_summary(user_id, start_date, end_date)
