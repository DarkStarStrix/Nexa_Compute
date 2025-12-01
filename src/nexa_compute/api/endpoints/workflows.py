"""Workflow management endpoints."""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from nexa_compute.orchestration.scheduler import get_scheduler
from nexa_compute.orchestration.workflow import WorkflowDefinition

router = APIRouter()


class WorkflowSubmitRequest(BaseModel):
    workflow_name: str
    parameters: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    run_id: str
    status: str
    workflow_name: str
    start_time: float
    end_time: Optional[float] = None


@router.post("/submit", response_model=WorkflowResponse)
def submit_workflow(request: WorkflowSubmitRequest):
    """Trigger a new workflow execution."""
    try:
        run_id = get_scheduler().trigger_workflow(
            request.workflow_name,
            request.parameters,
        )
        run = get_scheduler().get_run(run_id)
        if not run:
            raise HTTPException(status_code=500, detail="Failed to create workflow run")
            
        return WorkflowResponse(
            run_id=run.run_id,
            status=run.status.value,
            workflow_name=run.workflow_name,
            start_time=run.start_time,
            end_time=run.end_time,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{run_id}", response_model=WorkflowResponse)
def get_workflow_status(run_id: str):
    """Get status of a workflow run."""
    run = get_scheduler().get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Workflow run not found")
        
    return WorkflowResponse(
        run_id=run.run_id,
        status=run.status.value,
        workflow_name=run.workflow_name,
        start_time=run.start_time,
        end_time=run.end_time,
    )


@router.post("/{run_id}/cancel")
def cancel_workflow(run_id: str):
    """Cancel a running workflow."""
    success = get_scheduler().cancel_run(run_id)
    if not success:
        raise HTTPException(status_code=400, detail="Cannot cancel workflow (not found or already finished)")
    return {"status": "cancelled"}

