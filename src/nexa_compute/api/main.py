from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from nexa_compute.api.config import get_settings
from nexa_compute.api.database import init_db
from nexa_compute.api.endpoints import auth, billing, artifacts, jobs, workers

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize application resources at startup."""
    init_db()

    artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["datasets", "checkpoints", "evals", "deployments"]:
        (artifacts_dir / subdir).mkdir(parents=True, exist_ok=True)

    if os.getenv("STORAGE_BACKEND", settings.STORAGE_BACKEND) == "local":
        from nexa_compute.utils.storage import get_storage

        get_storage()

    yield
    # No teardown required currently


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
    lifespan=lifespan,
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(workers.router, prefix="/api/workers", tags=["workers"])
app.include_router(billing.router, prefix="/api/billing", tags=["billing"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(artifacts.router, prefix="/api/artifacts", tags=["artifacts"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/")
def root():
    return {"message": "Welcome to Nexa Forge API"}
