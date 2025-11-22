from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from nexa_compute.api.config import get_settings
from nexa_compute.api.database import init_db
from nexa_compute.api.endpoints import jobs, workers, billing, auth

settings = get_settings()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url=f"{settings.API_V1_STR}/docs",
    redoc_url=f"{settings.API_V1_STR}/redoc",
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
app.include_router(workers.router, prefix="/api/workers", tags=["workers"])
app.include_router(billing.router, prefix="/api/billing", tags=["billing"])
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "0.1.0"}

@app.get("/")
def root():
    return {"message": "Welcome to Nexa Forge API"}
