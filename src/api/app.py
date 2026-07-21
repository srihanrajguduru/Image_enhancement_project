from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pathlib import Path

from src.core.config import get_config
from src.core.pipeline import RestorationPipeline
from src.core.registry import registry
from src.utils.logger import get_logger

logger = get_logger(__name__)
config = get_config()

app = FastAPI(
    title="Image Restoration AI",
    version="1.0.0",
    description="API for automatic image restoration (Haze, Low-light, Rain)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = RestorationPipeline()

# Define paths
UPLOAD_DIR = Path("data/uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API server...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down API server...")
    registry.unload_all()


# Include routes
from src.api.routes import router as api_router

app.include_router(api_router)

if __name__ == "__main__":
    uvicorn.run(
        "src.api.app:app",
        host=config.app.host,
        port=config.app.port,
        reload=config.app.debug,
    )
