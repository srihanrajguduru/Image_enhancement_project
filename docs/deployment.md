# Deployment Guide

## Hardware Requirements
- **GPU**: NVIDIA GPU with CUDA support is recommended for fast inference (FP16).
- **RAM**: Minimum 8GB (16GB recommended for batch processing).

## Docker Deployment (Recommended)

The project includes three Dockerfiles:
1. `Dockerfile`: CPU only.
2. `Dockerfile.gpu`: NVIDIA GPU (requires nvidia-container-toolkit).
3. `Dockerfile.jetson`: Optimized for NVIDIA Jetson devices.

### Using Docker Compose
```bash
docker-compose up -d --build
```
This will start the FastAPI server on port 8000 with GPU support.

## Environment Variables
- `PYTHONDONTWRITEBYTECODE=1`
- `PYTHONUNBUFFERED=1`

## Downloading Checkpoints
Before deploying, ensure you place the `.pth` files in the `weights/` directory and update the `configs/models.yaml` with the correct paths.

## Metrics & Monitoring
The application outputs structured JSON logs to `logs/restoration.log`. You can ingest these logs into ELK or Datadog for monitoring latency, GPU VRAM usage, and quality scores in production.
