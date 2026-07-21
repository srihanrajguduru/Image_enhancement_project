# Image Restoration AI System

A complete, production-ready AI system that automatically restores images suffering from haze, low-light, and motion blur.

## Features
- **Automatic Degradation Detection**: Classical detectors (Dark Channel Prior, LAB Luminance, FFT Variance) automatically determine if an image suffers from haze, low-light, or blur.
- **Dynamic Routing**: Only the necessary state-of-the-art pretrained models are executed based on the detected degradations.
- **State-of-the-Art Models**:
  - Haze: DehazeFormer
  - Low-Light: Zero-DCE++
  - Blur: NAFNet
- **Optimized Inference**: FP16 autocasting, `torch.compile`, and efficient VRAM management (lazy loading).
- **Comprehensive API**: FastAPI server for RESTful inference.
- **CLI Interface**: Process single images or entire folders easily.
- **Quality Metrics**: Built-in PSNR, SSIM, LPIPS, and BRISQUE evaluation.
- **Production Ready**: Dockerized (CPU, GPU, Jetson), fully typed, and thoroughly tested.

## Installation

```bash
# Clone the repository
git clone <repository_url>
cd image_restore

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt # For testing
```

## Usage

### CLI
```bash
# Single Image
python -m src.cli.restore path/to/image.jpg --output output_dir/ --metrics

# Folder
python -m src.cli.restore path/to/folder/ --output output_dir/
```

### API
Start the server:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
API Documentation is available at `http://localhost:8000/docs`.

### Docker
```bash
docker-compose up --build
```

## Benchmarking
```bash
python scripts/benchmark.py --input path/to/test_images/ --output results.csv
```

## Documentation
- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Developer Guide](docs/developer.md)
