# Image Restoration AI System

A complete, production-ready AI system that automatically restores images suffering from haze, low-light, and motion blur using state-of-the-art models and CUDA acceleration.

## Features
- **Automatic Degradation Detection**: Classical detectors (Dark Channel Prior, LAB Luminance, FFT Variance) automatically determine if an image suffers from haze, low-light, or blur.
- **Dynamic Routing**: Only the necessary pretrained models are executed based on the detected degradations.
- **State-of-the-Art Models**:
  - Haze: DehazeFormer
  - Low-Light: Zero-DCE++
  - Blur/Rain: Restormer / NAFNet
- **Optimized Inference**: FP16 autocasting, `torch.compile`, and efficient VRAM management (lazy loading).
- **Comprehensive API**: FastAPI server for RESTful inference.
- **CLI Interface**: Process single images or entire folders effortlessly.
- **Quality Metrics**: Built-in PSNR, SSIM, LPIPS, and BRISQUE evaluation.

## Installation (Windows / CUDA 12.1)

We recommend using Conda to set up the environment, especially on Windows, to ensure seamless compatibility with NVIDIA CUDA libraries.

```bash
# 1. Clone the repository
git clone https://github.com/srihanrajguduru/Image_enhancement_project.git
cd Image_enhancement_project

# 2. Create a local Conda environment with Python 3.11
conda create --prefix ./env python=3.11 -y

# 3. Activate the environment
conda activate ./env

# 4. Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install the remaining project requirements
pip install -r requirements.txt
```

> **Note for PowerShell users**: If `conda activate .\env` fails to activate properly, you can bypass the activation step and use the Python executable directly: `.\env\python.exe` instead of `python`.

## Downloading Checkpoints

Before running inference, you must download the pre-trained model weights (`.pth` files) and place them in the `weights/` directory. Check `configs/models.yaml` for specific file names and URLs.

- Example: `weights/deraining.pth`
- Example: `weights/dehazeformer-b.pth`
- Example: `weights/Epoch99.pth`

## Usage

### CLI (Command Line Interface)

Process a single image or an entire folder of images.

```bash
# Process a single image (with metrics)
python -m src.cli.restore images/norain-105x2.png --output output_dir/restored.png --metrics

# Process an entire folder
python -m src.cli.restore images/ --output output_dir/
```

### API Server

Start the FastAPI server for RESTful inference:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
API Documentation will be available at `http://localhost:8000/docs`.

### Benchmarking
Evaluate the model against a dataset:
```bash
python scripts/benchmark.py --input images/ --output results.csv
```

## Docker Deployment

The project includes Dockerfiles for CPU, GPU, and Jetson devices.
```bash
docker-compose up --build
```

## Documentation
- [Architecture Guide](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)
- [Developer Guide](docs/developer.md)
