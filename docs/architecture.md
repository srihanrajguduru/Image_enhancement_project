# Architecture Guide

The system follows a highly modular, decoupled architecture following SOLID principles.

## 1. Detectors (`src/detectors/`)
Classical Computer Vision algorithms (OpenCV/NumPy) are used to generate confidence scores for specific degradations.
- `HazeDetector`: Uses Dark Channel Prior and transmission maps.
- `LowLightDetector`: Uses LAB luminance and Histogram thresholds.
- `BlurDetector`: Uses Laplacian variance and FFT magnitude.

## 2. Router (`src/core/router.py`)
Collects scores from all detectors and compares them against configured thresholds (`configs/config.yaml`). Generates an execution plan (e.g., `['lowlight', 'blur']`).

## 3. Models (`src/models/`)
Wrappers around PyTorch models. Models are loaded lazily to conserve VRAM and are optimized using `torch.autocast` and `torch.compile` during the `restore()` method.
- `DehazeFormerWrapper`
- `ZeroDCEWrapper`
- `NAFNetWrapper`

## 4. Pipeline (`src/core/pipeline.py`)
Orchestrates the entire process:
1. Loads Image
2. Runs Router
3. Sequentially applies models based on execution plan
4. Computes Quality Metrics (`src/utils/metrics.py`)
5. Saves result

## 5. Interfaces
- **API (`src/api/`)**: FastAPI wrapper around the pipeline.
- **CLI (`src/cli/`)**: Command line wrapper for batch processing.
