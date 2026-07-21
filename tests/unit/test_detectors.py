import numpy as np
import pytest
from src.detectors.haze import HazeDetector
from src.detectors.lowlight import LowLightDetector
from src.detectors.blur import BlurDetector

def test_haze_detector():
    detector = HazeDetector()
    # Create a dummy image (e.g., all white, very hazy)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    score = detector.detect(img)
    assert 0.0 <= score <= 1.0

def test_lowlight_detector():
    detector = LowLightDetector()
    # Create a dark image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    score = detector.detect(img)
    assert 0.0 <= score <= 1.0
    
def test_blur_detector():
    detector = BlurDetector()
    # Create a uniform image (no edges, highly blurred)
    img = np.ones((100, 100, 3), dtype=np.uint8) * 128
    score = detector.detect(img)
    assert 0.0 <= score <= 1.0
    assert score > 0.8  # Should have a high blur score
