import cv2
import numpy as np
from src.detectors.base import BaseDetector


class LowLightDetector(BaseDetector):
    """Detects low-light conditions using Histogram and LAB luminance analysis."""

    def __init__(self, low_light_threshold: int = 50, dark_pixel_ratio: float = 0.6):
        self.low_light_threshold = low_light_threshold
        self.dark_pixel_ratio = dark_pixel_ratio

    def _analyze_lab_luminance(self, image: np.ndarray) -> float:
        """Analyzes L channel in LAB color space."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, _, _ = cv2.split(lab)

        # L channel ranges from 0 to 255 in OpenCV (mapped from 0 to 100)
        mean_luminance = np.mean(l_channel)

        # Map mean luminance to a score (lower luminance -> higher score)
        # Assuming completely dark is L=0, very bright is L=255
        score = max(0.0, min(1.0, 1.0 - (mean_luminance / 128.0)))
        return score

    def _analyze_histogram(self, image: np.ndarray) -> float:
        """Analyzes the histogram to check for skewed dark pixels."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Count pixels below the threshold
        dark_pixels = np.sum(gray < self.low_light_threshold)
        total_pixels = gray.size

        ratio = dark_pixels / total_pixels

        # Map ratio to a score (higher ratio -> higher score)
        score = max(0.0, min(1.0, ratio / self.dark_pixel_ratio))
        return score

    def detect(self, image: np.ndarray) -> float:
        """
        Calculates low-light confidence based on a combination of LAB and histogram.
        """
        lab_score = self._analyze_lab_luminance(image)
        hist_score = self._analyze_histogram(image)

        # Combine scores (weighted average)
        confidence = 0.6 * lab_score + 0.4 * hist_score

        return float(max(0.0, min(1.0, confidence)))
