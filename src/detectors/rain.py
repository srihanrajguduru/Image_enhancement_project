import cv2
import numpy as np
from src.detectors.base import BaseDetector

class RainDetector(BaseDetector):
    """Detects rain streaks using vertical/diagonal edge analysis and high frequency extraction."""
    def __init__(self, rain_threshold: float = 0.02):
        self.rain_threshold = rain_threshold

    def _analyze_streaks(self, image: np.ndarray) -> float:
        """Analyzes directional edge gradients to detect rain streaks."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # 1. Isolate high-frequency noise (rain) by subtracting a blurred version
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(gray, blurred)
        
        # 2. Apply threshold to get only strong streaks
        _, thresh = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
        
        # 3. Find vertical-ish streaks using a morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7)) # Vertical kernel
        vertical_streaks = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Calculate density of these vertical streaks
        streak_density = np.count_nonzero(vertical_streaks) / vertical_streaks.size
        
        # Normalizing score based on density
        score = max(0.0, min(1.0, streak_density / self.rain_threshold))
        return float(score)

    def detect(self, image: np.ndarray) -> float:
        score = self._analyze_streaks(image)
        return score
