from typing import Dict, List
import numpy as np
from src.detectors.base import BaseDetector
from src.core.config import get_config


class DegradationRouter:
    """Routes an image to the appropriate restoration models based on detected degradations."""

    def __init__(self, detectors: Dict[str, BaseDetector]):
        self.detectors = detectors
        self.config = get_config().router
        self.thresholds = {
            "haze": self.config.haze_threshold,
            "lowlight": self.config.lowlight_threshold,
            "rain": self.config.rain_threshold,
        }

    def detect_all(self, image: np.ndarray) -> Dict[str, float]:
        """Runs all configured detectors and returns their confidence scores."""
        scores = {}
        for name, detector in self.detectors.items():
            scores[name] = detector.detect(image)
        return scores

    def route(self, scores: Dict[str, float]) -> List[str]:
        """
        Determines the execution plan (which models to run in which order).
        """
        active_degradations = []
        for deg in self.config.execution_order:
            if deg in scores and scores[deg] > self.thresholds.get(deg, 0.5):
                active_degradations.append(deg)

        return active_degradations

    def execute_routing(self, image: np.ndarray) -> tuple[Dict[str, float], List[str]]:
        """Convenience method to detect and route in one step."""
        scores = self.detect_all(image)
        plan = self.route(scores)
        return scores, plan
