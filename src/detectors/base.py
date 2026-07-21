from abc import ABC, abstractmethod
import numpy as np


class BaseDetector(ABC):
    """Abstract base class for degradation detectors."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> float:
        """
        Analyzes the image and returns a confidence score.

        Args:
            image (np.ndarray): The input RGB image in [0, 255] format.

        Returns:
            float: A confidence score between 0.0 (no degradation) and 1.0 (severe degradation).
        """
