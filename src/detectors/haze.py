import cv2
import numpy as np
from src.detectors.base import BaseDetector


class HazeDetector(BaseDetector):
    """Detects haze using Dark Channel Prior and Transmission Estimation."""

    def __init__(self, patch_size: int = 15):
        self.patch_size = patch_size

    def _get_dark_channel(self, image: np.ndarray) -> np.ndarray:
        """Calculates the dark channel of an image."""
        min_channel = np.min(image, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.patch_size, self.patch_size)
        )
        dark_channel = cv2.erode(min_channel, kernel)
        return dark_channel

    def _estimate_atmospheric_light(
        self, image: np.ndarray, dark_channel: np.ndarray
    ) -> np.ndarray:
        """Estimates the atmospheric light."""
        h, w = dark_channel.shape
        num_pixels = h * w
        num_brightest = int(max(num_pixels * 0.001, 1))

        # Find indices of the brightest pixels in the dark channel
        indices = np.argsort(dark_channel.flatten())[::-1][:num_brightest]

        # Get corresponding pixels in the original image
        image_flat = image.reshape(num_pixels, 3)
        brightest_pixels = image_flat[indices]

        # Atmospheric light is the mean of these brightest pixels
        A = np.mean(brightest_pixels, axis=0)
        return A

    def _estimate_transmission(
        self, image: np.ndarray, A: np.ndarray, omega: float = 0.95
    ) -> np.ndarray:
        """Estimates the transmission map."""
        norm_image = np.empty(image.shape, image.dtype)
        for i in range(3):
            norm_image[:, :, i] = image[:, :, i] / A[i]

        transmission = 1 - omega * self._get_dark_channel(norm_image)
        return transmission

    def detect(self, image: np.ndarray) -> float:
        """
        Calculates haze confidence based on average transmission.
        Lower transmission implies more haze. We map this to a score [0, 1].
        """
        # Convert to float [0, 1] for processing
        img_float = image.astype("float64") / 255.0

        dark_channel = self._get_dark_channel(img_float)
        A = self._estimate_atmospheric_light(img_float, dark_channel)

        # Ensure A is not 0
        A = np.maximum(A, 0.001)

        transmission = self._estimate_transmission(img_float, A)

        # Mean transmission over the image
        mean_t = np.mean(transmission)

        # Map to confidence: low transmission -> high haze score
        # Typically t ranges from 0 to 1.
        # A clear image has t close to 1. A hazy image has low t (e.g., 0.4).
        confidence = max(0.0, min(1.0, 1.0 - mean_t))

        # Apply a sigmoid-like scaling to make scores more distinct
        scaled_confidence = 1 / (1 + np.exp(-10 * (confidence - 0.5)))
        return float(scaled_confidence)
