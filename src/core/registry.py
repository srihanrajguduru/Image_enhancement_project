from typing import Dict
from src.detectors.base import BaseDetector
from src.detectors.haze import HazeDetector
from src.detectors.lowlight import LowLightDetector
from src.detectors.rain import RainDetector

from src.models.base import BaseModel
from src.models.dehaze import DehazeFormerWrapper
from src.models.lowlight import ZeroDCEWrapper
from src.models.derain import RestormerWrapper


class ModelRegistry:
    """Registry to lazily initialize and manage models and detectors."""

    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        self._models: Dict[str, BaseModel] = {}

    def get_detectors(self) -> Dict[str, BaseDetector]:
        if not self._detectors:
            self._detectors = {
                "haze": HazeDetector(),
                "lowlight": LowLightDetector(),
                "rain": RainDetector(),
            }
        return self._detectors

    def get_model(self, name: str) -> BaseModel:
        if name not in self._models:
            if name == "haze":
                self._models[name] = DehazeFormerWrapper()
            elif name == "lowlight":
                self._models[name] = ZeroDCEWrapper()
            elif name == "rain":
                self._models[name] = RestormerWrapper()
            else:
                raise ValueError(f"Unknown model name: {name}")

        # Lazy loading happens inside the model wrapper's restore() method
        return self._models[name]

    def unload_all(self):
        """Unloads all models to free VRAM."""
        for model in self._models.values():
            model.unload()


registry = ModelRegistry()
