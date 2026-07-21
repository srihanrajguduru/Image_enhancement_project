from abc import ABC, abstractmethod
import torch


class BaseModel(ABC):
    """Abstract interface for all restoration models."""

    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Loads the model weights and prepares it for inference."""

    @abstractmethod
    def restore(self, image: torch.Tensor) -> torch.Tensor:
        """
        Runs the restoration process on a given image tensor.

        Args:
            image: (1, C, H, W) Tensor in [0, 1] range.

        Returns:
            Restored (1, C, H, W) Tensor in [0, 1] range.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Runs a dummy inference step to compile/warm up the model."""

    def unload(self) -> None:
        """Removes the model from GPU memory to free up VRAM."""
        if self.model is not None:
            del self.model
            self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.is_loaded = False
