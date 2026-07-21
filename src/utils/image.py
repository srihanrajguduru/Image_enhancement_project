import cv2
import numpy as np
import torch
from typing import Union, Tuple
from pathlib import Path


def load_image(path: Union[str, Path]) -> np.ndarray:
    """Loads an image from path and returns as RGB numpy array (H, W, C) [0, 255]."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Failed to load image from {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_image(img: np.ndarray, path: Union[str, Path]) -> None:
    """Saves an RGB numpy array (H, W, C) [0, 255] to path."""
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def numpy_to_tensor(img: np.ndarray, device: str = "cpu") -> torch.Tensor:
    """Converts (H, W, C) [0, 255] numpy array to (1, C, H, W) [0.0, 1.0] tensor."""
    img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
    return img_tensor.unsqueeze(0).to(device)


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Converts (1, C, H, W) [0.0, 1.0] tensor back to (H, W, C) [0, 255] numpy array."""
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    img = (tensor.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    return img


def resize_for_inference(
    img: torch.Tensor, multiple: int = 16
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Pads tensor so that its spatial dimensions are multiples of `multiple`."""
    _, _, h, w = img.shape
    new_h = (h + multiple - 1) // multiple * multiple
    new_w = (w + multiple - 1) // multiple * multiple

    pad_h = new_h - h
    pad_w = new_w - w

    if pad_h > 0 or pad_w > 0:
        # Pad using reflection padding
        img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode="reflect")

    return img, (h, w)


def unpad_inference(img: torch.Tensor, original_size: Tuple[int, int]) -> torch.Tensor:
    """Crops the tensor back to its original size."""
    h, w = original_size
    return img[:, :, :h, :w]
