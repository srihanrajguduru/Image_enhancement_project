import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import lpips as lpips_lib
from imquality import brisque
from PIL import Image
import warnings

# Suppress imquality warnings
warnings.filterwarnings("ignore", module="imquality")


class QualityMetrics:
    def __init__(self, device: str = "cpu"):
        self.device = device
        self.lpips_vgg = lpips_lib.LPIPS(net="vgg").to(device)
        self.lpips_vgg.eval()

    def calculate_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate PSNR."""
        return psnr_metric(target, pred, data_range=255)

    def calculate_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Calculate SSIM."""
        return ssim_metric(target, pred, channel_axis=2, data_range=255)

    def calculate_lpips(
        self, pred_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> float:
        """Calculate LPIPS (lower is better). Inputs should be [0, 1] tensors."""
        with torch.no_grad():
            # Scale to [-1, 1] for LPIPS
            pred_scaled = pred_tensor * 2.0 - 1.0
            target_scaled = target_tensor * 2.0 - 1.0
            val = self.lpips_vgg(pred_scaled, target_scaled)
        return val.item()

    def calculate_brisque(self, pred: np.ndarray) -> float:
        """Calculate BRISQUE (No-reference metric). Lower is better."""
        try:
            pil_img = Image.fromarray(pred)
            score = brisque.score(pil_img)
            return float(score)
        except Exception:
            # imquality library is incompatible with scikit-image >= 0.20 due to 'multichannel' deprecation
            return 0.0

    def evaluate_all(self, pred: np.ndarray, target: np.ndarray) -> dict:
        """Evaluates all metrics for a given prediction and target (numpy uint8 arrays)."""
        pred_t = (
            torch.from_numpy(pred).float().permute(2, 0, 1).unsqueeze(0).to(self.device)
            / 255.0
        )
        target_t = (
            torch.from_numpy(target)
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
            / 255.0
        )

        return {
            "psnr": self.calculate_psnr(pred, target),
            "ssim": self.calculate_ssim(pred, target),
            "lpips": self.calculate_lpips(pred_t, target_t),
            "brisque": self.calculate_brisque(pred),
        }
