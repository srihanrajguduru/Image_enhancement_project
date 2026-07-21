import torch
from src.models.base import BaseModel
from src.core.config import get_models_config, get_config
from src.utils.optimization import optimize_model, infer_with_autocast
from src.utils.logger import get_logger

logger = get_logger(__name__)

from src.models.zero_dce_arch import enhance_net_nopool


class ZeroDCEWrapper(BaseModel):
    """Wrapper for Zero-DCE++ for low-light enhancement."""

    def __init__(self):
        super().__init__("Zero-DCE++")
        self.config = get_models_config().zero_dce
        self.sys_config = get_config().pipeline

    def load(self) -> None:
        if self.is_loaded:
            return

        logger.info(f"Loading {self.name} from {self.config.checkpoint_path}")

        self.model = enhance_net_nopool(scale_factor=1)

        try:
            state_dict = torch.load(
                self.config.checkpoint_path, map_location="cpu", weights_only=False
            )

            # The official checkpoints often have "module." prefix if trained with DataParallel.
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    clean_state_dict[k[7:]] = v
                else:
                    clean_state_dict[k] = v

            self.model.load_state_dict(clean_state_dict)
        except Exception as e:
            logger.warning(
                f"Could not load weights for {self.name}, using random initialization. Error: {e}"
            )

        self.model = optimize_model(
            self.model,
            device=self.sys_config.device,
            precision=self.sys_config.precision,
            compile_model=self.sys_config.compile_models,
        )

        self.is_loaded = True

    def warmup(self) -> None:
        if not self.is_loaded:
            self.load()
        logger.info(f"Warming up {self.name}")
        dummy_input = torch.randn(1, 3, 256, 256, device=self.sys_config.device)
        _ = self.restore(dummy_input)

    def restore(self, image: torch.Tensor) -> torch.Tensor:
        if not self.is_loaded:
            self.load()

        out = infer_with_autocast(
            self.model,
            image.to(self.sys_config.device),
            precision=self.sys_config.precision,
            device=self.sys_config.device,
        )

        # Zero-DCE++ returns a tuple: (enhanced_image, illumination_map)
        if isinstance(out, (list, tuple)):
            out = out[0]

        return out
