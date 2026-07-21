import time
from typing import Dict, Any, Union
from pathlib import Path

from src.core.config import get_config
from src.core.registry import registry
from src.core.router import DegradationRouter
from src.utils.logger import get_logger
from src.utils.image import (
    load_image,
    save_image,
    numpy_to_tensor,
    tensor_to_numpy,
    resize_for_inference,
    unpad_inference,
)
from src.utils.metrics import QualityMetrics

logger = get_logger(__name__)


class RestorationPipeline:
    """End-to-End Image Restoration Pipeline."""

    def __init__(self):
        self.config = get_config()
        self.router = DegradationRouter(registry.get_detectors())
        self.metrics = QualityMetrics(device=self.config.pipeline.device)

    def process_image(
        self, image_path: Union[str, Path], save_path: Union[str, Path] = None
    ) -> Dict[str, Any]:
        """
        Processes a single image.
        1. Loads image.
        2. Detects degradations.
        3. Routes to models.
        4. Applies models sequentially.
        5. Computes metrics.
        6. Saves restored image (if save_path is provided).
        """
        start_time = time.perf_counter()

        # 1. Load Image
        logger.info(f"Processing {image_path}")
        original_img_np = load_image(image_path)

        # 2. Detect and Route
        scores, execution_plan = self.router.execute_routing(original_img_np)
        logger.info(f"Degradation scores: {scores}")
        logger.info(f"Execution plan: {execution_plan}")

        # Fast path if no degradations
        if not execution_plan:
            logger.info(
                "No degradations detected above threshold. Skipping restoration."
            )
            if save_path:
                save_image(original_img_np, save_path)
            return {
                "scores": scores,
                "plan": [],
                "metrics": {},
                "latency_ms": (time.perf_counter() - start_time) * 1000,
            }

        # 3. Prepare tensor
        img_tensor = numpy_to_tensor(
            original_img_np, device=self.config.pipeline.device
        )
        img_tensor, original_size = resize_for_inference(img_tensor, multiple=16)

        # 4. Apply models
        for deg_type in execution_plan:
            logger.info(f"Applying model for: {deg_type}")
            model = registry.get_model(deg_type)
            img_tensor = model.restore(img_tensor)

        # Unpad and convert back to numpy
        img_tensor = unpad_inference(img_tensor, original_size)
        restored_img_np = tensor_to_numpy(img_tensor)

        # 5. Compute metrics
        quality_metrics = self.metrics.evaluate_all(restored_img_np, original_img_np)

        # 6. Save if needed
        if save_path:
            save_image(restored_img_np, save_path)

        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f"Finished processing in {latency:.2f} ms")



        return {
            "scores": scores,
            "plan": execution_plan,
            "metrics": quality_metrics,
            "latency_ms": latency,
        }
