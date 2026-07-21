import torch
import torch.nn as nn
from typing import Callable
import time


def optimize_model(
    model: nn.Module,
    device: str = "cuda",
    precision: str = "fp16",
    compile_model: bool = True,
) -> nn.Module:
    """Applies inference optimizations to a PyTorch model."""

    model = model.to(device)
    model.eval()

    if precision == "fp16":
        model = model.half()
    elif precision == "bf16":
        model = model.bfloat16()

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            import logging

            logging.getLogger("optimization").warning(
                f"torch.compile failed, skipping: {e}"
            )

    return model


def infer_with_autocast(
    model: nn.Module,
    inputs: torch.Tensor,
    precision: str = "fp16",
    device: str = "cuda",
) -> torch.Tensor:
    """Runs inference with automatic mixed precision."""
    dtype = (
        torch.float16
        if precision == "fp16"
        else torch.bfloat16 if precision == "bf16" else torch.float32
    )
    device_type = "cuda" if "cuda" in device else "cpu"

    with torch.no_grad():
        with torch.autocast(
            device_type=device_type, dtype=dtype, enabled=precision != "fp32"
        ):
            outputs = model(inputs)
    return outputs


def profile_inference(
    model_func: Callable,
    inputs: torch.Tensor,
    warmup_iters: int = 5,
    measure_iters: int = 20,
) -> dict:
    """Profiles a model execution for runtime and memory."""
    # Warmup
    for _ in range(warmup_iters):
        _ = model_func(inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start_time = time.perf_counter()
    for _ in range(measure_iters):
        _ = model_func(inputs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.perf_counter()

    avg_latency = ((end_time - start_time) / measure_iters) * 1000  # ms
    vram_mb = (
        torch.cuda.max_memory_allocated() / (1024**2)
        if torch.cuda.is_available()
        else 0
    )
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    return {
        "latency_ms": round(avg_latency, 2),
        "fps": round(fps, 2),
        "vram_mb": round(vram_mb, 2),
    }
