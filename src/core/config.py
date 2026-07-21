import yaml
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class AppConfig(BaseModel):
    name: str = "Image Restoration AI"
    version: str = "1.0.0"
    debug: bool = False
    host: str = "0.0.0.0"
    port: int = 8000


class PipelineConfig(BaseModel):
    device: str = "cuda"
    precision: str = "fp16"
    compile_models: bool = True
    enable_tensorrt: bool = False
    batch_size: int = 1
    max_workers: int = 4


class RouterConfig(BaseModel):
    haze_threshold: float = 0.5
    lowlight_threshold: float = 0.5
    rain_threshold: float = 0.5
    execution_order: List[str] = ["lowlight", "rain", "haze"]


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"
    file: str = "logs/restoration.log"


class SystemConfig(BaseModel):
    app: AppConfig
    pipeline: PipelineConfig
    router: RouterConfig
    logging: LoggingConfig


class ModelParams(BaseModel):
    name: str
    checkpoint_path: str
    download_url: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelsRegistryConfig(BaseModel):
    dehazeformer: ModelParams
    zero_dce: ModelParams
    restormer: ModelParams


from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_system_config(config_path: str = "configs/config.yaml") -> SystemConfig:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        data = yaml.safe_load(f)
    config = SystemConfig(**data)
    config.logging.file = str(PROJECT_ROOT / config.logging.file)
    return config


def load_models_config(
    config_path: str = "configs/models.yaml",
) -> ModelsRegistryConfig:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        data = yaml.safe_load(f)
    config = ModelsRegistryConfig(**data)
    # Resolve all checkpoint paths to absolute paths
    for model_cfg in [config.dehazeformer, config.zero_dce, config.restormer]:
        model_cfg.checkpoint_path = str(PROJECT_ROOT / model_cfg.checkpoint_path)
    return config


# Global configuration singletons
SYSTEM_CONFIG = None
MODELS_CONFIG = None


def get_config() -> SystemConfig:
    global SYSTEM_CONFIG
    if SYSTEM_CONFIG is None:
        SYSTEM_CONFIG = load_system_config()
    return SYSTEM_CONFIG


def get_models_config() -> ModelsRegistryConfig:
    global MODELS_CONFIG
    if MODELS_CONFIG is None:
        MODELS_CONFIG = load_models_config()
    return MODELS_CONFIG
