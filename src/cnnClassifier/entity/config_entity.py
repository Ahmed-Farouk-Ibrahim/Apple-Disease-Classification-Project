from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


# Data class for storing model preparation configuration
@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    built_model_path: Path    
    params_image_size: list
    params_n_classes: int
    params_learning_rate: float
    params_rho: float
    params_epsilon: float

