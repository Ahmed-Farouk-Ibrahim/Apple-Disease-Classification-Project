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
    all_params: dict


# Data class for model training configuration.
@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    built_model_path: Path
    trained_model_path: Path    
    training_data: Path
    all_params: dict


@dataclass(frozen=True)
class EvaluationConfig:
    root_dir: Path
    trained_model_json_path: Path
    trained_model_weights_path: Path
    training_data: Path
    all_params: dict
        