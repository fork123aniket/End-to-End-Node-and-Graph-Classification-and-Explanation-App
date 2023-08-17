from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    graph_root_dir: Path
    apply_data_transform: bool
    dataset_name: str
    graph_dataset_name: str
    data_split: str
    batch_size: int
    num_train_samples: int
    num_val_samples: int
    num_test_samples: int


@dataclass(frozen=True)
class TuningConfig:
    num_epochs: int
    num_features: int
    num_classes: int
    graph_num_features: int
    graph_num_classes: int
    n_startup_trials: int
    n_warmup_steps: int
    interval_steps: int
    n_trials: int
    direction: str


@dataclass(frozen=True)
class TrainingConfig:
    model_registry_path: Path
    graph_model_registry_path: Path
    num_epochs: int
    weight_decay_ratio: float
    learning_rate: float
    num_features: int
    num_classes: int
    graph_num_features: int
    graph_num_classes: int
    num_hidden_features: int
    num_hidden_1: int
    num_hidden_2: int
    graph_learning_rate: float
    graph_weight_decay_ratio: float
    exp_name: str
