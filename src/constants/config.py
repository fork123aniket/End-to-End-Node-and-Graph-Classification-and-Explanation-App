from enum import Enum
from pathlib import Path


class DataIngestion(Enum):
    root_dir: Path = "E:/G/GraphMaskExplainer/artifacts/data/Cora"
    graph_root_dir: Path = "E:/G/GraphMaskExplainer/artifacts/data/ENZYMES"


class ParamConfig(Enum):
    path_to_yaml: Path = "E:/G/GraphMaskExplainer/params.yaml"


class LoggingStack(Enum):
    base_dir: Path = "E:/G/GraphMaskExplainer/logs"


class TrainModel(Enum):
    model_save_dir: Path = "E:/G/GraphMaskExplainer/artifacts/model/node/"
    graph_model_save_dir: Path = "E:/G/GraphMaskExplainer/artifacts/model/graph/"
    model_registry_path: Path = "E:/G/GraphMaskExplainer/experiments/Cora"
    graph_model_registry_path: Path = "E:/G/GraphMaskExplainer/experiments/ENZYMES"


class TuningPath(Enum):
    params_path: Path = "E:/G/GraphMaskExplainer/src/constants/params.py"
