from enum import Enum
from pathlib import Path


class DataIngestion(Enum):
    root_dir: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/artifacts/data/Cora"
    graph_root_dir: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/artifacts/data/ENZYMES"


class ParamConfig(Enum):
    path_to_yaml: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/params.yaml"


class LoggingStack(Enum):
    base_dir: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/logs"


class TrainModel(Enum):
    model_save_dir: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/artifacts/model/node/"
    graph_model_save_dir: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/artifacts/model/graph/"
    model_registry_path: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/experiments/Cora"
    graph_model_registry_path: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/experiments/ENZYMES"


class GraphPath(Enum):
    node_net_graph: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/htmlfiles/explained_graph.html"
    node_subgraph: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/htmlfiles/graph.html"
    node_extract_graph: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/htmlfiles/small_explained_graph.html"
    graph_net_graph: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/htmlfiles/enzyme_explained_graph.html"
    original_graph: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/htmlfiles/enzyme_original_graph.html"
    graph_extract_graph: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/htmlfiles/enzyme_small_explained_graph.html"


class TuningPath(Enum):
    params_path: Path = "/mount/src/end-to-end-node-and-graph-classification-and-explanation-app/src/constants/params.py"
