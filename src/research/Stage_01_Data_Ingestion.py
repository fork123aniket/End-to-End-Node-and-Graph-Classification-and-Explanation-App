from src.constants.config import *
from src.constants.params import *
from src.log.logger import logger

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import warnings; warnings.filterwarnings("ignore")

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path


def get_data_ingestion_config() -> DataIngestionConfig:
    data_ingestion_config = DataIngestionConfig(
        root_dir=DataIngestion.root_dir.value
    )

    return data_ingestion_config


class DataIngestionOps:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def read_data(self, verbose: bool = True):
        self.dataset = self.data_preprocess_and_split(self.config.root_dir, apply_data_transform)

        logger.info(">>>>>> Dataset Successfully Moved to the Target Directory<<<<<<")

        self.data = self.dataset[0]
        if verbose:
            logger.info(
                f"node_feat_matrix_shape: {self.data.x.size()}, edge_index_shape: {self.data.edge_index.size()}, \n"
                f"total_num_labels: {self.data.y.size()}, num_edge_features: {self.data.num_edge_features}, \n"
                f"num_edges: {self.data.num_edges}, num_node_features: {self.data.num_node_features}, \n"
                f"num_nodes: {self.data.num_nodes}, num_classes: {self.dataset.num_classes}")
            self.data_analysis()

    @staticmethod
    def data_preprocess_and_split(path: Path, data_transform: bool = False) -> Planetoid:
        if data_transform and data_split != "random":
            transformation = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
            dataset = Planetoid(path, dataset_name, transform=transformation)
            return dataset
        elif not data_transform and data_split == "random":
            dataset = Planetoid(path, dataset_name, data_split, num_train_samples, num_val_samples,
                                num_test_samples)
            return dataset
        elif data_transform and data_split == "random":
            transformation = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
            dataset = Planetoid(
                path, dataset_name, data_split, num_train_samples, num_val_samples,
                num_test_samples, transform=transformation)
            return dataset
        else:
            dataset = Planetoid(path, dataset_name)
            return dataset

    def data_analysis(self):
        type_data, indices = ('train', 'val', 'test'), (
            sum(self.data.train_mask).item(), sum(self.data.val_mask).item(), sum(self.data.test_mask).item())
        plt.figure(figsize=(10, 3))
        ax = sns.barplot(x=list(type_data), y=list(indices))
        ax.set_xticklabels(type_data, rotation=0, fontsize=12)
        plt.title("Mask distribution", fontsize=16)
        plt.ylabel("Type of Data", fontsize=14)
        plt.ylabel("# of indices", fontsize=14)
        plt.show()


if __name__ == '__main__':
    try:
        data_ingestion_config = get_data_ingestion_config()
        data_ingestion = DataIngestionOps(config=data_ingestion_config)
        data_ingestion.read_data()
    except Exception as e:
        raise e
