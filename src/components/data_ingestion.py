from src.entity.config_entity import *

from pathlib import Path
from src.log.logger import logger

import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
import warnings; warnings.filterwarnings("ignore")

from torch_geometric.datasets import Planetoid, TUDataset
import torch_geometric.datasets as Dataset
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T


class DataIngestionOps:
    def __init__(self, config: DataIngestionConfig, task: str = 'node'):
        self.config = config
        self.task = task
        if self.task == 'node':
            self.dataset = self.data_preprocess_and_split(
                self.config.root_dir,
                self.config.apply_data_transform
            )
            self.data = self.dataset[0]
        else:
            self.dataset = self.data_preprocess_and_split(
                self.config.graph_root_dir
            )
            train_graph_list, val_graph_list, test_graph_list = self.dataset[:480], \
                self.dataset[480:540], self.dataset[540:600]
            self.train_graphs = DataLoader(train_graph_list, batch_size=self.config.batch_size, shuffle=True)
            self.val_graphs = DataLoader(val_graph_list, batch_size=self.config.batch_size)
            self.test_graphs = DataLoader(test_graph_list, batch_size=self.config.batch_size)

    def read_data(self, verbose: bool = True):
        if self.task == 'node':
            if verbose:
                logger.info(
                    f"node_feat_matrix_shape: {self.data.x.size()}, edge_index_shape: {self.data.edge_index.size()}, \n"
                    f"total_num_labels: {self.data.y.size()}, num_edge_features: {self.data.num_edge_features}, \n"
                    f"num_edges: {self.data.num_edges}, num_node_features: {self.data.num_node_features}, \n"
                    f"num_nodes: {self.data.num_nodes}, num_classes: {self.dataset.num_classes}")
                self.data_analysis()
        else:
            if verbose:
                logger.info(f'Dataset: {self.dataset}, Number of graphs: {len(self.dataset)}, \n'
                            f'Number of features: {self.dataset.num_features}, '
                            f'Number of classes: {self.dataset.num_classes}, \n'
                            f'num_edge_features: {self.dataset.num_edge_features}'
                            f'num_node_features: {self.dataset.num_node_features}')
                self.data_analysis()

    def data_preprocess_and_split(self, path: Path, data_transform: bool = False) -> Planetoid:
        if self.task == 'node':
            if data_transform and self.config.data_split != "random":
                transformation = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
                dataset = Planetoid(path, self.config.dataset_name, transform=transformation)
                return dataset
            elif not data_transform and self.config.data_split == "random":
                dataset = Planetoid(
                    path, self.config.dataset_name, self.config.data_split,
                    self.config.num_train_samples, self.config.num_val_samples,
                    self.config.num_test_samples)
                return dataset
            elif data_transform and self.config.data_split == "random":
                transformation = T.Compose([T.GCNNorm(), T.NormalizeFeatures()])
                dataset = Planetoid(
                    path, self.config.dataset_name, self.config.data_split,
                    self.config.num_train_samples, self.config.num_val_samples,
                    self.config.num_test_samples, transform=transformation)
                return dataset
            else:
                dataset = Planetoid(path, self.config.dataset_name)
                return dataset
        else:
            dataset = TUDataset(path, self.config.graph_dataset_name)
            return dataset

    def data_analysis(self):
        if self.task == 'node':
            type_data, indices = ('train', 'val', 'test'), (
                sum(self.data.train_mask).item(), sum(self.data.val_mask).item(), sum(self.data.test_mask).item())
            plt.figure(figsize=(10, 3))
            ax = sns.barplot(x=list(type_data), y=list(indices))
            ax.set_xticklabels(type_data, rotation=0, fontsize=12)
            plt.title("Mask distribution", fontsize=16)
            plt.ylabel("Type of Data", fontsize=14)
            plt.ylabel("# of indices", fontsize=14)
            plt.show()
        else:
            # for data in self.val_graphs:
            train_classes = []
            for data in self.train_graphs:
                train_classes.extend(data.y.tolist())
                # print(f'train_class: {train_classes}\nLength: {len(train_classes)}')
            val_class = [data.y.tolist() for data in self.val_graphs]
            test_class = [data.y.tolist() for data in self.test_graphs]
            fig, ax = plt.subplots(1, 3)
            ax[0].hist(train_classes)
            ax[0].set_title("Training Classes")
            ax[1].hist(val_class[0])
            ax[1].set_title("Validation Classes")
            ax[2].hist(test_class[0])
            ax[2].set_title("Test Classes")
            for axis in ax.flat:
                axis.set(xlabel='Class', ylabel='Count')
            fig.tight_layout()
            # plt.subplot(1, 3, 1)
            # plt.hist(train_classes)
            # plt.subplot(1, 3, 2)
            # plt.hist(val_class[0])
            # plt.subplot(1, 3, 3)
            # plt.hist(test_class[0])
            plt.show()
            # print(val_class, len(val_class[0]), test_class, len(test_class[0]), end='\n')
