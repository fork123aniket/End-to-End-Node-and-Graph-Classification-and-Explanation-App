from pyvis.network import Network

from torch_geometric.datasets import Planetoid
from torch_geometric.utils import k_hop_subgraph

import shutil

from src.utils.common import *
from src.components.data_ingestion import DataIngestionOps
from src.config.config import get_data_ingestion_config
from src.constants.config import *

params = read_yaml()
dataset = Planetoid(DataIngestion.root_dir.value, params.dataset_name)
data = dataset[0]

subset, edge_index, _, _ = k_hop_subgraph(
            5, 2, data.edge_index, relabel_nodes=False, num_nodes=data.x.size(0))

print(f'subset: {subset}\nedges: {edge_index}\n'
      f'edge_list: {list(zip(edge_index[0].tolist(), edge_index[1].tolist()))}')

net = Network()
net.add_nodes(subset.tolist())
net.add_edges(list(zip(edge_index[0].tolist(), edge_index[1].tolist())))
# net.show('graph.html', local=False, notebook=False)

# loaded_model = mlflow.pytorch.load_model(TrainModel.model_save_dir.value)
# print("Model loaded successfully")

# class Experiment:
#     def __init__(self):
#         self.data_ingestion_config = get_data_ingestion_config()
#         self.data_ingestion = DataIngestionOps(config=self.data_ingestion_config)
#
#     def _main(self):
#         print(self.data_ingestion.data.x.size(),
#               self.data_ingestion.data.edge_index.size())
#
# obj = Experiment()
# obj._main()


# shutil.copytree("E:/G/GraphMaskExplainer/experiments/1/623e60fe1ba241f9967361539a0d2092/artifacts/model",
#                 "E:/G/GraphMaskExplainer/artifacts/model/")
# print("Directory moved successfully")
# params = read_yaml()
# search_best_run_and_save_model(params.exp_name)

