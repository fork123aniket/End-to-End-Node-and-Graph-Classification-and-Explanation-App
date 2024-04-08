from torch_geometric.utils import k_hop_subgraph

from src.models.GNN import GCN
from src.config.config import *
from src.components.data_ingestion import *
from src.utils.common import *


class PredictPipeline:
    def __init__(self, node_index: int):
        self.node_index = node_index
        self.data_ingestion_config = get_data_ingestion_config()
        self.data_ingestion = DataIngestionOps(config=self.data_ingestion_config)
        self.params = read_yaml()
        set_seeds()

    def load_model_and_data(self):
        self.model = GCN(self.params.num_features, self.params.num_classes, self.params.num_hidden_features)
        self.model.load_state_dict(torch.load(TrainModel.model_save_path.value))

    def index_to_class(self, index: int) -> str:
        for key, value in self.params.ind_to_cls.items():
            if key == index:
                return value

    def predict(self) -> str:
        self.model.eval()

        subset, edge_index, _, _ = k_hop_subgraph(
            self.node_index, 2, self.data_ingestion.data.edge_index, self.data_ingestion.data.x.size(0))
        x = self.data_ingestion.data.x[subset]
        prediction = self.model(x, edge_index)
        true_index = subset.tolist().index(self.node_index)
        predicted_clas_index = prediction[true_index].argmax(dim=-1).item()
        predicted_class = self.index_to_class(predicted_clas_index)

        return predicted_class
