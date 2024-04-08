from typing import Tuple

from pyvis.network import Network
from pandas import DataFrame
import pandas as pd
from itertools import chain

from torch_geometric.utils import k_hop_subgraph
from torch_geometric.contrib.explain import GraphMaskExplainer
from torch_geometric.explain import Explainer, Explanation

from torch import Tensor

from src.config.config import *
from src.constants.config import *
from src.components.data_ingestion import *
from src.utils.common import *


class PredictPipeline:
    def __init__(self, node_index: int, task: str = 'node'):
        self.node_index = node_index
        self.task = task
        self.data_ingestion_config = get_data_ingestion_config()
        self.data_ingestion = DataIngestionOps(self.data_ingestion_config, self.task)
        self.params = read_yaml()
        set_seeds()

    def load_model(self):
        if self.task == 'node':
            self.model = mlflow.pytorch.load_model(TrainModel.model_save_dir.value)
        else:
            self.model = mlflow.pytorch.load_model(TrainModel.graph_model_save_dir.value)

    def compute_node_subgraph(self) -> Tuple[Tensor, Tensor]:
        subset, edge_index, _, _ = k_hop_subgraph(
            self.node_index, self.params.num_hops, self.data_ingestion.data.edge_index,
            num_nodes=self.data_ingestion.data.x.size(0)
        )

        return subset, edge_index

    def visualize_original_graph(self):
        enzyme_graph = Network(bgcolor='#222222', font_color='white', cdn_resources='remote')
        enzyme_graph.add_nodes(
            [node_id for node_id in range(self.data_ingestion.dataset[self.node_index].x.size(0))],
            label=[str(node_label) for node_label in range(self.data_ingestion.dataset[self.node_index].x.size(0))]
        )
        enzyme_graph.add_edges(list(
            zip(self.data_ingestion.dataset[self.node_index].edge_index[0].tolist(),
                self.data_ingestion.dataset[self.node_index].edge_index[1].tolist())
        ))

        enzyme_graph.write_html(GraphPath.original_graph.value, local=False)

    def visualize_node_subgraph(self):
        subset, edge_index = self.compute_node_subgraph()

        # print(f'subset: {subset}\nedges: {edge_index}\n'
        #       f'edge_list: {list(zip(edge_index[0].tolist(), edge_index[1].tolist()))}')

        net = Network(
            bgcolor='#222222', font_color='white',
            cdn_resources='remote'
        )
        net.add_nodes(subset.tolist(), label=[str(node_id) for node_id in subset.tolist()],
                      color=['#32cd32' if node_id == self.node_index else '#00ffff' for node_id in subset.tolist()])
        net.add_edges(list(zip(edge_index[0].tolist(), edge_index[1].tolist())))
        net.write_html(GraphPath.node_subgraph.value, local=False)

    def index_to_class(self, index: int) -> str:
        dict_to_search = self.params.ind_to_cls if self.task == 'node' else self.params.graph_ind_to_cls
        for key, value in dict_to_search.items():
            if key == index:
                return value

    @staticmethod
    def prepare_feature_mask(explain: Explanation, top_k: int = 5) -> DataFrame:
        node_mask = explain.get("node_mask").sum(dim=0).numpy()
        feat_labels = range(explain.get("node_mask").size(1))
        df = pd.DataFrame({'score': node_mask}, index=feat_labels)
        df = df.sort_values('score', ascending=False).reset_index()
        columns = ['feature_id', 'score']
        df.columns = columns
        df = df.round(decimals=3)
        df = df.head(top_k)

        return df

    def visualize_explanation_subgraph(self, explain: Explanation) -> bool:
        edge_weight = explain.get("edge_mask")
        # print(f'original edge weights: {edge_weight}')
        edge_weight = (edge_weight - edge_weight.min()) / edge_weight.max()
        mask = edge_weight > 1e-7
        if self.task == 'node':
            edge_index = self.data_ingestion.data.edge_index[:, mask]
        else:
            edge_index = self.data_ingestion.dataset[self.node_index].edge_index[:, mask]
        edge_weight = edge_weight[mask]
        edge_weight = torch.where(edge_weight != 1, edge_weight, self.params.edge_width)
        unique_edge_weight = edge_weight.view(-1).unique().tolist()
        # print(f'unique edge_weight values: {edge_weight.view(-1).unique().tolist()}')

        if len(unique_edge_weight) == 1 and self.params.edge_width in unique_edge_weight:
            explanation_check = False
        else:
            explanation_check = True

        if self.task == 'graph':
            net = Network(
                bgcolor='#222222', font_color='white',
                select_menu=True, filter_menu=True,
                cdn_resources='remote'
            )
            # net.barnes_hut(spring_strength=0.006)
            unique_nodes = edge_index.view(-1).unique().tolist()
            # if self.task == 'node':
            #     net.add_nodes(unique_nodes, label=[str(node_id) for node_id in unique_nodes],
            #                   color=['#32cd32' if node_id == self.node_index else '#00ffff' for node_id in unique_nodes])
            # else:
            net.add_nodes(unique_nodes, label=[str(node_id) for node_id in unique_nodes])
            for src, dst, width in zip(edge_index[0].tolist(), edge_index[1].tolist(),
                                       edge_weight.tolist()):
                if width == self.params.edge_width:
                    # print(f'tuple: {src, dst, width}')
                    # if not explanation_check:
                    #     net.add_edge(src, dst, value=width)
                    # else:
                    net.add_edge(src, dst, value=width, color='orange')
                else:
                    if width == torch.max(edge_weight):
                        net.add_edge(src, dst, value=width, color='orange')
                    else:
                        net.add_edge(src, dst, value=width)
        # net.add_edges(list(
        #     zip(edge_index[0].tolist(), edge_index[1].tolist(),
        #         edge_weight.tolist())
        # ))

        # net.write_html('explained_graph.html', local=False)

        if explanation_check:
            explained_edge_list = list(
                zip(edge_index[0].tolist(), edge_index[1].tolist(),
                    edge_weight.tolist()
                    )
            )
            # print(f'expl_edge_list: {explained_edge_list}')
            smaller_net = Network(
                bgcolor='#222222', font_color='white',
                cdn_resources='remote'
            )

            node_list = []
            # if self.task == 'node':
            #     for edge_tuple in explained_edge_list:
            #         if self.params.edge_width in
            # explained_edge_list = explained_edge_list if len(explained_edge_list) <= 10 else explained_edge_list[:10]

            for edge_tuple in explained_edge_list:
                if (self.params.edge_width in edge_tuple or
                        self.params.edge_width not in unique_edge_weight and
                        torch.max(edge_weight) in edge_tuple):
                    # print(f'edge_tuple: {edge_tuple, type(edge_tuple[0])}')
                    node_list.append([edge_tuple[0], edge_tuple[1]])
                    # smaller_net.add_edge(edge_tuple[0], edge_tuple[1], color='orange')

            # print(f'node_list before pruning: {node_list}')
            if self.task == 'node':
                node_list = node_list if len(node_list) <= 10 else node_list[:10]
            unique_node_list = list(set(list(chain.from_iterable(node_list))))
            # print(f'unique_node_list: {unique_node_list}')
            # if self.task == 'node':
            #     smaller_net.add_nodes(unique_node_list, label=[str(node_id) for node_id in unique_node_list],
            #                           color=['#32cd32' if node_id == self.node_index else '#00ffff' for node_id in
            #                                  unique_node_list])
            # else:
            smaller_net.add_nodes(unique_node_list, label=[str(node_id) for node_id in unique_node_list])

            # print(f'node_list: {node_list}')

            for edge_list in node_list:
                smaller_net.add_edge(edge_list[0], edge_list[1], color='orange')

        if self.task == 'node':
            # net.write_html(GraphPath.node_net_graph.value, local=False)
            if explanation_check:
                smaller_net.write_html(GraphPath.node_extract_graph.value, local=False)
        else:
            net.write_html(GraphPath.graph_net_graph.value, local=False)
            if explanation_check:
                smaller_net.write_html(GraphPath.graph_extract_graph.value, local=False)

        return explanation_check

    def train_and_explain(self) -> Tuple[Explanation, int]:
        if self.task == 'node':
            explainer = Explainer(
                model=self.model,
                algorithm=GraphMaskExplainer(params.num_hops, params.num_epochs),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='node',
                    return_type='log_probs',
                ),
            )

            explanation = explainer(
                self.data_ingestion.data.x, self.data_ingestion.data.edge_index,
                index=self.node_index
            )
        else:
            explainer = Explainer(
                model=self.model,
                algorithm=GraphMaskExplainer(params.graph_num_hops, params.num_epochs),
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='graph',
                    return_type='log_probs',
                ),
            )

            batch = torch.zeros(
                self.data_ingestion.dataset[self.node_index].x.shape[0], dtype=int
            )
            explanation = explainer(
                self.data_ingestion.dataset[self.node_index].x,
                self.data_ingestion.dataset[self.node_index].edge_index,
                batch=batch, index=self.node_index
            )

        return explanation, self.params.top_k

    def predict(self) -> str:
        self.model.eval()

        if self.task == 'node':
            # subset, edge_index = self.compute_node_subgraph()
            # x = self.data_ingestion.data.x[subset]
            prediction = self.model(self.data_ingestion.data.x, self.data_ingestion.data.edge_index)
            # true_index = subset.tolist().index(self.node_index)
            predicted_class_index = prediction[self.node_index].argmax(dim=-1).item()
        else:
            batch = torch.zeros(
                self.data_ingestion.dataset[self.node_index].x.shape[0], dtype=int
            )
            prediction = self.model(
                self.data_ingestion.dataset[self.node_index].x,
                self.data_ingestion.dataset[self.node_index].edge_index,
                batch=batch
            )
            predicted_class_index = prediction.argmax(dim=-1).item()
            # print(f'prediction: {prediction}\nindex: {predicted_class_index}')

        predicted_class = self.index_to_class(predicted_class_index)

        return predicted_class
