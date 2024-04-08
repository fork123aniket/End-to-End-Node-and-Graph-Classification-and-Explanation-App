import mlflow

from src.models.GNN import *
from src.models.GNN_Graph import *
from src.config.config import *
from src.components.data_ingestion import *
from src.utils.common import set_seeds


class Training:
    def __init__(self, config: TrainingConfig, task: str = 'node'):
        self.config = config
        self.task = task
        self.data_ingestion_config = get_data_ingestion_config()
        self.data_ingestion = DataIngestionOps(self.data_ingestion_config, self.task)
        set_seeds()

    def set_model_tracker(self):
        if self.task == 'node':
            self.MLFLOW_TRACKING_URI = "file:///" + self.config.model_registry_path
        else:
            self.MLFLOW_TRACKING_URI = "file:///" + self.config.graph_model_registry_path
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.config.exp_name)

    def get_base_model_and_data(self):
        if self.task == 'node':
            self.model = GCN(
                self.config.num_features, self.config.num_classes,
                self.config.num_hidden_features
            )
        else:
            self.model = Net(
                self.config.graph_num_features, self.config.graph_num_classes,
                self.config.num_hidden_1, self.config.num_hidden_2
            )
        self.data_ingestion.read_data(False)

    def evaluate_model(self):
        if self.task == 'node':
            self.model.eval()
            prediction = self.model(
                self.data_ingestion.data.x, self.data_ingestion.data.edge_index
            ).argmax(dim=-1)

            val_accuracy = int(
                (prediction[self.data_ingestion.data.val_mask] == self.data_ingestion.data.y[
                    self.data_ingestion.data.val_mask]
                 ).sum()) / int(self.data_ingestion.data.val_mask.sum())

            test_accuracy = int(
                (prediction[self.data_ingestion.data.test_mask] == self.data_ingestion.data.y[
                    self.data_ingestion.data.test_mask]
                 ).sum()) / int(self.data_ingestion.data.test_mask.sum())

            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)
        else:
            self.model.eval()

            val_correct = 0
            for data in self.data_ingestion.val_graphs:
                out = self.model(data.x, data.edge_index, data.batch)
                prediction = out.argmax(dim=1)
                val_correct += int((prediction == data.y).sum())
            val_accuracy = val_correct / len(self.data_ingestion.val_graphs.dataset)

            test_correct = 0
            for data in self.data_ingestion.test_graphs:
                out = self.model(data.x, data.edge_index, data.batch)
                prediction = out.argmax(dim=1)
                test_correct += int((prediction == data.y).sum())
            test_accuracy = test_correct / len(self.data_ingestion.test_graphs.dataset)

            mlflow.log_metric("val_accuracy", val_accuracy)
            mlflow.log_metric("test_accuracy", test_accuracy)

    def train_model(self):
        if self.task == 'node':
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay_ratio
            )

            with mlflow.start_run(run_name='gcn_mlflow_exp') as run:
                for epoch in range(1, self.config.num_epochs):
                    logger.info(f'Active Run ID: {run.info.run_uuid}, Epoch:{epoch}')

                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(self.data_ingestion.data.x, self.data_ingestion.data.edge_index)
                    loss = F.nll_loss(
                        out[self.data_ingestion.data.train_mask],
                        self.data_ingestion.data.y[self.data_ingestion.data.train_mask]
                    )
                    loss.backward()
                    optimizer.step()

                    mlflow.log_metric("train_loss", loss.float(), step=epoch)

                self.evaluate_model()

                mlflow.log_param("num_epochs", self.config.num_epochs)
                mlflow.log_param("learning_rate", self.config.learning_rate)
                mlflow.log_param("weight_decay_ratio", self.config.weight_decay_ratio)
                mlflow.pytorch.log_model(self.model, "model")
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.config.graph_learning_rate,
                weight_decay=self.config.graph_weight_decay_ratio
            )

            with mlflow.start_run(run_name='graph_net_mlflow_exp') as run:
                for epoch in range(1, self.config.num_epochs):
                    loss_all = 0.0
                    logger.info(f'Active Run ID: {run.info.run_uuid}, Epoch:{epoch}')
                    for data in self.data_ingestion.train_graphs:
                        self.model.train()
                        optimizer.zero_grad()
                        output = self.model(data.x, data.edge_index, data.batch)
                        loss = F.nll_loss(output, data.y)
                        loss.backward()
                        torch.nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
                        optimizer.step()
                        loss_all += loss.item() * data.num_graphs

                    mlflow.log_metric("train_loss", loss_all, step=epoch)

                self.evaluate_model()

                mlflow.log_param("num_epochs", self.config.num_epochs)
                mlflow.log_param("learning_rate", self.config.graph_learning_rate)
                mlflow.log_param("weight_decay_ratio", self.config.graph_weight_decay_ratio)
                mlflow.pytorch.log_model(self.model, "model")
