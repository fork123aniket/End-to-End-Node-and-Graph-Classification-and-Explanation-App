from dataclasses import dataclass
from pathlib import Path
import os
import shutil
from urllib.parse import urlparse
from tqdm import tqdm
import mlflow

from src.constants.config import *
from src.constants.params import *
from src.models.GNN import *
from src.config.config import *
from src.components.data_ingestion import *
from src.utils.common import set_seeds

os.environ["GIT_PYTHON_REFRESH"] = "quiet"


@dataclass(frozen=True)
class TrainingConfig:
    model_registry_path: Path
    num_epochs: int
    weight_decay_ratio: float
    learning_rate: float
    num_features: int
    num_classes: int
    num_hidden_features: int


def get_training_config() -> TrainingConfig:
    training_config = TrainingConfig(
        model_registry_path=TrainModel.model_registry_path.value,
        num_epochs=num_epochs,
        weight_decay_ratio=weight_decay_ratio,
        learning_rate=learning_rate,
        num_features=num_features,
        num_classes=num_classes,
        num_hidden_features=num_hidden_features
    )

    return training_config


def search_best_run_and_save_model():
    sorted_runs = mlflow.search_runs(
        experiment_names=[exp_name], order_by=["metrics.test_accuracy DESC"],
        search_all_experiments=True
    )

    logger.info(f'sorted_runs: {sorted_runs}')

    artifact_dir = urlparse(mlflow.get_run(str(sorted_runs['run_id'][0])).info.artifact_uri).path
    path = Path(artifact_dir[1:], 'model/data/model.pth')

    shutil.copy(path, TrainModel.model_save_dir.value + 'model.pth')
    logger.info(">>>>>> Best Model Saved Successfully <<<<<<")


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_ingestion_config = get_data_ingestion_config()
        self.data_ingestion = DataIngestionOps(config=self.data_ingestion_config)
        set_seeds()

    def set_model_tracker(self):
        self.MLFLOW_TRACKING_URI = "file:///" + self.config.model_registry_path
        mlflow.set_tracking_uri(self.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(exp_name[1])

    def get_base_model_and_data(self):
        self.model = GCN(
            self.config.num_features, self.config.num_classes,
            self.config.num_hidden_features
        )
        self.data_ingestion.read_data(False)

    def evaluate_model(self):
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

    def train_model(self):
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


if __name__ == '__main__':
    try:
        for _ in tqdm(range(n_runs)):
            train_config = get_training_config()
            training_config = Training(config=train_config)
            training_config.set_model_tracker()
            training_config.get_base_model_and_data()
            training_config.train_model()
        search_best_run_and_save_model()
    except Exception as e:
        raise e
