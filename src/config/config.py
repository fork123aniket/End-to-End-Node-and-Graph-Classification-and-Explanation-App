from src.constants.config import *
from src.entity.config_entity import *
from src.utils.common import read_yaml


params = read_yaml()


def get_data_ingestion_config() -> DataIngestionConfig:
    data_ingestion_config = DataIngestionConfig(
        root_dir=DataIngestion.root_dir.value,
        graph_root_dir=DataIngestion.graph_root_dir.value,
        apply_data_transform=params.apply_data_transform,
        dataset_name=params.dataset_name,
        graph_dataset_name=params.graph_dataset_name,
        data_split=params.data_split,
        batch_size=params.batch_size,
        num_train_samples=params.num_train_samples,
        num_val_samples=params.num_val_samples,
        num_test_samples=params.num_test_samples
    )

    return data_ingestion_config


def get_tuning_config() -> TuningConfig:
    tuning_config = TuningConfig(
        num_epochs=params.num_epochs,
        num_features=params.num_features,
        num_classes=params.num_classes,
        graph_num_features=params.graph_num_features,
        graph_num_classes=params.graph_num_classes,
        n_startup_trials=params.n_startup_trials,
        n_warmup_steps=params.n_warmup_steps,
        interval_steps=params.interval_steps,
        n_trials=params.n_trials,
        direction=params.direction,
    )

    return tuning_config


def get_training_config() -> TrainingConfig:
    training_config = TrainingConfig(
        model_registry_path=TrainModel.model_registry_path.value,
        graph_model_registry_path=TrainModel.graph_model_registry_path.value,
        num_epochs=params.num_epochs,
        weight_decay_ratio=params.weight_decay_ratio,
        learning_rate=params.learning_rate,
        num_features=params.num_features,
        num_classes=params.num_classes,
        graph_num_features=params.graph_num_features,
        graph_num_classes=params.graph_num_classes,
        num_hidden_features=params.num_hidden_features,
        num_hidden_1=params.num_hidden_1,
        num_hidden_2=params.num_hidden_2,
        graph_learning_rate=params.graph_learning_rate,
        graph_weight_decay_ratio=params.graph_weight_decay_ratio,
        exp_name=params.exp_name
    )

    return training_config
