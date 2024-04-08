from tqdm import tqdm
import yaml
import optuna

from src.components.data_ingestion import *
from src.config.config import *
from src.models.GNN import *
from src.models.GNN_Graph import *
from src.utils.common import set_seeds
from src.log.logger import logger


class Tuning:
    def __init__(self, config: TuningConfig, task: str = 'node'):
        self.config = config
        self.task = task
        self.data_ingestion_config = get_data_ingestion_config()
        self.data_ingestion = DataIngestionOps(self.data_ingestion_config, self.task)
        set_seeds()

    def get_data(self):
        self.data_ingestion.read_data(False)

    def objective(self, trial: optuna.Trial):
        if self.task == 'node':
            loss = 0.0
            hidden_features = trial.suggest_int("num_hidden_features", 12, 16)
            learning_rate = trial.suggest_float("learning_rate", 0.001, 0.1, log=True)
            weight_decay_ratio = trial.suggest_float("weight_decay_ratio", 0.0001, 0.0005, log=True)

            self.model = GCN(
                self.config.num_features, self.config.num_classes,
                hidden_features
            )

            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate,
                weight_decay=weight_decay_ratio
            )

            for epoch in tqdm(range(self.config.num_epochs), desc=f"Running Trial: {trial.number}"):

                self.model.train()
                optimizer.zero_grad()
                out = self.model(self.data_ingestion.data.x, self.data_ingestion.data.edge_index)
                loss = F.nll_loss(
                    out[self.data_ingestion.data.train_mask],
                    self.data_ingestion.data.y[self.data_ingestion.data.train_mask]
                )

                trial.report(loss.float(), epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

                loss.backward()
                optimizer.step()

            return loss
        else:
            hidden_1 = trial.suggest_int("num_hidden_1", 5, 10)
            hidden_2 = trial.suggest_int("num_hidden_2", 5, 10)
            learning_rate = trial.suggest_float("graph_learning_rate", 0.001, 0.1, log=True)
            weight_decay_ratio = trial.suggest_float("graph_weight_decay_ratio", 0.0001, 0.0005, log=True)

            self.model = Net(
                self.config.graph_num_features, self.config.graph_num_classes,
                hidden_1, hidden_2
            )

            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=learning_rate,
                weight_decay=weight_decay_ratio
            )

            for epoch in tqdm(range(self.config.num_epochs), desc=f"Running Trial: {trial.number}"):
                loss_all = 0.0
                for data in self.data_ingestion.train_graphs:
                    self.model.train()
                    optimizer.zero_grad()
                    out = self.model(data.x, data.edge_index, data.batch)
                    loss = F.nll_loss(out, data.y)
                    loss_all += loss.item() * data.num_graphs

                    trial.report(loss_all, epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

                    loss.backward()
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), 2.0)
                    optimizer.step()

            return loss_all

    @staticmethod
    def tuning_log():
        optuna.logging.enable_propagation()
        optuna.logging.disable_default_handler()

    @staticmethod
    def save_params(best_params):
        with open(ParamConfig.path_to_yaml.value, "a") as f:
            for key, value in best_params.items():
                yaml.dump({key: value}, f)
                # if isinstance(value, int):
                #     f.write(f'{key}: {"int"} = {value}\n')
                # else:
                #     f.write(f'{key}: {"float"} = {value}\n')

        logger.info(f'>>>>>> params successfully saved <<<<<<')

    def start_tuning_process(self):
        self.tuning_log()
        study = optuna.create_study(
            direction=self.config.direction,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=self.config.n_startup_trials,
                n_warmup_steps=self.config.n_warmup_steps,
                interval_steps=self.config.interval_steps
            ),
        )
        study.optimize(self.objective, n_trials=self.config.n_trials)

        logger.info(f'best_params: {study.best_params}, best_trial: {study.best_trial}, \n'
                    f'best_value: {study.best_value}, system_attrs: {study.system_attrs}, \n'
                    f'trials: {study.trials}')

        self.save_params(study.best_params)
