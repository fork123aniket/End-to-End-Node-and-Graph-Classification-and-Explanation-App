from tqdm import tqdm
from box import ConfigBox

from src.components.training_model import Training
from src.config.config import get_training_config
from src.utils.common import search_best_run_and_save_model, read_yaml
from src.log.logger import logger


class TrainingPipeline:
    def __init__(self, params: ConfigBox, task: str = 'node'):
        self.params = params
        self.task = task

    def main(self):
        for _ in tqdm(range(self.params.n_runs)):
            train_config = get_training_config()
            training_config = Training(train_config, self.task)
            training_config.set_model_tracker()
            training_config.get_base_model_and_data()
            training_config.train_model()
        search_best_run_and_save_model(self.params.exp_name, self.task)


if __name__ == '__main__':
    try:
        params = read_yaml()
        logger.info(f">>>>>> stage: Tracking and Training Model started <<<<<<")
        obj = TrainingPipeline(params, 'graph')
        obj.main()
        logger.info(f">>>>>> stage: Tracking and Training Model completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
