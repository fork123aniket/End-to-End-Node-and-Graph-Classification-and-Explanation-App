from src.components.tuning_model import Tuning
from src.config.config import get_tuning_config
from src.log.logger import logger


class TuningPipeline:
    def __init__(self, task: str = 'node'):
        self.task = task

    def main(self):
        tune_config = get_tuning_config()
        tuning_config = Tuning(tune_config, self.task)
        tuning_config.get_data()
        tuning_config.start_tuning_process()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage: Tuning Hyperparameters started <<<<<<")
        obj = TuningPipeline('graph')
        obj.main()
        logger.info(f">>>>>> stage: Tuning Hyperparameters completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
