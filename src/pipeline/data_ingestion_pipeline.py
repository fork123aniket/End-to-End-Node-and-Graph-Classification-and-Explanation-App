from src.config.config import *
from src.components.data_ingestion import *
from src.log.logger import logger


class DataIngestionPipeline:
    def __init__(self, task: str = 'node'):
        self.task = task

    def main(self):
        data_ingestion_config = get_data_ingestion_config()
        data_ingestion = DataIngestionOps(data_ingestion_config, self.task)
        data_ingestion.read_data()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage: Data Ingestion started <<<<<<")
        obj = DataIngestionPipeline('graph')
        obj.main()
        logger.info(f">>>>>> stage: Data Ingestion completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
