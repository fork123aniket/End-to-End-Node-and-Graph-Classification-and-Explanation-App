from src.pipeline.data_ingestion_pipeline import *
from src.pipeline.tuning_pipeline import *
from src.pipeline.training_pipeline import *


try:
    logger.info(f">>>>>> stage: Data Ingestion started <<<<<<")
    obj = DataIngestionPipeline('graph')
    obj.main()
    logger.info(f">>>>>> stage: Data Ingestion completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


# try:
#     logger.info(f">>>>>> stage: Tuning Hyperparameters started <<<<<<")
#     obj = TuningPipeline('graph')
#     obj.main()
#     logger.info(f">>>>>> stage: Tuning Hyperparameters completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e
#
# try:
#     params = read_yaml()
#     logger.info(f">>>>>> stage: Tracking and Training Model started <<<<<<")
#     obj = TrainingPipeline(params, 'graph')
#     obj.main()
#     logger.info(f">>>>>> stage: Tracking and Training Model completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#    raise e
