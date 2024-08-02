from cnnClassifier.logging import logger
from cnnClassifier.pipeline._01_data_ingestion import DataIngestion_TrainingPipeline
from cnnClassifier.pipeline._02_model_preparation import PrepareModel_TrainingPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = DataIngestion_TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e


STAGE_NAME = "Model Preparation"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = PrepareModel_TrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e

