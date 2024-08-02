from src.cnnClassifier.logging import logger
from src.cnnClassifier.pipeline._01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e

    