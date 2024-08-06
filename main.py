from cnnClassifier.logging import logger
from cnnClassifier.pipeline._01_data_ingestion import Data_Ingestion_Pipeline
from cnnClassifier.pipeline._02_model_preparation import Model_Preparation_Pipeline
from cnnClassifier.pipeline._03_model_trainer import Model_Training_Pipeline
from cnnClassifier.pipeline._04_model_evaluation import Model_Evaluation_Pipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = Data_Ingestion_Pipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e


STAGE_NAME = "Model Preparation"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = Model_Preparation_Pipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e


STAGE_NAME = "Model Training"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = Model_Training_Pipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e


STAGE_NAME = "Model Evaluation"
try:
    logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
    obj = Model_Evaluation_Pipeline()
    obj.main()
    logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
except Exception as e:
    # Log the exception details and raise it
    logger.exception(e)
    raise e
