from cnnClassifier.components._02_model_preparation import PrepareModel
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.logging import logger


STAGE_NAME = "Model Preparation"


class Model_Preparation_Pipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the ConfigurationManager to load configuration and parameter settings
        config = ConfigurationManager()
        
        # Retrieve the configuration settings for preparing the base model
        prepare_model_config = config.get_prepare_model_config()
        
        # Initialize the PrepareModel process with the configuration settings
        prepare_model = PrepareModel(config=prepare_model_config)
        
        # Build the model by adding custom layers and compiling it with specified parameters
        prepare_model.build_model()
        

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
        obj = Model_Preparation_Pipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
