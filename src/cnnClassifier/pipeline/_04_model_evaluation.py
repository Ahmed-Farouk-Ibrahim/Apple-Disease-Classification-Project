from cnnClassifier.components._04_model_evaluation import Evaluation
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.logging import logger


STAGE_NAME = "Model Evaluation"

class Model_Evaluation_Pipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the configuration manager to load config and params files
        config = ConfigurationManager()       
        
        # Get the evaluation configuration from the configuration manager
        eval_config = config.get_evaluation_config()
        
        # Initialize the Evaluation class with the evaluation configuration
        evaluation = Evaluation(eval_config)
        
        # Prepare the training and validation data generators
        evaluation.test_generator()
        
        # Load the built model
        evaluation.load_model()
          
        # evaluate the model
        evaluation.evaluation()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
        obj = Model_Evaluation_Pipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
