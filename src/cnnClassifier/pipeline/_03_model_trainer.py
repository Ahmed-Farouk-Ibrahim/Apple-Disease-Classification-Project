from cnnClassifier.components._03_model_trainer import TrainModel
from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.logging import logger



STAGE_NAME = "Model Training"



class Model_Training_Pipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the configuration manager to load config and params files
        config = ConfigurationManager()
        
        # Get the training configuration from the configuration manager
        training_config = config.get_training_config()
        
        # Initialize the Training class with the training configuration
        training = TrainModel(config=training_config)
        
        # Load the built model
        training.get_built_model()
        
        # Prepare the training and validation data generators
        training.data_generator()

        training.calculate_class_weights()
        
        # Train the model
        training.train_model()

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
        obj = Model_Training_Pipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
        
