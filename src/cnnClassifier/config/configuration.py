import os
from cnnClassifier.constants import *
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareModelConfig, TrainingConfig, EvaluationConfig)

class ConfigurationManager:
    """
    ConfigurationManager handles the setup and retrieval of configuration settings for the pipeline.
    """

    def __init__(self, 
                 config_filepath=CONFIG_FILE_PATH, 
                 params_filepath=PARAMS_FILE_PATH):
        # Load configuration settings from the specified config.yaml file
        self.config = read_yaml(config_filepath)
        
        # Load parameters from the specified params file
        self.params = read_yaml(params_filepath)

        # Create the root directory for artifacts as declared in config.yaml file
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Retrieves and prepares the data ingestion configuration settings.

        Returns:
            DataIngestionConfig: Data ingestion configuration settings.
        """
        # Extract data_ingestion settings from config.yaml file
        config_di = self.config.data_ingestion  

        # Create directory for data ingestion artifacts
        create_directories([config_di.root_dir])

        # Create a DataIngestionConfig object with the settings
        data_ingestion_config = DataIngestionConfig(
            root_dir=config_di.root_dir,
            source_URL=config_di.source_URL,
            local_data_file=config_di.local_data_file,
            unzip_dir=config_di.unzip_dir
        )
        return data_ingestion_config

    def get_prepare_model_config(self) -> PrepareModelConfig:
        """
        Retrieve the configuration for preparing the model.

        Returns:
            PrepareModelConfig: An object containing the configuration settings for preparing the model.
        """
        # Extract the model preparation configuration from the loaded config
        config = self.config.model_preparation

        # Create directories specified in the model configuration
        create_directories([config.root_dir])

        # Instantiate and return the PrepareModelConfig object with the necessary settings
        prepare_model_config = PrepareModelConfig(
            root_dir=Path(config.root_dir),
            built_model_path=Path(config.built_model_path),
            all_params=self.params, 
        )
        return prepare_model_config
    

    def get_training_config(self) -> TrainingConfig:
        """
        Retrieve the configuration for training the model.

        Returns:
            TrainingConfig: An object containing the configuration settings for training the model.
        """
        training = self.config.training
        model_preparation = self.config.model_preparation
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "dataset")
        create_directories([Path(training.root_dir)])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            built_model_path=Path(model_preparation.built_model_path),
            trained_model_path=Path(training.trained_model_path),
            training_data=Path(training_data),
            all_params=self.params, 
        )
        return training_config
    
    
    def get_evaluation_config(self) -> EvaluationConfig:
        training = self.config.training
        training_data = os.path.join(self.config.data_ingestion.unzip_dir, "dataset")
        create_directories([ Path(training.root_dir) ])


        eval_config = EvaluationConfig(
            root_dir=Path(training.root_dir),            
            trained_model_json_path=Path(training.trained_model_path).with_suffix(".json"), # = "artifacts/training/model.json"
            trained_model_weights_path=Path(training.trained_model_path).with_suffix(".h5"),
            training_data=Path(training_data),
            all_params=self.params,         
        )
        return eval_config
    
