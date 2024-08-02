import os
from src.cnnClassifier.constants import *
from src.cnnClassifier.utils.common import read_yaml, create_directories
from src.cnnClassifier.entity.config_entity import (DataIngestionConfig, PrepareModelConfig, )

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
            params_image_size=self.params.IMAGE_SIZE,
            params_n_classes=self.params.N_CLASSES,
            params_learning_rate=self.params.LEARNING_RATE,
            params_rho= self.params.RHO,
            params_epsilon= self.params.EPSILON,
        )

        return prepare_model_config