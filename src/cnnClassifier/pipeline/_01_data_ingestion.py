from src.cnnClassifier.config.configuration import ConfigurationManager
from src.cnnClassifier.components._01_data_ingestion import DataIngestion
from src.cnnClassifier.logging import logger



STAGE_NAME = "Data Ingestion"

class Data_Ingestion_Pipeline:
    def __init__(self):
        pass

    def main(self):
        # Initialize the ConfigurationManager to load configuration settings
        config = ConfigurationManager()
        
        # Retrieve the data ingestion configuration settings
        data_ingestion_config = config.get_data_ingestion_config()
        
        # Initialize the DataIngestion process with the configuration settings
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        # Download the dataset file from the specified source URL
        data_ingestion.download_file()
        
        # Extract the downloaded zip file to the specified directory
        data_ingestion.extract_zip_file()


if __name__ == '__main__':
    try:
        logger.info(f">>>>>> {STAGE_NAME} stage has started <<<<<<")
        obj = Data_Ingestion_Pipeline()
        obj.main()
        logger.info(f">>>>>> {STAGE_NAME} stage has completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception details and raise it
        logger.exception(e)
        raise e

