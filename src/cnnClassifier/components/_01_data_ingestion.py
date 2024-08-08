import os
import zipfile
# import gdown package to download from google drive
import gdown
from cnnClassifier.logging import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    """
    DataIngestion handles downloading and extracting the dataset for the machine learning pipeline.

    Attributes:
        config (DataIngestionConfig): Configuration for data ingestion process.
    """
    
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Downloads the dataset from the specified URL.

        Returns:
            str: Path to the downloaded file.
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            # Ensure the directory for downloading the dataset exists
            os.makedirs(self.config.root_dir, exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Construct the download URL for Google Drive
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Download the file using gdown
            gdown.download(prefix + file_id, zip_download_dir, quiet=False)
            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

            return str(zip_download_dir)

        except Exception as e:
            logger.error(f"Error downloading data: {e}")
            raise e
  
    def extract_zip_file(self):
        """
        Extracts the downloaded zip file into the specified directory.

        Returns:
            None
        """
        unzip_path = self.config.unzip_dir

        # Ensure the directory for unzipping the dataset exists
        os.makedirs(unzip_path, exist_ok=True)
        logger.info(f"Extracting zip file {self.config.local_data_file} into directory {unzip_path}")

        # Extract the contents of the zip file
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

        logger.info(f"Extracted zip file {self.config.local_data_file} into directory {unzip_path}")