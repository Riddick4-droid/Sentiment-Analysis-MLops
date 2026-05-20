#in this file, we will use kagglehub api to download data
#and save it to data/raw directory

import os
from pathlib import Path
from kagglehub import KaggleHub
from src.exceptions.custom_exceptions import DataDownloadError, DataIngestionError
from src.utils.logger import setup_logger
from src.utils.tracer import trace

#setting up the logger for this module
logger = setup_logger("download_data")

@trace(log_args=True, log_return=False)
def download_data(
    dataset:str,
    file_name:str,
    save_path:Path,
    force_download:bool=False,
):
    """
    Download a file from a Kaggle dataset using KaggleHub.

    Args:
        dataset: Kaggle dataset identifier (e.g., 'username/dataset-name').
        file_name: Name of the file to download from the dataset.
        save_path: Directory where the downloaded file will be saved.
        force_download: If True, re-download even if file exists.

    Returns:
        Path to the downloaded file.

    Raises:
        DataDownloadError: If download fails or file not found in dataset.
    """
    try:
        logger.info(f"Starting download of {file_name} from Kaggle dataset {dataset}")

        # Ensure save directory exists
        save_path.mkdir(parents=True, exist_ok=True)
        destination = save_path / file_name

        # Check if file already exists
        if destination.exists() and not force_download:
            logger.info(f"File {destination} already exists. Skipping download.")
            return destination

        # Initialize KaggleHub and attempt download
        kh = KaggleHub()
        kh.dataset_download_file(dataset, file_name, str(destination))

        if not destination.exists():
            error_msg = f"Failed to download {file_name} from {dataset}"
            logger.error(error_msg)
            raise DataDownloadError(error_msg)

        logger.info(f"Successfully downloaded {file_name} to {destination}")
        return destination

    except Exception as e:
        error_msg = f"Error downloading data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise DataDownloadError(error_msg, original_exception=e)

if __name__ == "__main__":
    # Example usage
    dataset = "mdismielhossenabir/sentiment-analysis"
    file_name = "sentiment_data.csv"
    save_path = Path("data/raw")
    try:
        downloaded_file = download_data(dataset, file_name, save_path)
        print(f"Data downloaded to: {downloaded_file}")
    except DataDownloadError as e:
        print(f"Failed to download data: {str(e)}")