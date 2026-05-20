# in this file, we will use kagglehub api to download data
# and save it to data/raw directory

import os
import shutil
from pathlib import Path

import kagglehub

from src.exceptions.custom_exceptions import (
    DataDownloadError,
)

from src.utils.logger import setup_logger
from src.utils.tracer import trace


# setting up logger
logger = setup_logger("download_data")


@trace(log_args=True, log_return=False)
def download_data(
    dataset: str,
    save_path: Path,
    force_download: bool = False,
):
    """
    Download dataset from KaggleHub,
    automatically detect CSV file,
    and save it into data/raw.

    Args:
        dataset: Kaggle dataset identifier
        save_path: destination directory
        force_download: force re-download

    Returns:
        Path to saved CSV file
    """

    try:

        logger.info(
            f"Starting dataset download from '{dataset}'"
        )

        # create destination directory
        save_path.mkdir(
            parents=True,
            exist_ok=True
        )

        # download dataset
        dataset_path = kagglehub.dataset_download(
            dataset,
            force_download=force_download
        )

        logger.info(
            f"Dataset downloaded to: {dataset_path}"
        )

        # list all files in dataset directory
        files = os.listdir(dataset_path)

        logger.info(
            f"Files found in dataset: {files}"
        )

        # find csv file automatically
        csv_file = None

        for file in files:

            if file.endswith(".csv"):

                csv_file = file
                break

        # if no csv file found
        if csv_file is None:

            error_msg = (
                f"No CSV file found in dataset '{dataset}'"
            )

            logger.error(error_msg)

            raise DataDownloadError(error_msg)

        logger.info(
            f"CSV file detected: {csv_file}"
        )

        # build full source path
        source_file_path = os.path.join(
            dataset_path,
            csv_file
        )

        # build destination path
        destination_file_path = save_path / csv_file

        # copy file into data/raw
        shutil.copy(
            source_file_path,
            destination_file_path
        )

        logger.info(
            f"CSV file saved to: {destination_file_path}"
        )

        return destination_file_path

    except Exception as e:

        error_msg = f"Error downloading data: {str(e)}"

        logger.error(
            error_msg,
            exc_info=True
        )

        raise DataDownloadError(
            error_msg,
            original_exception=e
        )


if __name__ == "__main__":

    dataset = "mdismielhossenabir/sentiment-analysis"

    save_path = Path("data/raw")

    try:

        downloaded_file = download_data(
            dataset=dataset,
            save_path=save_path,
        )

        print(
            f"Data downloaded to: {downloaded_file}"
        )

    except DataDownloadError as e:

        print(
            f"Failed to download data: {str(e)}"
        )