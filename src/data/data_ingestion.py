import pandas as pd
from pathlib import Path
from typing import Optional

from src.exceptions.custom_exceptions import DataIngestionError
from src.utils.logger import setup_logger
from src.utils.tracer import trace

#set up the logger for this module
logger = setup_logger("data_ingestion")

@trace(log_args=True, log_return=False)
def load_data(
    file_path:str,
    text_column:str,
    label_column:str,
    encoding:str="utf-8",
) -> pd.DataFrame:
    """
    Load raw sentiment data from a CSV file.

    Args:
        file_path: Path to the CSV file.
        text_column: Name of the column containing text (e.g., 'comment').
        label_column: Name of the column containing sentiment labels (e.g., 'sentiment').
        encoding: File encoding (default utf-8).

    Returns:
        DataFrame with raw data.

    Raises:
        DataIngestionError: If file missing, empty, columns missing, or read fails.
    """
    path = Path(file_path)

    #check if the file exists
    if not path.exists():
        error_msg = f"Data file not found at {file_path}"
        logger.error(error_msg)
        raise DataIngestionError(error_msg)
    #check if the file is empty
    if path.stat().st_size == 0:
        error_msg = f"Data file at {file_path} is empty"
        logger.error(error_msg)
        raise DataIngestionError(error_msg)
    
    #attempt to read the file
    #but we will handle the extension of the file thus, either .csv or .xlsx
    try:
        logger.info(f"Loading data from |{file_path}| with encoding |{encoding}|")

        #handle .csv files
        if file_path.endswith(".csv"):
            logger.debug(f"Detected .csv file format for {file_path}")
            df = pd.read_csv(file_path, encoding=encoding)
            logger.info(f"Successfully loaded data from {file_path} with shape {df.shape}| total instances: {len(df)}")

        #handle .xlsx files
        elif file_path.endswith(".xlsx"):
            logger.debug(f"Detected .xlsx file format for {file_path}")
            df = pd.read_excel(file_path)
            logger.info(f"Successfully loaded data from {file_path} with shape {df.shape}| total instances: {len(df)}")
        else:
            error_msg = f"Unsupported file format for {file_path}"
            logger.error(error_msg)
            raise DataIngestionError(error_msg)
    except (Exception, UnicodeDecodeError) as e:
        error_msg = f"Error reading data from {file_path}: {str(e)}"
        logger.error(error_msg)
        raise DataIngestionError(error_msg, original_exception=e)

    #validate required columns
    missing_columns = []
    if text_column not in df.columns:
        missing_columns.append(text_column)
    if label_column not in df.columns:
        missing_columns.append(label_column)

    if missing_columns:
        error_msg = f"Missing required columns in data file {file_path}: {', '.join(missing_columns)}"
        logger.error(error_msg)
        raise DataIngestionError(error_msg)
    if df.empty:
        error_msg = f"Data file {file_path} contains no data after loading"
        logger.error(error_msg)
        raise DataIngestionError(error_msg)
    
    #log basic statistics about the loaded data
    total_rows = len(df)
    missing_text = df[text_column].isnull().sum()
    missing_label = df[label_column].isnull().sum()

    logger.info(
        f"Data summary: total_rows={total_rows}, "
        f"missing_text={missing_text} ({missing_text/total_rows:.2%}), "
        f"missing_label={missing_label} ({missing_label/total_rows:.2%})"
    )

    #log unique values in the label column
    unique_labels = df[label_column].unique()
    logger.debug(f"Unique label values: {unique_labels}")

    #if you know more about investigating the data, you can add it, like visualization of the distribution of labels, 
    # or the length of the text, etc. but for now we will just return the dataframe

    return df

@trace
def get_data_stats(df:pd.DataFrame, text_column:str, label_column:str) -> dict:
    """
    Get basic statistics about the loaded data.

    Args:
        df: DataFrame containing the data.
        text_column: Name of the text column.
        label_column: Name of the label column.

    Returns:
        Dictionary with statistics.
    """
    total_rows = len(df)
    missing_text = df[text_column].isnull().sum()
    missing_label = df[label_column].isnull().sum()
    unique_labels = df[label_column].unique()
    label_distribution = df[label_column].value_counts(normalize=True)

    stats = {
        "total_rows": total_rows,
        "missing_text": missing_text,
        "missing_label": missing_label,
        "unique_labels": unique_labels.tolist() if isinstance(unique_labels, pd.Series) else unique_labels,
        "label_distribution": label_distribution.to_dict() if isinstance(label_distribution, pd.Series) else label_distribution
    }
    return stats

if __name__ == "__main__":
    #example usage
    try:
        df = load_data("data/raw/sentiment_data.csv", text_column="comment", label_column="sentiment")
        stats = get_data_stats(df, text_column="comment", label_column="sentiment")
        print(stats)
    except DataIngestionError as e:
        logger.error(f"Data ingestion failed: {str(e)}")