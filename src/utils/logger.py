import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path("logs")

#make it a directory
LOG_DIR.mkdir(exist_ok=True)

#defualt log file paths
DEFAULT_LOG_FILE = LOG_DIR/"pipeline.log"
ERROR_LOG_FILE = LOG_DIR / "errors.log"

#setting up the log file
def setup_logger(
        name:str,
        log_file:Path=DEFAULT_LOG_FILE, 
        error_log_file: Path = ERROR_LOG_FILE,
        level: int = logging.DEBUG,
        console_level:int = logging.INFO,
        max_bytes: int = 10_485_760,
        backup_count: int=5,
)->logging.Logger:
    """
    Set up a logger with console and file handlers.

    Args:
        name: Logger name (usually __name__).
        log_file: Path to the main log file (rotating).
        error_log_file: Path to the error-only log file.
        level: Minimum level for the logger (e.g., logging.DEBUG).
        console_level: Level for console output (e.g., logging.INFO).
        max_bytes: Max size of log file before rotation.
        backup_count: Number of backup files to keep.

    Returns:
        Configured logging.Logger instance.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    #removing existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    #setting up formatter with timestamp, level, module, function, line, message
    detailed_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    simple_formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    )

    #console handler (stderr for warnings/errors, stdout for info)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(detailed_formatter)
    logger.addHandler(console_handler)

    # Main rotating file handler (all logs)
    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)

    # Separate error-only file handler (rotating)
    error_handler = RotatingFileHandler(
        error_log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    logger.addHandler(error_handler)

    return logger


def log_exception(logger: logging.Logger, e: Exception, message: str = None, exc_info: bool = True):
    """
    Log an exception with full traceback.

    Args:
        logger: Logger instance.
        e: Exception object.
        message: Optional custom message to prepend.
        exc_info: If True, includes traceback.
    """
    if message:
        logger.error(f"{message} - {str(e)}", exc_info=exc_info)
    else:
        logger.error(str(e), exc_info=exc_info)


# Example usage (commented out)
if __name__ == "__main__":
    test_logger = setup_logger("test_logger")
    test_logger.debug("Debug message")
    test_logger.info("Info message")
    test_logger.warning("Warning message")
    try:
        raise ValueError("Test error")
    except Exception as e:
        log_exception(test_logger, e, "Custom error message")