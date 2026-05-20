#this loads the configuration yaml file for the entire pipeline
import yaml
from pathlib import Path
from typing import Dict, Any

from src.exceptions.custom_exceptions import ConfigurationError
from src.utils.logger import setup_logger, log_exception
from src.utils.tracer import trace


logger = setup_logger("config_loader")

@trace
def load_config(config_path:str="config.yaml")->Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the config.yaml file.

    Returns:
        Dictionary with configuration parameters.

    Raises:
        ConfigurationError: If file not found, invalid YAML, or missing required keys.
    """
    path = Path(config_path)

    if not path.exists():
        error_msg = f"configuration file not found: {config_path}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        error_msg = f"Invalid YAML in {config_path}"
        log_exception(logger, e, error_msg)
        raise ConfigurationError(error_msg,e)
    except Exception as e:
        error_msg = f"failed to read config file {config_path}"
        log_exception(logger, e, error_msg)
        raise ConfigurationError(error_msg, e)
    
    required_keys = ["data","vectorizer","training","models","mlflow"]
    missing_keys = [key for key in required_keys if key not in config]

    if missing_keys:
        error_msg = f"missing required top-level keys in config: {missing_keys}"
        logger.error(error_msg)
        raise ConfigurationError(error_msg,e)
    
    logger.info(f"configuration loaded successfully from {config_path}")
    logger.debug(f"config keys: {list(config.keys())}")
    return config

# Example usage (commented out)
if __name__ == "__main__":
    cfg = load_config("config.yaml")
    print(cfg["active_model"])
