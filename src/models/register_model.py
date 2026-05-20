"""
Model evaluation module: Compute metrics, load models, compare performance.
"""

import mlflow
import mlflow.sklearn
import joblib
import numpy as np
from mlflow.tracking import MlflowClient
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List


from src.exceptions.custom_exceptions import ModelRegistrationError
from src.utils.logger import setup_logger, log_exception
from src.utils.tracer import trace

logger = setup_logger("register_model")

@trace
def get_mlflow_client(tracking_uri: Optional[str]=None)->MlflowClient:
    """
    Get an MLflow client, optionally setting the tracking uri
    Returns MLflow client instance
    """
    try:
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
            client = MlflowClient()
            logger.debug(f"MlFlow client created. Tracking URI: {mlflow.get_tracking_uri()}")
        return client
    except Exception as e:
        log_exception(logger, e, "Failed to create Mlflow client")
        raise ModelRegistrationError(f"Failed to create MLFlow client", e)
    
@trace
def register_model_version(
    run_id:str,
    model_pth: str,
    registered_model_name:str,
    tracking_uri: Optional[str]=None,
    description: Optional[str]=None,
) -> Dict[str,Any]:
    """
    Register a model from an MLflow run to the Model Registry.

    Args:
        run_id: MLflow run ID containing the logged model.
        model_path: Path within the run's artifacts (usually the model name, e.g., 'logistic_regression').
        registered_model_name: Name to register under in the registry.
        tracking_uri: Optional tracking URI override.
        description: Optional description for this version.

    Returns:
        Dictionary with version details: {'name': ..., 'version': ..., 'stage': ...}
    """
    try:
        client = get_mlflow_client(tracking_uri)
        model_uri = f"runs:/{run_id}/{model_pth}"

        #registering the model
        logger.info(f"Registering the model: {model_uri}....")
        version = mlflow.register_model(model_uri,registered_model_name)
        logger.info(f"Registered model: {registered_model_name}: version: {version} from run: {run_id}")

        if description:
            client.update_model_version(
                name=registered_model_name,
                version=version.version,
                description=description
            )
            logger.debug(f"Added description to version: {version.version}")

            return {
                "name": registered_model_name,
                "version": version.version,
                "stage":version.stage,
                "run_id":run_id
            }
    except Exception as e:
        log_exception(logger, e, f"failed to register model {registered_model_name}")
        raise ModelRegistrationError(f"failed to register model {registered_model_name}")

    
    
        



