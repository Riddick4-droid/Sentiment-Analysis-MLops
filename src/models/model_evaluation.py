"""
Model evaluation module: Compute metrics, load models, compare performance.
"""
import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from src.exceptions.custom_exceptions import ModelEvaluationError
from src.utils.logger import setup_logger, log_exception
from src.utils.tracer import trace

logger = setup_logger("model_evaluation")

@trace
def compute_metrics(
    y_true:np.ndarray,
    y_pred:np.ndarray,
    labels: Optional[List]=None,
) -> Dict[str,Any]:
    """
    Compute standard classification metrics.

    Args:
        y_true: Ground truth labels (encoded integers).
        y_pred: Predicted labels.
        labels: Optional list of class labels (for multiclass).

    Returns:
        Dictionary with keys: accuracy, f1_weighted, precision_weighted,
        recall_weighted, confusion_matrix, classification_report (as dict).
    """
    try:
        metrics = {
            "accuracy": accuracy_score(y_true,y_pred),
            "f1_weighted": f1_score(y_true,y_pred,average="weighted"),
            "precision_weighted":precision_score(y_true,y_pred,average="weighted", zero_division=0),
            "recall_weighted":recall_score(y_true, y_pred,average="weighted", zero_division=0),
            "confusion_matrix":confusion_matrix(y_true,y_pred).tolist(),
            "classification_report":classification_report(y_true,y_pred,output_dict=True)
        }
        logger.info(f"Computed metrics: accuracy={metrics['accuracy']:.3f}, f1_weighted={metrics['f1_weighted']:.3f}")
        return metrics if isinstance(metrics,dict) else metrics.to_dict()
    except Exception as e:
        log_exception(logger, e, f"Failed to compute metrics")
        raise ModelEvaluationError("metric computation failed",e)
    
@trace
def load_local_model(
    model_pth: Union[str, Path]
):
    """Load a scikit-learn model from a .pkl file."""
    path = Path(model_pth)
    if not path.exists():
        raise ModelEvaluationError(f"Model file not found: {path}")
    try:
        model = joblib.load(path)
        logger.info(f"loaded local model from: {path}")
        return model
    except Exception as e:
        log_exception(logger, e, f"failed to load model from {path}")
        raise ModelEvaluationError(f"failed to load model from {path}",e)
    
@trace
def load_mlflow_model(model_uri: str):
    """
    Load a model from MLflow registry or run.

    Args:
        model_uri: e.g., "models:/ModelName/Production" or "runs:/<run_id>/model_path"

    Returns:
        Loaded model (sklearn compatible).
    """
    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info(f"loaded mlflow model from  {model_uri}")
        return model
    except Exception as e:
        log_exception(logger, e, f"failed to load model from mlflow on uri: {model_uri}")
        raise ModelEvaluationError(f"failed to load mlflow model from {model_uri}", e)
    
@trace
def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "unknown",
    log_to_mlflow: bool = False,
    mlflow_config: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    Evaluate a model on test data, optionally log results to MLflow.

    Args:
        model: Fitted model (predict method available).
        X_test: Test features.
        y_test: True labels.
        model_name: Name for logging.
        log_to_mlflow: Whether to log metrics to MLflow as a new run.
        mlflow_config: Required if log_to_mlflow True (tracking_uri, experiment_name).

    Returns:
        Metrics dictionary (as from compute_metrics).
    """
    try:
        logger.info(f"evaluating model {model_name} on test data of shape: {X_test.shape}")
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test,y_pred) #returns a dict

        #log to mlflow server
        if log_to_mlflow:
            if mlflow_config is None:
                mlflow_config = {}
            if mlflow_config.get("tracking_uri"):
                mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
            if mlflow_config.get("experiment_name"):
                mlflow.set_experiment(mlflow_config["experiment_name"])
            with mlflow.start_run(run_name=-f"evaluation_{model_name}"):
                mlflow.log_params({
                    "model_name":model_name,
                    "test_samples":len(y_test)
                })
                mlflow.log_metrics({
                    "accuracy": metrics["accuracy"],
                    "f1_weighted": metrics["f1_weighted"],
                    "precision_weighted": metrics["precision_weighted"],
                    "recall_weighted": metrics["recall_weighted"]
                })
                #log classification report as artifact
                report_df = pd.DataFrame(metrics["classification_report"]).transpose()
                report_path = f"eval_report_{model_name}.csv"
                report_df.to_csv(report_path)
                mlflow.log_artifact(report_path)
                logger.info(f"logged evaluation to mlflow for {model_name}")
        return metrics
    except Exception as e:
        log_exception(logger, e, f"evaluation failed for {model_name}")
        raise ModelEvaluationError(f"evaluation failed for {model_name}",e)
    
@trace
def compare_models(
    models_dict: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    log_to_mlflow: bool = False,
    mlflow_config: Optional[Dict] = None
)->pd.DataFrame:
    """
    Evaluate multiple models and return a comparison DataFrame.

    Args:
        models_dict: {model_name: model_instance} (model can be loaded or fitted).
        X_test, y_test: Test data.
        log_to_mlflow: If True, log each evaluation as a separate MLflow run.
        mlflow_config: Required if log_to_mlflow True.

    Returns:
        DataFrame with columns: model_name, accuracy, f1_weighted, precision_weighted, recall_weighted.
        """
    results = []
    for name, model in models_dict.items():
        logger.info(f"comparing_model: {name}")
        metrics = evaluate_model(
            model, X_test, y_test, log_to_mlflow=log_to_mlflow, mlflow_config=mlflow_config
        )
        results.append({
            "model_name": name,
            "accuracy": metrics["accuracy"],
            "f1_weighted": metrics["f1_weighted"],
            "precision_weighted": metrics["precision_weighted"],
            "recall_weighted": metrics["recall_weighted"]
        })

    #covert to dataframe
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)
    logger.info(f"model comparison completed:\n{df.to_string()}")
    return df

@trace
def evaluate_from_config(
    config: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_source: str = "local",  # 'local' or 'mlflow'
    model_name: Optional[str] = None
)->Dict[str, Any]:
    
    """
    Evaluate a model based on config.

    Args:
        config: Full config dict.
        X_test, y_test: Test data.
        model_source: 'local' to load from trained_models/<model_name>/model.pkl,
                      'mlflow' to load from MLflow registry (using model_name as registered model name).
        model_name: If None, uses config['active_model'].

    Returns:
        Metrics dictionary.
    """
    if model_name is None:
        model_name = config['active_model']
    
    if model_source == 'local':
        model_path = Path("trained_models") / model_name / f"{model_name}_model.pkl"
        model = load_local_model(model_path)
    elif model_source == 'mlflow':
        #assume registered model name is the same as model_name, stage production
        model_uri = f"models:/{model_name}/Production"
        model = load_mlflow_model(model_uri)
    else:
        raise ModelEvaluationError(f"invalid model_source provided: {model_source}")
    
    mlflow_cfg = config.get("mlflow", {})
    metrics = evaluate_model(
        model,X_test,y_test,model_name=model_name,log_to_mlflow=True, mlflow_config=mlflow_cfg
    )
    return metrics


# Example usage (commented out)
if __name__ == "__main__":
    # Dummy data
    from sklearn.ensemble import RandomForestClassifier
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)
    model = RandomForestClassifier().fit(X, y)
    metrics = evaluate_model(model, X, y, model_name="dummy")
    print(metrics.get('accuracy','None'))

    


