from ast import If

import mlflow
import mlflow.sklearn
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier

from src.exceptions.custom_exceptions import ModelTrainigError, ModelEvaluationError, MLflowRegistrationError
from src.utils.logger import setup_logger
from src.utils.tracer import trace

#set up the logger for this module
logger = setup_logger("model_building")

@trace
def get_n_classes(y: np.ndarray) -> int:
    """Determine the number of unique classes in the target variable."""
    n_classes = len(np.unique(y))
    logger.debug(f"Detected {n_classes} classes: {np.unique(y)}")
    return n_classes

@trace
def get_model_instance(model_name:str, model_params: Dict[str, Any], n_classes: Optional[int] = None) -> Any:
    """Factory function to create model instances based on name and parameters."""
    logger.debug(f"Instantiating model |{model_name}| with parameters: '{model_params}'| n_classes: '{n_classes}'")

    #this will handle binary classification and multiclass classification 
    if model_name == "logistic_regression":
        if n_classes is not None and n_classes > 2:
            #multiclass case
            model_params = {**model_params, "multi_class":"ovr"}
        return LogisticRegression(**model_params, random_state=42)
    elif model_name == "random_forest":
        return RandomForestClassifier(**model_params, random_state=42, n_jobs=-1)
    elif model_name == "xgboost":
        if n_classes == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        model_params = {**model_params, "objective": objective}

        if "eval_metric" in model_params and n_classes > 2:
            del model_params["eval_metric"]  #remove eval_metric
        return XGBClassifier(**model_params,random_state=42,use_label_encoder=False, n_jobs=-1)
    else:
        error_msg = f"Unsupported model name: {model_name}"
        logger.error(error_msg)
        raise ModelTrainigError(error_msg)
    
@trace
def train_and_evaluate(
    model:Any,
    X_train:np.ndarray,
    y_train:np.ndarray,
    X_test:np.ndarray,
    y_test:np.ndarray,
    model_name:str,
    run_name:str=None,
    mlflow_config:Dict[str, Any]=None,
    local_save_pth:Optional[Path]=None,
) -> Dict[str,float]:
    """
    Train model, evaluate, log to MLflow, and optionally save locally.

    Args:
        model: Unfitted model instance.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        model_name: Name used for logging and saving.
        run_name: MLflow run name (default: model_name).
        mlflow_config: Dict with 'tracking_uri', 'experiment_name', 'register_model' (bool).
        local_save_path: Directory to save model.pkl. If None, not saved.

    Returns:
        Dictionary with metrics: {'accuracy': ..., 'f1_weighted': ...}

    Raises:
        ModelTrainingError: If training fails.
        ModelEvaluationError: If evaluation fails.
    """
    if mlflow_config is None:
        mlflow_config = {}
    try:
        if mlflow_config.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_config["tracking_uri"])
        if mlflow_config.get("experiment_name"):
            mlflow.set_experiment(mlflow_config["experiment_name"])

        with mlflow.start_run(run_name=run_name or model_name) as run:
            #log model parameters(from model's get_params method)
            try:
                params = model.get_params()
                mlflow.log_params(params)
            except Exception as e:
                logger.warning(f"Failed to log model parameters for {model_name}: {e}")
            
            logger.info(f"Training model |{model_name}| on training data with shape {X_train.shape} with parameters: {params}")
            model.fit(X_train, y_train)
            logger.info(f"Model |{model_name}| training completed")

            #predict and evaluate
            logger.info(f"Evaluating model |{model_name}| on test data with shape {X_test.shape}")
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            logger.info(f"Model |{model_name}| evaluation completed with accuracy: {accuracy}, F1 (weighted): {f1_weighted}")

            #log metrics to mlflow
            mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1_weighted})
            logger.debug(f"Logged metrics to MLflow for model |{model_name}|: accuracy={accuracy:.3f}, f1_weighted={f1_weighted:.3f}")

            #log vlassification report as an mlflow artifact
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_csv = f"{model_name}_classification_report.csv"
            report_df.to_csv(report_csv, index=True)
            mlflow.log_artifact(report_csv, artifact_path="classification_reports")
            logger.debug(f"Logged classification report artifact to MLflow for model |{model_name}|: {report_csv}")

            #log the model to mlflow 
            mlflow.sklearn.log_model(model, artifact_path=f"{model_name}_model")
            logger.debug(f"Logged model artifact to MLflow for model |{model_name}|")

            #register the model in mlflow model registry if specified
            if mlflow_config.get("register_model",False):
                try:
                    registered_model_name = mlflow_config.get("registered_model_name", model_name)
                    mlflow.register_model(
                        model_uri=f"runs:/{run.info.run_id}/{model_name}_model",
                        name=registered_model_name
                    )
                    logger.info(f"Registered model |{model_name}| in MLflow Model Registry with name |{registered_model_name}|")
                except Exception as e:
                    logger.error(f"Failed to register model |{model_name}| in MLflow Model Registry: {e}")
                    raise MLflowRegistrationError(f"Failed to register model in MLflow Model Registry: {e}", original_exception=e)
            #optionally save the model locally as a .pkl file
            if local_save_pth:
                save_model_locally(model, model_name, local_save_pth)
    except Exception as e:
        raise ModelTrainigError(f"Error during training and evaluation of model |{model_name}|: {e}", original_exception=e)


@trace
def save_model_locally(model: Any, model_name: str, base_dir:Path = Path("trained_models")):
    """Save the trained model locally in base_dir/model_name/model_name_model.pkl"""
    try:
       model_dir = base_dir / model_name
       model_dir.mkdir(parents=True, exist_ok=True)
       model_path = model_dir / f"{model_name}_model.pkl"
       joblib.dump(model,model_path)
       logger.info(f"Saved model |{model_name}| locally at {model_path}")
    except Exception as e:
        logger.error(f"Failed to save model |{model_name}| locally: {e}")
        raise ModelTrainigError(f"Failed to save model locally: {e}", original_exception=e)


@trace
def train_from_config(
    config: Dict[str, Any],
    X_train:np.ndarray,
    y_train:np.ndarray,
    X_test:np.ndarray,
    y_test:np.ndarray,
    train_all:bool=False,
) -> Dict[str, Dict[str, float]]:
    """
    Train model(s) based on config.

    Args:
        config: Full configuration dict (loaded from config.yaml).
        X_train, y_train, X_test, y_test: Preprocessed data (labels already encoded).
        train_all: If True, train all models defined in config['models'].
                  If False, train only the model specified in config['active_model'].

    Returns:
        Dictionary with results: {model_name: metrics_dict}
    """
    n_classes = get_n_classes(y_train)
    mlflow_config = config.get("mlflow", {})
    results = {}

    if train_all:
        model_names = list(config["models"].keys())
        logger.info(f"Training all models defined in config: {model_names}")
    else:
        model_names = [config["active_model"]]
        logger.info(f"Training only active model defined in config: {model_names[0]}")
    for model_name in model_names:
        logger.info(f"\n{'='*40}\nStarting training for model: {model_name}\n{'='*40}")
        model_cfg = config["models"][model_name]

        #check if it is a stacking model
        if model_cfg.get("use_stacking"):
            logger.info("Building stacking classifier model...")
            base_estimators = []
            for base_name in model_cfg["base_models"]:
                base_params = config["models"][base_name]
                base_est = get_model_instance(base_name, base_params, n_classes)
                base_estimators.append((base_name, base_est))
            meta_model_name = model_cfg["meta_model"]
            meta_model_params = config["models"][meta_model_name]
            meta_est = get_model_instance(meta_model_name, meta_model_params, n_classes)
            model = StackingClassifier(estimators=base_estimators, 
                                       final_estimator=meta_est, 
                                       n_jobs=-1,
                                       cv = model_cfg.get("cv",5)
                                       )
        else:
            model = get_model_instance(model_name, model_cfg, n_classes)
        
        #define local save path
        local_path = Path("trained_models") / model_name

        metrics = train_and_evaluate(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_name=model_name,
            run_name = f"{model_name}_run",
            mlflow_config={
                "tracking_uri": mlflow_config.get("tracking_uri"),
                "experiment_name": mlflow_config.get("experiment_name","Sentiment_Analysis_Experiment"),
                "register_model": mlflow_config.get("register_model", False),
                "registered_model_name": mlflow_config.get("registered_model_name", model_name)
            },
            local_save_pth=local_path
        )
        results[model_name] = metrics
        logger.info(f"Completed training for model |{model_name}| with results: {metrics}")
    return results

if __name__ == "__main__":
    # Dummy data
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    config_example = {
        "active_model": "logistic_regression",
        "models": {
            "logistic_regression": {"C": 1.0, "solver": "lbfgs"},
            "random_forest": {"n_estimators": 10}
        },
        "mlflow": {
            "tracking_uri": "http://localhost:8000",
            "experiment_name": "test",
            "register_model": False
        }
    }
    results = train_from_config(config_example, X_train, y_train, X_test, y_test, train_all=False)
    print(results)