"""
Main pipeline orchestration script for sentiment analysis MLOps.
Runs end-to-end: data ingestion -> preprocessing -> training -> evaluation -> registration.
"""

import sys
import argparse
from pathlib import Path

# Add src to path if running from root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, log_exception
from src.utils.tracer import trace, TRACING_ENABLED
from src.data.data_ingestion import load_data, get_data_stats
from src.data.data_preprocessing import (
    preprocess_pipeline,
    save_preprocessing_artifacts
)
from src.models.model_building import train_from_config
from src.models.model_evaluation import evaluate_model, compare_models, load_local_model
from src.models.register_model import (
    register_model_version,
    transition_model_stage,
    promote_best_model
)

# Setup logger
logger = setup_logger("pipeline")

@trace
def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Run sentiment analysis MLOps pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--train-all", action="store-true", help="Train all models defined in config")
    parser.add_argument("--skip-training", action="store-true", help="skip training, only evaluate existing models")
    parser.add_argument("--register-best", action="store_true", help="Register best model to MLflow Production")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Sentiment Analysis MLOps Pipeline")
    logger.info("=" * 60)

    