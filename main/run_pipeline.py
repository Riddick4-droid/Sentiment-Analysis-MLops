#!/usr/bin/env python3
"""
Main pipeline orchestration script for sentiment analysis MLOps.
Runs end-to-end: data ingestion -> preprocessing -> training -> evaluation -> registration.
"""

import sys
import argparse
from pathlib import Path
import glob
import numpy as np
import pickle
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger, log_exception
from src.utils.tracer import trace
from src.data.data_ingestion import load_data, get_data_stats
from src.data.data_preprocessing import (
    preprocess_pipeline,
    save_preprocessing_artifacts,
    create_processed_dataframe
)
from src.models.model_building import train_from_config
from src.models.model_evaluation import evaluate_model, compare_models, load_local_model
import src.models.register_model as register_model_mod

logger = setup_logger("pipeline")


def save_processed_data(X_train, X_test, y_train, y_test, label_encoder, vectorizer, processed_df, output_dir="processed_data"):
    """Save preprocessed data and artifacts to disk."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save features as numpy arrays
    np.save(output_path / "X_train.npy", X_train)
    np.save(output_path / "X_test.npy", X_test)
    np.save(output_path / "y_train.npy", y_train)
    np.save(output_path / "y_test.npy", y_test)

    # Save label encoder and vectorizer
    with open(output_path / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    with open(output_path / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save the processed DataFrame (raw text, cleaned text, original labels, encoded labels)
    if processed_df is not None:
        processed_df.to_csv(output_path / "processed_data.csv", index=False)
        logger.info(f"Processed DataFrame saved to {output_path / 'processed_data.csv'}")

    logger.info(f"All processed data saved to {output_path}")


@trace
def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Run sentiment analysis MLOps pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--train-all", action="store_true", help="Train all models defined in config")
    parser.add_argument("--skip-training", action="store_true", help="Skip training, only evaluate existing models")
    parser.add_argument("--register-best", action="store_true", help="Register best model to MLflow Production")
    parser.add_argument("--save-processed", action="store_true", help="Save preprocessed data to disk")
    parser.add_argument("--save-processed-only", action="store_true", help="Only preprocess, save data, and exit (skip training)")

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Starting Sentiment Analysis MLOps Pipeline")
    logger.info("=" * 60)

    try:
        # 1. Load configuration
        logger.info(f"Loading config from {args.config}")
        config = load_config(args.config)
        logger.info("Config loaded successfully")

        # 2. Data ingestion
        logger.info("Data ingestion commencing...")
        df = load_data(
            file_path=config["data"]["raw_path"],
            text_column=config["data"]["text_column"],
            label_column=config["data"]["label_column"],
        )
        stats = get_data_stats(df, config["data"]["text_column"], config["data"]["label_column"])
        logger.info(f"Data stats: {stats}")

        # 3. Preprocessing step
        logger.info("Preprocessing...")
        X_train, X_test, y_train, y_test, label_encoder, fitted_vectorizer = preprocess_pipeline(
            df=df,
            text_column=config["data"]["text_column"],
            label_column=config["data"]["label_column"],
            vectorizer_method=config["vectorizer_method"],
            vectorizer_kwargs=config["vectorizer_params"],
            test_size=config["training"]["test_size"],
            random_state=config["training"].get("random_state", 42),
            cleaning_kwargs=config.get("cleaning", {})
        )
        logger.info(f"Preprocessing complete: train shape: {X_train.shape} | test shape: {X_test.shape}")

        # Create processed DataFrame with raw text, cleaned text, original labels, encoded labels
        processed_df, full_label_encoder = create_processed_dataframe(
            df,
            config["data"]["text_column"],
            config["data"]["label_column"],
            clean_kwargs=config.get("cleaning", {})
        )
        # Note: label_encoder from preprocess_pipeline is the same as full_label_encoder (fitted on all data)
        # Ensure the directory for saving exists
        Path("data/processed").mkdir(parents=True, exist_ok=True)

        # Save processed DataFrame to CSV (always saved for reference)
        processed_df.to_csv("data/processed/processed_data.csv", index=False)
        logger.info("Saved processed DataFrame with cleaned texts and encoded labels to data/processed/processed_data.csv")

        # Save preprocessing artifacts (vectorizer, label encoder)
        artifacts_dir = config.get("artifacts", {}).get("preprocessing_dir", "artifacts/preprocessing")
        save_preprocessing_artifacts(
            vectorizer=fitted_vectorizer,
            label_encoder=label_encoder,
            output_dir=artifacts_dir
        )
        logger.info(f"Saved preprocessing artifacts to: {artifacts_dir}")

        # If only preprocessing is requested, save everything and exit
        if args.save_processed_only:
            save_processed_data(X_train, X_test, y_train, y_test, label_encoder, fitted_vectorizer, processed_df)
            logger.info("Preprocessing completed and data saved. Exiting.")
            return

        # Optionally save processed data (including train/test splits) even if training continues
        if args.save_processed:
            save_processed_data(X_train, X_test, y_train, y_test, label_encoder, fitted_vectorizer, processed_df)

        # 4. Training
        if not args.skip_training:
            logger.info("Model training commencing...")
            results = train_from_config(
                config=config,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                train_all=args.train_all
            )
            logger.info(f"Training results: {results}")
        else:
            logger.info("Skipping training as requested...")
            results = {}

        # 5. Evaluation stage
        model_paths = glob.glob("trained_models/**/*_model.pkl", recursive=True)

        if not model_paths:
            logger.warning("No *_model.pkl files found in trained_models/")
        else:
            models_dict = {}
            for pkl_path in model_paths:
                filename = Path(pkl_path).name
                if filename.endswith("_model.pkl"):
                    model_name = filename[:-10]  # remove "_model.pkl"
                else:
                    model_name = Path(pkl_path).parent.name
                try:
                    model = load_local_model(pkl_path)
                    models_dict[model_name] = model
                    logger.info(f"Loaded model: {model_name} from {pkl_path}")
                except Exception as e:
                    logger.error(f"Failed to load model from {pkl_path}: {e}")

            if models_dict:
                comparison_df = compare_models(
                    models_dict, X_test, y_test,
                    log_to_mlflow=True,
                    mlflow_config=config.get("mlflow")
                )
                logger.info(f"Model comparison:\n{comparison_df.to_string()}")
            else:
                logger.warning("No valid models could be loaded for comparison")

        # 6. Model registration if requested
        if args.register_best and config.get("mlflow", {}).get("register_model", False):
            logger.info("Model Registration...")
            registered_model_name = config.get("mlflow", {}).get("registered_model_name", "Sentiment_Model")

            if results:
                best_model_name = max(results.items(), key=lambda x: x[1]["accuracy"])[0]
                best_run_id = results[best_model_name]["run_id"]
                logger.info(f"Best model: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}")

                version_info = register_model_mod.register_model_version(
                    run_id=best_run_id,
                    model_path=best_model_name,
                    registered_model_name=registered_model_name,
                    tracking_uri=config["mlflow"].get("tracking_uri"),
                    description=f"Best model from pipeline run: {best_model_name} with accuracy {results[best_model_name]['accuracy']:.4f}"
                )
                logger.info(f"Registered version {version_info['version']} for {registered_model_name}")

                register_model_mod.transition_model_stage(
                    registered_model_name=registered_model_name,
                    version=version_info["version"],
                    stage="Production",
                    tracking_uri=config["mlflow"].get("tracking_uri"),
                    archive_existing_versions=True
                )
                logger.info(f"Promoted version {version_info['version']} to Production")
            else:
                logger.warning("No training results available to determine best model for registration")

        logger.info("=" * 60)
        logger.info("Pipeline completed successfully")
        logger.info("=" * 60)

    except Exception as e:
        log_exception(logger, e, "Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()