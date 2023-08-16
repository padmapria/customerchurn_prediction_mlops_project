# model/model_prediction.py

import os, logging, mlflow, joblib
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
from model.baseline_model import calculate_scores

# Load the model configuration from config.config_handler
cfg = load_config()
logger = LoggerSingleton().get_logger()
logging.getLogger().propagate = False

MODEL_ARTIFACTS_DIR = cfg.models.model_artifact_dir

def load_and_predict(X_test, model_name):
    # Load the model from the directory based on the model name
    model_filename = None

    if model_name == 'random_forest':
        model_filename = cfg.models.random_forest.model_filename
    elif model_name == 'gradient_boosting':
        model_filename = cfg.models.gradient_boosting.model_filename
    else:
        logging.error("Invalid model name provided.")
        return None

    model_path = os.path.join(MODEL_ARTIFACTS_DIR, model_filename)
    best_model = joblib.load(model_path)

    # Make predictions using the loaded model
    predictions = best_model.predict(X_test)

    return predictions

def evaluate_model(X_test, y_test, model_name):
    logger.info("Evaluating model")
    predictions = load_and_predict(X_test, model_name)

    if y_test is not None:
        # Log the actual and predicted labels as an MLflow artifact
        results_df = pd.DataFrame({
            "Actual": y_test.values.ravel(),
            "Predicted": predictions
        })

        results_path = os.path.join(MODEL_ARTIFACTS_DIR, f"{model_name}_predictions.csv")
        results_df.to_csv(results_path, index=False)

        # Start MLflow run with a custom run name
        with mlflow.start_run(run_name=f"Evaluation_{model_name}"):

            mlflow.log_artifact(results_path)

            # Calculate evaluation metrics
            accuracy, precision, recall, f1 = calculate_scores(y_test, predictions)

            # Log the evaluation metrics as MLflow metrics
            mlflow.log_metrics({
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            })

            return accuracy, precision, recall, f1
    else:
        return None

