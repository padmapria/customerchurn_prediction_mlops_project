# model/model_prediction.py

import os, logging, mlflow, joblib
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
from model.baseline_model import calculate_scores
from data_processing.preprocess import preprocess_unseen_data

# Load the model configuration from config.config_handler
cfg = load_config()
logger = LoggerSingleton().get_logger()
logging.getLogger().propagate = False

MODEL_ARTIFACTS_DIR = cfg.models.model_artifact_dir

# Define a prediction function
def load_model_mlflow_and_predict(data, model_name1):
    # Set the experiment name
    experiment_name = cfg.mlflow.best_artifact_experiment_name
    mlflow.set_experiment(experiment_name)

    model_filename = cfg.models.random_forest.model_filename
    
    client = mlflow.tracking.MlflowClient()

    # Get the latest version of the registered model
    model_instance = client.get_latest_versions(name = model_filename)[0]
    model_version  = model_instance.version 

    # Load the latest version of the registered model
    model_uri = f"models:/{model_filename}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    logger.info("Loaded model from MLFlow for prediction")

    # You can process input data as needed
    predictions = model.predict(data)
    return predictions

def load_model_and_predict(X_test, model_name):
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


def compare_prediction(X_test, y_test, model_name):
    predictions = load_model_mlflow_and_predict(X_test, model_name)
    
    print(X_test.index)
    X_test.set_index(cfg.data.index_col[0], inplace=True)
    print(X_test.index)
    if y_test is not None:
        # Log the actual and predicted labels as an MLflow artifact
        results_df = pd.DataFrame({
            cfg.data.index_col[0]: X_test.index,
            "Actual": y_test.values.ravel(),
            "Predicted": predictions
        })

        return predictions,results_df

    return predictions,pd.DataFrame()

def evaluate_model(X_test, y_test, model_name):
    logger.info("Evaluating model")

    predictions, results_df = compare_prediction(X_test, y_test, model_name)

    if y_test is not None:
        # Log the actual and predicted labels as an MLflow artifact
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


def predict_for_unseen_data(fileName,processed_fileName, prediction_fileName):
    df = pd.read_csv(fileName)
    test_data = preprocess_unseen_data(df)
    
    target_col = cfg.data.target_column[0]
    
    if target_col in test_data.columns:
        print("Target column is present")
        X_test = test_data.drop(columns=[target_col])
        y_te = test_data[target_col]
    else:
        print("Target column is not present")
        print("Columns in test_data:", test_data.columns)
    
    test_data.to_csv(os.path.join(cfg.data.data_dir,  processed_fileName), index=False)
    predictions,results_df = compare_prediction(X_test, y_te, 'random_forest')

    results_path = os.path.join(cfg.data.data_dir, prediction_fileName)
    results_df.to_csv(results_path, index=False)
