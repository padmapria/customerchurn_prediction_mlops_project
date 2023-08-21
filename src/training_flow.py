# src/training_flow.py
import os,sys
import logging

# Import necessary modules from your project
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from config.config_handler import load_config
from config.logger import LoggerSingleton
import mlflow,prefect
from prefect import task, flow, context, get_run_logger
from data_processing.load_data import prepare_data_for_project, read_file
from data_processing.eda import perform_eda
from data_processing.training_data_preprocess import preprocess_data
from model.baseline_model import train_evaluate_LR
from model.hyperparameter_tuning import grid_search_RF, grid_search_GB
from model.model_prediction import load_model_and_predict, evaluate_model, predict_for_unseen_data
from model.monitoring_data_model_drift import generate_drift_report

def get_config():
    cfg = load_config()
    logger = LoggerSingleton().get_logger()  # Create the logger instance
    return cfg,logger

def log(msg):
    # Use your existing logger here
    logger = LoggerSingleton().get_logger()  # Get your existing logger instance

    # Print to stdout so logs will arrive at Prefect cloud
    print(msg)

    # Log to existing logger
    logger.info(msg)

# Create tasks using Prefect's @task decorator
@task(retries=3, retry_delay_seconds=2)
def load_data():
    try:
        cfg, logger = get_config()

        log("Load data log")

        data_dir = cfg.data.data_dir
        RAW_FILE = os.path.join(data_dir, cfg.data.raw_filename)
        prepare_data_for_project(RAW_FILE)

        TRAIN_TEST_FILE = os.path.join(data_dir, cfg.data.train_test_filename)
        return read_file(TRAIN_TEST_FILE)
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error loading data: %s", str(e))
        raise

@task
def perform_eda_task(df):
    # EDA tasks
    cfg, logger = get_config()
    cat_fileName = cfg.data.cat_eda_file
    num_fileName = cfg.data.num_eda_file
    corr_fileName = cfg.data.corr_eda_file
    perform_eda(df, cat_fileName, num_fileName, corr_fileName)

@task
def preprocess_data_task(df):
    try:
        cfg, logger = get_config()
        # Preprocessing task
        X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess_data(df)
        return X_train, X_test, X_validation, y_train, y_test, y_validation
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error in preprocessing data: %s", str(e))
        raise


@task
def train_evaluate_LR_task(X_train, y_train, X_validation, y_validation):
    try:
        cfg, logger = get_config()
        # Training and evaluating a baseline LR model
        logger.info("LR baseline")
        train_evaluate_LR(X_train, y_train, X_validation, y_validation)
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error training and evaluating LR model: %s", str(e))
        raise

@task(log_prints=True)
def grid_search_RF_task(X_train, y_train, X_validation, y_validation):
    try:
        cfg, logger = get_config()
        # Hyperparameter tuning using grid search for RF model
        logger.info("grid search")
        grid_search_RF(X_train, y_train, X_validation, y_validation)
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error during grid search: %s", str(e))
        raise

@task
def load_and_predict_task(X_test, model_name):
    try:
        cfg, logger = get_config()
        return load_model_and_predict(X_test, model_name)
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error during load and predict task: %s", str(e))
        raise

@task
def evaluate_model_task(X_test, y_test, model_name):
    try:
        cfg, logger = get_config()
        # Evaluating the trained model
        rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(X_test, y_test, model_name)
        return rf_accuracy, rf_precision, rf_recall, rf_f1
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error during evaluate model task: %s", str(e))
        raise

@task
def predict_for_unseen_data_task(fileName, processed_fileName, prediction_fileName):
    try:
        cfg, logger = get_config()
        # Predicting for unseen batch data
        predict_for_unseen_data(fileName, processed_fileName, prediction_fileName)
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error during predict for unseen data task: %s", str(e))
        raise

@task
def drift_report_task():
    # Generating drift report
    cfg, logger = get_config()
    generate_drift_report()

# Create a Prefect Flow
@flow
def main_flow(fileName, processed_fileName, prediction_fileName):
    try:
        cfg, logger = get_config()

        logger.info("within flow *******")
        print("main flow called")
        df = load_data()

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

        # Set the experiment name from the configuration file
        mlflow.set_experiment(cfg.mlflow.experiment_name)

        #perform_eda_task(df)
        X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess_data_task(df)
        #train_evaluate_LR_task(X_train, y_train, X_validation, y_validation)

        grid_search_RF_task(X_train, y_train, X_validation, y_validation)
        test_predictions = load_model_and_predict(X_test, 'random_forest')
        rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model_task(X_test, y_test, 'random_forest')

        predict_for_unseen_data_task(fileName, processed_fileName, prediction_fileName)
        drift_report_task()

        return 'success'
    except Exception as e:
        # Handle the exception, log an error, and potentially raise it again
        logger.error("Error in main flow: %s", str(e))
        return None


if __name__ == "__main__":
    try:
        cfg, logger = get_config()
        data_dir = cfg.data.data_dir
        # Run the Prefect Flow and capture its return value
        fileName = os.path.join(data_dir, cfg.data.batch_test_filename)
        processed_fileName = cfg.data.batch_test_processed_filename
        prediction_fileName = cfg.data.batch_test_predicted_filename

        flow_result = main_flow(fileName, processed_fileName, prediction_fileName)

        if flow_result is None:
            # Handle the case where an error occurred in the main flow
            logger.error("Main flow encountered an error.")
        else:
            logger.info("Main flow completed successfully.")
    except Exception as e:
        # Handle any unhandled exception at the top level
        logger.error("Error in main script: %s", str(e))