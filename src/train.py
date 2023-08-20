# src/train.py

import os,sys,logging
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
from config.logger import LoggerSingleton
import mlflow, prefect
from prefect import task, flow, context, get_run_logger
from data_processing.load_data import prepare_data_for_project,read_file
from data_processing.eda import perform_eda
from data_processing.preprocess import preprocess_data
from model.baseline_model import train_evaluate_LR
from model.hyperparameter_tuning import grid_search_RF, grid_search_GB
from model.model_prediction import load_model_and_predict,evaluate_model,predict_for_unseen_data
from model.monitoring_data_model_drift import generate_drift_report

logging.getLogger().propagate = False
cfg = load_config()
data_dir = cfg.data.data_dir

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
    log("Load data log")

    data_dir = cfg.data.data_dir
    RAW_FILE = os.path.join(data_dir, cfg.data.raw_filename)
    prepare_data_for_project(RAW_FILE)

    TRAIN_TEST_FILE = os.path.join(data_dir, cfg.data.train_test_filename)
    return read_file(TRAIN_TEST_FILE)

@task
def perform_eda_task(df):
    cat_fileName = cfg.data.cat_eda_file
    num_fileName = cfg.data.num_eda_file
    corr_fileName = cfg.data.corr_eda_file
    perform_eda(df, cat_fileName, num_fileName, corr_fileName)

@task
def preprocess_data_task(df):
    X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess_data(df)
    return X_train, X_test, X_validation, y_train, y_test, y_validation

@task
def train_evaluate_LR_task(X_train, y_train, X_validation, y_validation):
    logger.info("LR baseline")
    train_evaluate_LR(X_train, y_train, X_validation, y_validation)

@task(log_prints=True)
def grid_search_RF_task(X_train, y_train, X_validation, y_validation):
    logger.info("grid search")
    grid_search_RF(X_train, y_train, X_validation, y_validation)

@task
def load_and_predict_task(X_test, model_name):
    return load_model_and_predict(X_test, model_name)

@task
def evaluate_model_task(X_test, y_test, model_name):
    rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(X_test, y_test, model_name)
    return rf_accuracy, rf_precision, rf_recall, rf_f1
    
@task
def predict_for_unseen_data_task():
    fileName = os.path.join(data_dir, cfg.data.batch_test_filename)
    processed_fileName = cfg.data.batch_test_processed_filename
    prediction_fileName = cfg.data.batch_test_predicted_filename
    
    predict_for_unseen_data(fileName,processed_fileName,prediction_fileName)
    
    
@task
def drift_report_task():
    generate_drift_report()

# Create a Prefect Flow
@flow
def main_flow():
    logger.info("within floww *******")
    print("main flow called")
    df = load_data()
    #perform_eda_task(df)
    X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess_data_task(df)
    #train_evaluate_LR_task(X_train, y_train, X_validation, y_validation)

    grid_search_RF_task(X_train, y_train, X_validation, y_validation)
    test_predictions = load_model_and_predict(X_test, 'random_forest')
    rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model_task(X_test, y_test, 'random_forest')
    
    predict_for_unseen_data_task()
    drift_report_task()


def normal():
    logger.warning("info log from normal")
    logger.warning("Normal log not passed")

if __name__ == "__main__":

    cfg = load_config()
    logger = LoggerSingleton().get_logger()  # Create the logger instance
    logger.info("Some message from train ***##")
    normal()

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Set the experiment name from the configuration file
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Run the Prefect Flow
    main_flow()




