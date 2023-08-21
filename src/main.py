# src/main.py
import sys,os,logging
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
import mlflow
from config.logger import LoggerSingleton

from data_processing.load_data import read_file,prepare_data_for_project
from data_processing.eda import perform_eda
from data_processing.training_data_preprocess import preprocess_data
from model.baseline_model import train_evaluate_LR
from model.hyperparameter_tuning import grid_search_RF,grid_search_GB
from model.model_prediction import load_model_and_predict,evaluate_model,predict_for_unseen_data
from model.monitoring_data_model_drift import generate_drift_report

def get_config():
    cfg = load_config()

    # Set up logging using the loaded configuration
    logging.getLogger().propagate = False
    logger = LoggerSingleton().get_logger()  # Create the logger instance
    logger.info("Some message from train ***##")

    return cfg,logger

def workflow_main(newdata_fileName,newdata_processed_fileName,newdata_prediction_fileName):
    cfg, logger  = get_config()
    data_dir = cfg.data.data_dir

    RAW_FILE = os.path.join(data_dir,  cfg.data.raw_filename)
    TRAIN_TEST_FILE = os.path.join(data_dir, cfg.data.train_test_filename)

    cat_fileName = cfg.data.cat_eda_file
    num_fileName = cfg.data.num_eda_file
    corr_fileName =cfg.data.corr_eda_file

    prepare_data_for_project(RAW_FILE)

    df = read_file(TRAIN_TEST_FILE)

    # Set MLflow tracking URI
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Set the experiment name from the configuration file
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    perform_eda(df,cat_fileName,num_fileName,corr_fileName)
    X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess_data(df)
    #train_evaluate_LR(X_train, y_train, X_validation, y_validation)

    # Assuming you have X_val and y_val as your validation data, call the functions with your data
    #grid_search_RF(X_train, y_train, X_validation, y_validation)
    #grid_search_GB(X_train, y_train, X_validation, y_validation)

    # Assuming you have X_test as your test data and MODEL_ARTIFACTS_DIR as the directory where the model is stored
    test_predictions = load_model_and_predict(X_test, 'random_forest')

    # Call the function with your data and the desired model name (e.g., 'random_forest' or 'gradient_boosting')
    rf_accuracy, rf_precision, rf_recall, rf_f1 = evaluate_model(X_test, y_test, 'random_forest')

    # Print the results
    logger.info("Test data RF Accuracy: %s", rf_accuracy)
    logger.info("Test data RF Precision: %s", rf_precision)
    logger.info("Test data RF Recall: %s", rf_recall)
    logger.info("Test data RF F1-score: %s", rf_f1)

    predict_for_unseen_data(newdata_fileName,newdata_processed_fileName,newdata_prediction_fileName)
    generate_drift_report()

if __name__ == "__main__":
    cfg = load_config()
    data_dir = cfg.data.data_dir
    newdata_fileName = os.path.join(data_dir, cfg.data.batch_test_filename)
    newdata_processed_fileName = cfg.data.batch_test_processed_filename
    newdata_prediction_fileName = cfg.data.batch_test_predicted_filename

    workflow_main(newdata_fileName,newdata_processed_fileName,newdata_prediction_fileName)

