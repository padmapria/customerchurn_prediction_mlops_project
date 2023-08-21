# src/main.py
import sys,os,logging
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
from config.logger import LoggerSingleton
from model.model_prediction import predict_for_unseen_data

def predict_data():
   logger.info("Calling prediction over unseen data")
   predict_for_unseen_data(newdata_fileName,newdata_processed_fileName,newdata_prediction_fileName)
   logger.info("prediction complated")

if __name__ == "__main__":
    cfg = load_config()
    data_dir = cfg.data.data_dir

    # Set up logging using the loaded configuration
    logging.getLogger().propagate = False
    logger = LoggerSingleton().get_logger()  # Create the logger instance
    logger.info("Some message from train ***##")

    newdata_fileName = os.path.join(data_dir, cfg.data.batch_test_filename)
    newdata_processed_fileName = cfg.data.batch_test_processed_filename
    newdata_prediction_fileName = cfg.data.batch_test_predicted_filename
    predict_data()