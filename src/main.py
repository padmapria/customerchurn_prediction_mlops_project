# src/main.py
import os,logging,sys
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
from data_processing.load_data import read_data_and_drop_duplicates
from data_processing.eda import perform_eda
from data_processing.preprocess import preprocess_data
from model.baseline_model import train_evaluate_LR
from model.hyperparameter_tuning import grid_search_RF,grid_search_GB

# Call the load_config to get the configuration object
cfg = load_config()

def workflow_main_():
    logging.info('starting the process')
    
    data_dir = cfg.data.data_dir
    raw_file = cfg.data.raw_filename
    INPUT_FILE = os.path.join(data_dir,raw_file )
    
    cat_fileName = cfg.data.cat_eda_file
    num_fileName = cfg.data.num_eda_file
    corr_fileName =cfg.data.corr_eda_file

    df = read_data_and_drop_duplicates(INPUT_FILE)
    
    perform_eda(df,cat_fileName,num_fileName,corr_fileName)
    X_train, X_test, X_validation, y_train, y_test, y_validation = preprocess_data(df)
    train_evaluate_LR(X_train, y_train, X_validation, y_validation)
    
    # Assuming you have X_val and y_val as your validation data, call the functions with your data
    grid_search_RF(X_train, y_train, X_validation, y_validation)
    grid_search_GB(X_train, y_train, X_validation, y_validation)
    
    
if __name__ == "__main__":    
    workflow_main_()

