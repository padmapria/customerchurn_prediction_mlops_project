# model/monitoring_data_model_drift.py

import os,logging,joblib,mlflow,json
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset

cfg = load_config()
data_dir = cfg.data.data_dir
logger = LoggerSingleton().get_logger()
logging.getLogger().propagate = False

def generate_drift_report():
    # Load historical and new data
    historical_data = pd.read_csv(os.path.join(data_dir, cfg.data.training_data_filename))
    new_data = pd.read_csv(os.path.join(data_dir, cfg.data.batch_test_processed_filename))
    
    target_col = cfg.data.target_column[0]
    row_col = cfg.data.index_col[0]
    
    historical_data.drop(columns=[target_col,row_col ], inplace=True)
    new_data.drop(columns=[target_col,row_col ], inplace=True)
    
    # Select only the columns you want to analyze
    columns_to_analyze = ['CreditScore','Age','Balance','EstimatedSalary','Complain']
    
    # List the columns you want
    historical_subset = historical_data[columns_to_analyze]
    new_subset = new_data[columns_to_analyze]
        
    data_drift = Report(metrics = [DataDriftPreset()])
    data_drift.run(current_data = historical_subset,
                   reference_data = new_subset)

    data_drift.show()

    data_drift._save("data_drift_dashboard_after_week1.html")

