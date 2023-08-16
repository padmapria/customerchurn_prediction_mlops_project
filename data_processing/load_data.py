# data_processing/download_data.py
from config.config_handler import load_config
from config.logger import LoggerSingleton
import logging,os
import pandas as pd

cfg = load_config()
data_dir = cfg.data.data_dir
target_col = cfg.data.target_column

logging.getLogger().propagate = False
logger = LoggerSingleton().get_logger()  # Create the logger instance

def save_dataframe_subset(df, start_index, end_index, filename):
    subset_df = df.iloc[start_index:end_index]
    subset_df.to_csv(filename, index=False)

def prepare_data_for_project(input_file, reset_index=True):
    # Read the DataFrame
    df = pd.read_csv(input_file)
    
    # Check for duplicate rows
    duplicates = df.duplicated()

    # Print the duplicate rows (if any)
    if duplicates.any():
        logging.info("Duplicate rows found: ", len(df[duplicates]))
    else:
        logging.info("No Duplicates")

    # Drop duplicate rows and keep the first occurrence
    df.drop_duplicates(keep='first', inplace=True)

    # Optionally, reset the index if needed
    if reset_index:
        df.reset_index(drop=True, inplace=True)

    _drop_unwanted_cols(df)

    # Define start and end indices for last 300 rows
    start_index_last_100 = -300
    end_index_last_100 = None  # No need to specify an end index, as it's automatically the end of the DataFrame
    filename_last_100 =  os.path.join(data_dir,cfg.data.batch_test_filename)

    # Define start and end indices for 600 to 300 rows
    start_index_200_to_100 = -600
    end_index_200_to_100 = -300
    filename_200_to_100 =  os.path.join(data_dir,cfg.data.unit_integration_filename)

    # Save the last 100 rows
    save_dataframe_subset(df, start_index_last_100, end_index_last_100, filename_last_100)

    # Save 200 to 100 rows
    save_dataframe_subset(df, start_index_200_to_100, end_index_200_to_100, filename_200_to_100)

    # Save remaining data as train_cv_test_data.csv
    save_dataframe_subset(df, 0, -200,  os.path.join(data_dir,cfg.data.train_test_filename))


def read_file(input_file, reset_index=True):
    df = pd.read_csv(input_file)
    logger.info(df[target_col].value_counts())
    return df


def _drop_unwanted_cols(df):
    cols_to_drop = cfg.data.cols_to_drop
    columns_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
    
 