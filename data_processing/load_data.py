# data_processing/download_data.py
import pandas as pd
import logging,os,sys
from config.config_handler import load_config

cfg = load_config()
data_dir = cfg.data.data_dir
target_col = cfg.data.target_column

def read_data_and_drop_duplicates(input_file, reset_index=True):
    # Read the DataFrame from the Parquet file
    
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
    
    logging.info(df[target_col].value_counts())
    
    _drop_unwanted_cols(df)

    return df
    
def _drop_unwanted_cols(df):
    cols_to_drop = cfg.data.cols_to_drop
    columns_to_drop = [col for col in cols_to_drop if col in df.columns]
    
    if columns_to_drop:
        df.drop(columns_to_drop, axis=1, inplace=True)
    
 