# data_processing/unseen_data_preprocess.py
import os, joblib, logging
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd

cfg = load_config()
logger = LoggerSingleton().get_logger()  # Create the logger instance
logging.getLogger().propagate = False

data_dir = cfg.data.data_dir

ENCODER_DIR = os.path.join(data_dir, cfg.data.encoder_dir)
if not os.path.exists(ENCODER_DIR):
    os.makedirs(ENCODER_DIR)

cat_columns_to_encode = cfg.data.cat_columns_to_encode
numerical_columns = cfg.data.numerical_columns
target_col = cfg.data.target_column

## TO encode unseen data
def load_encoders(encoder_folder=ENCODER_DIR):
    encoder_files = os.listdir(encoder_folder)
    encoder_files = [file for file in encoder_files if file.endswith('.joblib')]

    encoders = {}
    for file in encoder_files:
        name, _ = os.path.splitext(file)
        encoder_path = os.path.join(encoder_folder, file)
        encoder_categories = joblib.load(encoder_path)
        encoder_name = name.rsplit('_', 1)[0]
        encoders[encoder_name] = encoder_categories

    return encoders


def apply_encoding_and_scaling(df, onehot_columns=None, scaling_columns=None):
    df_encoded = df.copy()
    # Reset the index to ensure unique indices before encoding
    df_encoded.reset_index(drop=True, inplace=True)

    encoders = load_encoders()
    # One-hot encode specified columns
    if onehot_columns is not None:
        for column in onehot_columns:
            encoder = encoders[column]
            encoded_data = encoder.transform(df_encoded[[column]])
            encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column]))
            df_encoded.drop(column, axis=1, inplace=True)
            df_encoded = pd.concat([df_encoded, encoded_df], axis=1)

    # Perform Min-Max scaling on specified columns
    if scaling_columns is not None:
        for column in scaling_columns:
            scaler = encoders[column]
            df_encoded[column] = scaler.transform(df_encoded[[column]])

    return df_encoded


def preprocess_unseen_data(df):
    X_test = apply_encoding_and_scaling(df, onehot_columns=cat_columns_to_encode,
                                        scaling_columns=numerical_columns)
    return X_test
