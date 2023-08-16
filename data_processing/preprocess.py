# data_processing/preprocess.py
import os,joblib,logging
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

cfg = load_config()
logger = LoggerSingleton().get_logger()  # Create the logger instance
logging.getLogger().propagate = False

data_dir = cfg.data.data_dir

ENCODER_DIR = os.path.join(data_dir,cfg.data.encoder_dir)
if not os.path.exists(ENCODER_DIR):
    os.makedirs(ENCODER_DIR)

cat_columns_to_encode = cfg.data.cat_columns_to_encode
numerical_columns = cfg.data.numerical_columns
target_col = cfg.data.target_column

training_data_filename = cfg.data.validation_data_filename
validation_data_filename = cfg.data.validation_data_filename
testing_data_filename = cfg.data.testing_data_filename

def shuffle_split_unbalanced_df(df, target_column, test_size=0.2, validation_size=0.15, random_state=None):
    # Separate features (X) and target variable (y)
    
    X = df.drop(columns=target_column)
    y = df[target_column]

    # Create a ShuffleSplit cross-validator
    shuffle_split = ShuffleSplit(n_splits=1, test_size=test_size + validation_size, random_state=random_state)

    # Generate the shuffling splits
    for train_temp_index, test_index in shuffle_split.split(X):
        # Split data into training and temporary set (remaining data)
        X_train_temp, X_test = X.iloc[train_temp_index], X.iloc[test_index]
        y_train_temp, y_test = y.iloc[train_temp_index], y.iloc[test_index]

    # Calculate the percentage of validation set out of the temporary set
    validation_ratio = validation_size / (1 - test_size)

    # Create another ShuffleSplit cross-validator for the temporary set
    shuffle_split_temp = ShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=random_state)

    # Generate the shuffling splits for the temporary set
    for train_index, validation_index in shuffle_split_temp.split(X_train_temp):
        # Split temporary set into training and validation sets
        X_train, X_validation = X_train_temp.iloc[train_index], X_train_temp.iloc[validation_index]
        y_train, y_validation = y_train_temp.iloc[train_index], y_train_temp.iloc[validation_index]

    return X_train, X_test, X_validation, y_train, y_test, y_validation


def encode_scale_data(df, onehot_columns=None,scaling_columns=None, save_encoder=True):
    df_encoded = df.copy()
    
    # Reset the index to ensure unique indices before encoding
    df_encoded.reset_index(drop=True, inplace=True)
    
    # One-hot encode specified columns
    if onehot_columns is not None:
        for column in onehot_columns:
            encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
            encoded_column = encoder.fit_transform(df_encoded[[column]])
            encoded_column_df = pd.DataFrame(encoded_column, columns=encoder.get_feature_names_out([column]))
            df_encoded = pd.concat([df_encoded, encoded_column_df], axis=1)
            df_encoded.drop(column, axis=1, inplace=True)
            
            if save_encoder:
                joblib.dump(encoder, os.path.join(ENCODER_DIR, f'{column}_encoder.joblib'))
     

    # Perform Min-Max scaling on specified columns
    if scaling_columns is not None:
        scalers = {}
        for column in scaling_columns:
            scaler = MinMaxScaler()
            df_encoded[column] = scaler.fit_transform(df_encoded[[column]])

            if save_encoder:
                joblib.dump(scaler, os.path.join(ENCODER_DIR, f'{column}_scaler.joblib'))
            
            scalers[column] = scaler

    return df_encoded
    
    
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

def apply_encoding_and_scaling(df, encoders,onehot_columns=None,scaling_columns=None):
    df_encoded = df.copy()
    # Reset the index to ensure unique indices before encoding
    df_encoded.reset_index(drop=True, inplace=True)

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
    encoders = load_encoders()
    if target_col in df.columns:
        y_te = df[target_col]
        X_te = df.drop(columns=target_col)
        X_test = apply_encoding_and_scaling(X_te, encoders, onehot_columns=cat_columns_to_encode,
                                            scaling_columns=numerical_columns)
        test_data = pd.concat([X_test, y_te], axis=1)
        return test_data
    else:
        X_te = df
        X_test = apply_encoding_and_scaling(X_te, encoders, onehot_columns=cat_columns_to_encode,
                                            scaling_columns=numerical_columns)
        return X_test


def preprocess_data(df):
    try:
        logger.info("Split data to train test validation")
        X_tr, X_te, X_val, y_train, y_test, y_validation = shuffle_split_unbalanced_df(df, target_column=target_col, 
                                                                                              test_size=0.2, validation_size=0.15, random_state=42)
        X_train = encode_scale_data(X_tr, onehot_columns=cat_columns_to_encode,scaling_columns=numerical_columns, save_encoder=True)
        encoders = load_encoders()

        X_validation = apply_encoding_and_scaling(X_val, encoders, onehot_columns=cat_columns_to_encode, scaling_columns=numerical_columns)
        X_test = apply_encoding_and_scaling(X_te, encoders, onehot_columns=cat_columns_to_encode, scaling_columns=numerical_columns)

        # Save the data to files
        train_data = pd.concat([X_train, y_train], axis=1)
        validation_data = pd.concat([X_validation, y_validation], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)

        train_data.to_csv(os.path.join(data_dir, training_data_filename), index=False)
        validation_data.to_csv(os.path.join(data_dir, validation_data_filename), index=False)
        test_data.to_csv(os.path.join(data_dir, testing_data_filename), index=False)

        return X_train, X_test, X_validation, y_train, y_test, y_validation
    
    except Exception as e:
        logger.exception("An error occurred during data preprocessing: %s", str(e))
        return None