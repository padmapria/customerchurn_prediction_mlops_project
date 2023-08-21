# data_processing/training_data_preprocess.py
import os,joblib,logging
from config.config_handler import load_config
from config.logger import LoggerSingleton
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from config.logger import LoggerSingleton
from data_processing.unseen_data_preprocess import apply_encoding_and_scaling

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

training_data_filename = cfg.data.training_data_filename
validation_data_filename = cfg.data.validation_data_filename
testing_data_filename = cfg.data.testing_data_filename

def shuffle_split_unbalanced_df(df, target_column, test_size=0.2, validation_size=0.15, random_state=None):
    # Separate features (X) and target variable (y)
    
    X = df.drop(columns=target_column)
    y = df[target_column]
    
    # Shuffle the data
    X_shuffled, y_shuffled = X.sample(frac=1, random_state=42), y.sample(frac=1, random_state=42)

    # Split the shuffled data into train, validation, and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.2, random_state=42)
    
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.15, random_state=42)

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
    

def preprocess_data(df):
    try:
        logger.info("Split data to train test validation")
        X_tr, X_te, X_val, y_train, y_test, y_validation = shuffle_split_unbalanced_df(df, target_column=target_col, 
                                                                                              test_size=0.2, validation_size=0.15, random_state=42)
        
        # Save the data to files
        train_data1 =  X_tr.join(y_train)
        validation_data1 = X_val.join(y_validation)
        test_data1 = X_te.join(y_test)   
        
        train_data = encode_scale_data(train_data1, onehot_columns=cat_columns_to_encode,scaling_columns=numerical_columns, save_encoder=True)

        validation_data = apply_encoding_and_scaling(validation_data1, onehot_columns=cat_columns_to_encode, scaling_columns=numerical_columns)
        test_data = apply_encoding_and_scaling(test_data1, onehot_columns=cat_columns_to_encode, scaling_columns=numerical_columns)

        train_data.to_csv(os.path.join(data_dir, training_data_filename),index=False)
        validation_data.to_csv(os.path.join(data_dir, validation_data_filename),index=False)
        test_data.to_csv(os.path.join(data_dir, testing_data_filename),index=False)
        
        X_train = train_data.drop(columns=target_col)
        y_train = train_data[target_col]
        X_validation = validation_data.drop(columns=target_col)
        y_validation = validation_data[target_col]
        X_test = test_data.drop(columns=target_col)
        y_test = test_data[target_col]
        
        return X_train, X_test, X_validation, y_train, y_test, y_validation
    
    except Exception as e:
        logger.exception("An error occurred during data preprocessing: %s", str(e))
        return None