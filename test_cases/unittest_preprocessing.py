# test_cases/unittest_preprocessing.py
from config.config_handler import load_config
from config.logger import LoggerSingleton
import unittest,logging,os
import numpy as np
import pandas as pd
from data_processing.training_data_preprocess import preprocess_unseen_data

class TestPreprocessing(unittest.TestCase):

    def test_all_columns_converted_to_numbers(self):
        cfg = load_config()
        data_dir = cfg.data.data_dir
        logger = LoggerSingleton().get_logger()  # Create the logger instance
        logging.getLogger().propagate = False

        logger.info("unit Testing started **")
    
        # Create a sample DataFrame with the same structure as your input data
        fileName = os.path.join(data_dir, cfg.data.unit_integration_filename)
        df = pd.read_csv(fileName)
        X_test = preprocess_unseen_data(df)

        # Get the data types of all columns
        column_data_types = X_test.dtypes

        # Assert that all columns in the output are numeric (float64 or int64)
        self.assertTrue(all(column_data_types.isin([np.dtype('float64'), np.dtype('int64')])))
        
        logger.info("unit Testing done **")

if __name__ == '__main__':
    unittest.main()

