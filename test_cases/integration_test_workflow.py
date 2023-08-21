# test_cases/integration_test_workflow.py

import unittest
import os
import sys,prefect

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
from config.logger import LoggerSingleton
from prefect.testing.utilities import prefect_test_harness


from src.training_flow import main_flow  # Import your actual main_flow function

class TestMainFlow(unittest.TestCase):

    def test_integration(self):
        # Set up any necessary configuration or data for your test
        cfg, logger = self.setup()

        logger.info("Prefect flow Integration Testing started **")

        fileName = os.path.join(cfg.data.data_dir, cfg.data.unit_integration_filename)
        processed_fileName = cfg.data.unit_integration_processed_filename
        prediction_fileName = cfg.data.unit_integration_predicted_filename

        with prefect_test_harness():
            # run the flow against a temporary testing database
            assert main_flow(fileName, processed_fileName, prediction_fileName) == 'success'

    def setup(self):
        cfg = load_config()
        logger = LoggerSingleton().get_logger()  # Create the logger instance
        return cfg, logger

if __name__ == "__main__":
    unittest.main()
