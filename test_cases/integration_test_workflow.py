# test_cases/integration_test_workflow.py

import unittest,logging,os,sys

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
from config.logger import LoggerSingleton

import sys
import os
import unittest
from io import StringIO
from contextlib import redirect_stdout
from src.main import workflow_main

class IntegrationTestWorkflow(unittest.TestCase):

    cfg = load_config()
    data_dir = cfg.data.data_dir
    logger = LoggerSingleton().get_logger()  # Create the logger instance
    logging.getLogger().propagate = False

    logger.info("unit Testing started **")

    def setUp(self):
        self.captured_output = StringIO()  # To capture print statements
        sys.stdout = self.captured_output  # Redirect stdout to capture print

    def tearDown(self):
        sys.stdout = sys.__stdout__  # Reset stdout

    def test_workflow(self):
        # Call the workflow_main function
        workflow_main()

        # Get the captured output and assert on expected results
        captured_output = self.captured_output.getvalue().strip().split('\n')
        
        # Example assertion on captured output
        self.assertIn("Test data RF Accuracy:", captured_output)

if __name__ == '__main__':
    unittest.main()
