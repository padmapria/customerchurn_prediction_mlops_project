# src/runtest_deploy.py
import sys,os,logging
# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from config.config_handler import load_config
from config.logger import LoggerSingleton

import subprocess,unittest
#from test_cases import unittest_preprocessing
from test_cases.unittest_preprocessing import TestPreprocessing
from test_cases.integration_test_workflow import TestMainFlow


def run_tests():
    logger.info("Starting tests")

    # Create test suites for unit and integration tests
    unit_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestPreprocessing)
    integration_test_suite = unittest.TestLoader().loadTestsFromTestCase(TestMainFlow)

    # Run the tests
    unit_test_result = unittest.TextTestRunner().run(unit_test_suite)
    integration_test_result = unittest.TextTestRunner().run(integration_test_suite)

    # Check the test results
    if unit_test_result.wasSuccessful() and integration_test_result.wasSuccessful():
        print("All tests passed. Building Docker image and deploying...")
        build_and_deploy_docker_image()
    else:
        print("Tests failed. Aborting deployment.")

def build_and_deploy_docker_image():
    print("starting Docker image...")
    image_name = "myproject-image"
    dockerfile_path = "./"  # Path to the directory containing Dockerfile

    # Build and deploy the Docker image
    print("Building Docker image...")
    subprocess.run(["docker", "build", "-t", image_name, dockerfile_path])

if __name__ == "__main__":
    logger = LoggerSingleton().get_logger()  # Create the logger instance
    logging.getLogger().propagate = False

    run_tests()
