# config/config_handler.py
from hydra import initialize, compose
import logging, sys, os,warnings

# Define a variable to hold the configuration object
_config = None

def singleton_load_config(func):
    def wrapper(environment=None):
        global _config

        if _config is None:
            # Set the configuration path to the parent directory containing "config"
            config_path = "../config"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with initialize(config_path=config_path):
                    # Load the configuration based on the environment (default.yaml if not specified)
                    cfg = compose(config_name=environment if environment else "default")

            _config = cfg
        return _config
    return wrapper

@singleton_load_config
def load_config(environment=None):
    pass



