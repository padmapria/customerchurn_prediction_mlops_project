# config/logger.py
import logging
from config.config_handler import load_config

class LoggerSingleton:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance.setup_logging()
        return cls._instance

    def setup_logging(self):
        self.logger = logging.getLogger()  # Get the root logger instance

        cfg = load_config()
        self.logger.setLevel(cfg.logging.level)  # Set the logging level
        formatter = logging.Formatter(cfg.logging.format)  # Create a formatter

        # Configure handlers for both file and console logging
        file_handler = logging.FileHandler(cfg.logging.file_name)
        file_handler.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

        self.logger.info("Logging configuration set up")

    def get_logger(self):
        return self.logger

# Usage in other modules:
# from config.logger import LoggerSingleton
# logger_instance = LoggerSingleton().get_logger()
# logger_instance.info("This is a log message from another module.")
