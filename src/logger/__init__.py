import os
from logging.handlers import RotatingFileHandler
import logging
from from_root import from_root
from datetime import datetime


# logging constants
LOG_DIR = "log"
LOG_FILE = f"database_{datetime.now().strftime('%Y-%m-%d')}.log"
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10 MB
BACKUP_COUNT = 5  # Keep 5 backup log files

#construct log file path
LOG_FILE_PATH = os.path.join(from_root(), LOG_DIR)
os.makedirs(LOG_FILE_PATH, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_FILE_PATH, LOG_FILE)

def configure_logging():
    """
    Configures the logging settings for the application.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create a rotating file handler
    handler = RotatingFileHandler(LOG_FILE_PATH, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

configure_logging() 
