from src.logger import logging
from src.exception import MyException, error_message_detail
import sys

logging.info("Logging is configured successfully.")

try:
    a = 1 / 0
except Exception as e:
    error_message = error_message_detail(e, sys) # type: ignore
    #logging.error(f"An error occurred: {error_message}")