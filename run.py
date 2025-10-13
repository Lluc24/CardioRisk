from implementations import mean_squared_error_gd, predict_labels, least_squares
from data_cleaning import get_initial_data, get_thresholded_data, get_headers, cleaning_pipeline, save_arrays
import numpy as np
from helpers import create_csv_submission
import logging
import sys
import pathlib

DATASET_DIR = "dataset"
METADATA_FILE = "dataset_metadata.json"

def setup_logger():
    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler("logging.log", mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    # Stream (stdout) handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    # Get root logger and set level
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicated handlers if script is run multiple times
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    else:
        logger.handlers.clear()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    # Example usage
    logger.info("The messages will go both to logging.log and stdout.")
    logger.info("Logger is set up.")

if __name__ == '__main__':
    setup_logger()
    x_train, x_test, y_train, train_ids, test_ids = cleaning_pipeline(dataset_path="dataset")
    save_arrays(x_train=x_train, x_test=x_test, y_train=y_train, train_ids=train_ids, test_ids=test_ids)


