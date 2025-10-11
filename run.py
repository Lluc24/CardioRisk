from implementations import mean_squared_error_gd, predict_labels, least_squares
from data_cleaning import get_initial_data, get_thresholded_data, get_data_columns, get_headers, cleaning_pipeline
import numpy as np
from helpers import create_csv_submission
import logging
import sys
import pathlib

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

if __name__ == 'x__main__':
    setup_logger()
    x_train, x_test, y_train, train_ids, test_ids = cleaning_pipeline(dataset_path="dataset")
    p = pathlib.Path("clean_arrays")
    np.save(p / "x_train.npy", x_train)
    np.save(p / "x_test.npy", x_test)
    np.save(p / "y_train.npy", y_train)
    np.save(p / "train_ids.npy", train_ids)
    np.save(p / "test_ids.npy", test_ids)
    w, loss = mean_squared_error_gd(y_train, x_train, initial_w=np.zeros(x_train.shape[1]), max_iters=1000, gamma=0.7)
    logging.info(f"Loss (MSE) of the training set: {loss}")
    y_pred = predict_labels(x_test, w)
    create_csv_submission(test_ids, y_pred)
    logging.info("Submission file created.")

if __name__ == "__main__":
    x_train = np.load("clean_arrays/x_train.npy")
    x_test = np.load("clean_arrays/x_test.npy")
    y_train = np.load("clean_arrays/y_train.npy")
    train_ids = np.load("clean_arrays/train_ids.npy")
    test_ids = np.load("clean_arrays/test_ids.npy")
    w, loss = least_squares(y_train, x_train)
    print("Loss (MSE) of the training set:", loss)
    y_pred = predict_labels(x_test, w)
    create_csv_submission(test_ids, y_pred)


