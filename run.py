from implementations import mean_squared_error_gd, predict_labels
from data_cleaning import get_initial_data, get_thresholded_data, get_data_columns
import numpy as np
from helpers import create_csv_submission
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    x_train, x_test, y_train, train_ids, test_ids = get_data_columns("dataset", 0, 106)

