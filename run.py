from implementations import mean_squared_error_gd, predict_labels
from data_cleaning import get_initial_data, get_thresholded_data, add_intercept_column
import numpy as np
from helpers import create_csv_submission

if __name__ == '__main__':
    x_train, x_test, y_train, train_ids, test_ids = get_thresholded_data("dataset", 0.3)
    N, D = x_train.shape
    initial_w = np.zeros(D)
    max_iters = 50
    gamma = 0.7
    w, loss = mean_squared_error_gd(y_train, x_train, initial_w, max_iters, gamma)
    print(f"{w = }")
    print(f"{loss = }")

    predictions = predict_labels(x_test, w)
    print(f"{predictions = }")
    create_csv_submission(test_ids, predictions)
