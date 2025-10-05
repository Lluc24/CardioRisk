from implementations import mean_squared_error_gd
from data_cleaning import get_initial_data
import numpy as np

if __name__ == '__main__':
    x_train, x_test, y_train, train_ids, test_ids = get_initial_data("sub-dataset")
    initial_w = np.zeros(2)
    max_iters = 50
    gamma = 0.7
    w, loss = mean_squared_error_gd(y_train, x_train, initial_w, max_iters, gamma)
    print(f"{w = }")
    print(f"{loss = }")
