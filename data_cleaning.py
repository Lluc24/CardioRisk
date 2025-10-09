import numpy as np
import csv
import os
from helpers import load_csv_data

def set_nans_to(data: np.ndarray, value: float) -> np.ndarray:
    """Set NaN values in the input data to a specified value.

    Args:
        data: numpy array of shape=(num_samples, num_features)
        value: float value to replace NaNs with
    Returns:
        numpy array with NaNs replaced by the specified value, shape=(num_samples, num_features
    """
    result: np.ndarray = np.nan_to_num(data, nan=value)
    return result

def add_intercept_column(x: np.ndarray) -> np.ndarray:
    """Add an intercept column to the input data x

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        numpy array with an added intercept column, shape=(num_samples, num_features + 1)

    >>> add_intercept_column(np.array([[1, 2], [3, 4], [5, 6]]))
    array([[1., 1., 2.],
           [1., 3., 4.],
           [1., 5., 6.]])
    """
    num_samples: int = x.shape[0]
    intercept: np.ndarray = np.ones((num_samples, 1))
    result: np.ndarray = np.hstack((intercept, x))
    return result

def remove_homogeneous_columns(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    non_homogeneous_mask: np.ndarray = np.std(x_train, axis=0) != 0  # type: ignore
    x_train: np.ndarray = x_train[:, non_homogeneous_mask]
    x_test: np.ndarray = x_test[:, non_homogeneous_mask]
    return x_train, x_test


def standardize(x: np.ndarray) -> np.ndarray:
    """Stadartize the input data x

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        standartized data, shape=(num_samples, num_features)

    >>> standardize(np.array([[1, 2], [3, 4], [5, 6]]))
    array([[-1.22474487, -1.22474487],
           [ 0.        ,  0.        ],
           [ 1.22474487,  1.22474487]])
    """
    means: np.ndarray = np.mean(x, axis=0)
    stds: np.ndarray = np.std(x, axis=0)
    result: np.ndarray = (x - means) / stds
    return result

def get_headers(dataset_path: str) -> list[str]:
    with open(os.path.join(dataset_path, "x_train.csv"), 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
    return headers

def get_initial_data(dataset_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    headers: list[str] = get_headers(dataset_path)
    # Get the index of the 'GENHLTH' column
    index: int = headers.index('GENHLTH')
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)
    N = x_train.shape[0]

    # x_train
    x_train = set_nans_to(x_train, 0)
    x_train = x_train[:, index].reshape(-1, 1)  # By default, selecting one row transform the matrix to a vector
    x_train = standardize(x_train)
    x_train = add_intercept_column(x_train)

    # x_test
    x_test = set_nans_to(x_test, 0)
    x_test = x_test[:, index].reshape(-1, 1)
    x_test = standardize(x_test)
    x_test = add_intercept_column(x_test)

    return x_train, x_test, y_train, train_ids, test_ids

def filter_thresholded_columns(threshold: float, x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    nan_means = np.mean(np.isnan(x_train), axis=0)
    filtered_x_train = x_train[:, nan_means < threshold]
    filtered_x_test = x_test[:, nan_means < threshold]
    return filtered_x_train, filtered_x_test

def get_thresholded_data(dataset_path: str, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)
    x_train, x_test = filter_thresholded_columns(threshold, x_train, x_test)
    x_train, x_test = remove_homogeneous_columns(x_train, x_test)

    # x_train
    x_train = set_nans_to(x_train, 0)
    x_train = standardize(x_train)
    x_train = add_intercept_column(x_train)

    # x_test
    x_test = set_nans_to(x_test, 0)
    x_test = standardize(x_test)
    x_test = add_intercept_column(x_test)

    return x_train, x_test, y_train, train_ids, test_ids



if __name__ == "__main__":
    dataset_path = "dataset"
    get_initial_data(dataset_path)