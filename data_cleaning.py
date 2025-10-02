import numpy as np
import csv
import os
from helpers import load_csv_data



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
    (x_train, x_test, y_train, train_ids, test_ids) = load_csv_data(dataset_path)
    x_train = np.nan_to_num(x_train, nan=0.0)
    x_train: np.ndarray = standardize(x_train[:, index])
    x_test = np.nan_to_num(x_test, nan=0.0)
    x_test: np.ndarray = standardize(x_test[:, index])
    return x_train, x_test, y_train, train_ids, test_ids


if __name__ == "__main__":
    dataset_path = "dataset"
    get_initial_data(dataset_path)