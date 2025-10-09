import numpy as np
import csv
import os
from helpers import load_csv_data
import json

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

def get_data_columns(dataset_path: str, start: int, end: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)

    x_train = x_train[:, start:end]
    x_test = x_test[:, start:end]

    x_train = set_nans_to(x_train, 0)
    x_test = set_nans_to(x_test, 0)

    x_train, x_test = remove_homogeneous_columns(x_train, x_test)

    x_train = standardize(x_train)
    x_test = standardize(x_test)

    x_train = add_intercept_column(x_train)
    x_test = add_intercept_column(x_test)

    return x_train, x_test, y_train, train_ids, test_ids

def cleaning_pipeline(dataset_path: str, metadata="dataset_metadata.json") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)
    headers: list[str] = get_headers(dataset_path)

    x_train = clean_x_data(x_train, headers, metadata)
    x_test = clean_x_data(x_test, headers, metadata)



    return x_train, x_test, y_train, train_ids, test_ids

def read_json(path: str) ->  dict:
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def clean_x_data(x: np.ndarray, headers: list[str], metadata_path: str) -> np.ndarray:
    metadata = read_json(metadata_path)
    irrelevant_features = metadata["irrelevant_features"]
    few_values_features = metadata["few_values_features"]
    categorical_features = metadata["categorical_features"]
    pseudo_categorical_features = metadata["pseudo_categorical_features"]

    x, headers = clean_irrelevant_features(x, headers, irrelevant_features)
    x, headers = clean_few_values_features(x, headers, few_values_features)
    x = clean_categorical_features(x, headers, categorical_features)
    x = clean_continuous_features(x, headers, pseudo_categorical_features)

    x = standardize(x)
    x = add_intercept_column(x)
    return x

def clean_irrelevant_features(x: np.ndarray, headers: list[str], irrelevant_features: list[dict]) -> np.ndarray:
    for feature in irrelevant_features:
        id = feature["id"]
        index = headers.index(id)
        headers.pop(index)
        x = np.delete(x, index, axis=1)
    return x

def clean_few_values_features(x: np.ndarray, headers: list[str], few_values_features: list[dict]) -> np.ndarray:
    for feature in few_values_features:
        id = feature.pop("id")
        index = headers.index(id)

        num_nan_values = np.sum(np.isnan(x[:, index]))
        if x.shape[1] - num_nan_values != feature["values"]:
            raise ValueError(f"Feature {id} does not have {feature['values']} non-NaN values")

        x = np.delete(x, index, axis=1)
    return x

def clean_categorical_features(x: np.ndarray, headers: list[str], categorical_features: list[dict]) -> np.ndarray:
    for feature in categorical_features:
        id: str = feature["id"]
        index: int = headers.index(id)

        if "mapping" in feature:
            mapping: dict = feature["mapping"]
            x = solve_mapping(x, index, mapping)
        x = one_hot_encode_column(x, index)
    return x

def clean_continuous_features(x: np.ndarray, headers: list[str], continuous_features: list[dict]) -> np.ndarray:
    for feature in continuous_features:
        id: str = feature["id"]
        index: int = headers.index(id)

        if "mapping" in feature:
            mapping: dict = feature["mapping"]
            x = solve_mapping(x, index, mapping)
    return x

def encode_as_nan(x: np.ndarray, column_index: int, values: list[float]) -> np.ndarray:
    for value in values:
        x[:, column_index] = np.where(x[:, column_index] == value, np.nan, x[:, column_index])
    return x

def solve_mapping(x: np.ndarray, column_index: int, mapping: dict) -> np.ndarray:
    col = x[:, column_index]
    for key, value in mapping.items():
        mask = np.isnan(col) if key == "NaN" else col == float(key)
        col[mask] =  np.nan if value == "NaN" else value
    return x

def one_hot_encode_column(x: np.ndarray, column_index: int) -> np.ndarray:
    unique_values, inverse = np.unique(x[:, column_index], return_inverse=True)  # unique values are sorted in ascending order, where NaNs are placed at the end
    one_hot = np.eye(len(unique_values))[inverse]
    x = np.delete(x, column_index, axis=1)
    x = np.hstack((x, one_hot))
    return x