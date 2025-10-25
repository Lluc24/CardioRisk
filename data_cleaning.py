from typing import Callable
from helpers import process_metadata
import numpy as np
import csv
import os
from helpers import load_csv_data
import logging
import pathlib

from implementations import build_poly

NUMPY_ARRAYS_DIR = "clean_arrays"

logger = logging.getLogger(__name__)

def save_arrays(x_train: np.ndarray, x_test: np.ndarray, num_cont_features: np.ndarray, y_train: np.ndarray, train_ids: np.ndarray, test_ids: np.ndarray) -> None:
    arrays = (("x_train", x_train), ("x_test", x_test), ("num_cont_features", num_cont_features), ("y_train", y_train),
              ("train_ids", train_ids), ("test_ids", test_ids))
    p = pathlib.Path(NUMPY_ARRAYS_DIR)
    p.mkdir(parents=True, exist_ok=True)
    for name, array in arrays:
        array_path = p / f"{name}.npy"
        np.save(array_path, array)
        logger.info(f"Saved {name} array to {array_path}")

def load_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = ["x_train.npy", "x_test.npy", "num_cont_features.npy", "y_train.npy", "train_ids.npy", "test_ids.npy"]
    p = pathlib.Path(NUMPY_ARRAYS_DIR)
    arrays = tuple(np.load(p / name) for name in names)
    return arrays

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
    """Standardize the input data x

    Args:
        x: numpy array of shape=(num_samples, num_features)

    Returns:
        standardized data, shape=(num_samples, num_features)
    """
    means: np.ndarray = np.mean(x, axis=0)
    stds: np.ndarray = np.std(x, axis=0)
    result: np.ndarray = (x - means) / stds
    return result

def get_headers(dataset_path: str) -> list[str]:
    with open(os.path.join(dataset_path, "x_train.csv"), 'r') as file:
        reader = csv.reader(file)
        headers = next(reader)
    headers = headers[1:]  # Remove the 'id' column
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

def filter_thresholded_columns(threshold: float, x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nan_means = np.mean(np.isnan(x_train), axis=0)
    filtered_x_train = x_train[:, nan_means < threshold]
    filtered_x_test = x_test[:, nan_means < threshold]
    column_ids=np.array(np.linspace(0,(x_train.shape[1]-1),x_train.shape[1]))
    filtered_column_ids=column_ids[nan_means < threshold]
    return filtered_x_train, filtered_x_test, filtered_column_ids

def get_thresholded_data(dataset_path: str, threshold: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)
    x_train, x_test, columns_ids = filter_thresholded_columns(threshold, x_train, x_test)
    x_train, x_test = remove_homogeneous_columns(x_train, x_test)

    # x_train
    x_train = set_nans_to(x_train, 0)
    x_train = standardize(x_train)
    x_train = add_intercept_column(x_train)

    # x_test
    x_test = set_nans_to(x_test, 0)
    x_test = standardize(x_test)
    x_test = add_intercept_column(x_test)

    return x_train, x_test, y_train, train_ids, test_ids, columns_ids

def cleaning_pipeline(dataset_path: str, metadata_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_train, x_test, y_train, train_ids, test_ids = load_csv_data(dataset_path)
    headers: list[str] = get_headers(dataset_path)
    metadata = process_metadata(metadata_path, headers)

    num_cont_features = 0
    for feature in metadata:
        id = feature["id"]
        index = headers.index(id)
        logger.info(f"Processing feature {id}")

        if feature["type"] == "delete":
            x_train = np.delete(x_train, index, axis=1)
            x_test = np.delete(x_test, index, axis=1)
            headers.pop(index)
            logger.info(f"Deleted feature {id}, new shape: {x_train.shape}")
        elif feature["type"] == "continuous":
            x_train, x_test = solve_mapping(x_train, x_test, feature, index)
            num_cont_features += 1
            logger.info(f"Processed continuous feature {id}, new shape: {x_train.shape}, now {num_cont_features} continuous features processed")
        elif feature["type"] == "categorical":
            x_train, x_test = solve_mapping(x_train, x_test, feature, index)
            x_train, x_test = one_hot_encoding(x_train, x_test, feature, index)
            headers.pop(index)
            logger.info(f"Processed categorical feature {id}, new shape: {x_train.shape}")

    x_train, x_test = remove_homogeneous_columns(x_train, x_test)
    x_train[:, :num_cont_features] = standardize(x_train[:, :num_cont_features])
    x_test[:, :num_cont_features] = standardize(x_test[:, :num_cont_features])
    logger.info(f"Standardized columns 0 to {num_cont_features - 1}, final shape: {x_train.shape}")
    logger.info(f"Finished cleaning data")
    num_cont_features = np.array(num_cont_features)

    return x_train, x_test, num_cont_features, y_train, train_ids, test_ids

def solve_mapping(x_train: np.ndarray, x_test: np.ndarray, feature: dict, column_index: int) -> tuple[np.ndarray, np.ndarray]:
    if "mapping" in feature:
        mapping = feature["mapping"]
        for col in (x_train[:, column_index], x_test[:, column_index]):
            for [key, value] in mapping:
                mask = np.isnan(col) if np.isnan(key) else col == key
                if value == "mean":
                    mean_value = np.nanmean(col)
                    col[mask] = mean_value
                    logger.info(f"Mapping feature {feature['id']} value {key} to mean value {mean_value}")
                else:
                    col[mask] = value
                    logger.info(f"Mapping feature {feature['id']} value {key} to {value}")
    return x_train, x_test

def one_hot_encoding(x_train: np.ndarray, x_test: np.ndarray, feature: dict, column_index: int) -> tuple[np.ndarray, np.ndarray]:
    vocabulary: np.ndarray = feature["vocabulary"]
    logger.info(f"One-hot encoding feature {feature['id']} with vocabulary (len={len(vocabulary)}) {vocabulary}")
    updated = []
    for x in (x_train, x_test):
        unique = np.unique(x[:, column_index])
        included: Callable = lambda x: np.any(np.isnan(vocabulary)) if np.isnan(x) else x in vocabulary
        if not all(map(included, unique)):
            raise ValueError(f"Feature {feature['id']} contains values not in the vocabulary: {unique[~np.isin(unique, vocabulary)]}")
        inverse: np.ndarray = np.searchsorted(vocabulary, x[:, column_index])
        one_hot: np.ndarray = np.eye(len(vocabulary))[inverse]
        logger.info(f"One-hot encoded feature {feature['id']} into shape {one_hot.shape}")
        x = np.delete(x, column_index, axis=1)
        x = np.hstack((x, one_hot))
        updated.append(x)
    logger.info(f"One-hot encoding completed for feature {feature['id']}, new shape: {x_train.shape}")
    return updated[0], updated[1]


def prepare_arrays(x_tr, x_te, num_cont_features, degree):
    updated = []
    for array in [x_tr, x_te]:
        intercept = np.ones((array.shape[0], 1))
        poly = build_poly(array[:, :num_cont_features], degree)
        cat = array[:, num_cont_features:]
        new_array = np.hstack([intercept, poly, cat])
        updated.append(new_array)
    x_tr, x_val = updated
    return x_tr, x_val
