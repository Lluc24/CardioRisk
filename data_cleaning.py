from typing import Callable
import numpy as np
import csv
import os
from helpers import load_csv_data
import json
from copy import copy

from implementations import build_poly


class Data():

    def __init__(self):
        self.x_train: np.ndarray | None = None
        self.x_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.train_ids: np.ndarray | None = None
        self.test_ids: np.ndarray | None = None
        self.num_cont_features: int = 0
        self.headers: list[str] = []


    def load_from_csv(self, dataset_path: str, metadata_path: str):
        self.headers = get_headers(dataset_path)
        self._cleaning_pipeline(dataset_path, metadata_path)
        print(f"Loaded and cleaned data from {dataset_path} with metadata {metadata_path}, shape: {self.x_train.shape}")


    def add_intercept(self) -> None:
        intercept_train = np.ones((self.x_train.shape[0], 1))
        intercept_test = np.ones((self.x_test.shape[0], 1))
        self.x_train = np.hstack((intercept_train, self.x_train))
        self.x_test = np.hstack((intercept_test, self.x_test))
        print(f"Added intercept to data, new shape: {self.x_train.shape}")


    def feature_expansion(self, degree: int) -> None:
        x_train_poly = build_poly(self.x_train[:, :self.num_cont_features], degree)
        x_train_cat = self.x_train[:, self.num_cont_features:]
        self.x_train = np.hstack([x_train_poly, x_train_cat])
        x_test_poly = build_poly(self.x_test[:, :self.num_cont_features], degree)
        x_test_cat = self.x_test[:, self.num_cont_features:]
        self.x_test = np.hstack([x_test_poly, x_test_cat])
        print(f"Expanded features to degree {degree}, new shape: {self.x_train.shape}")


    def load_from_numpy_file(self, numpy_path: str):
        """Load data from a compressed numpy file."""
        data = np.load(numpy_path)
        self.x_train = data['x_train']
        self.x_test = data['x_test']
        self.y_train = data['y_train']
        self.train_ids = data['train_ids']
        self.test_ids = data['test_ids']
        self.num_cont_features = int(data['num_cont_features'])
        self.headers = data['headers'].tolist()
        print(f"Loaded data from {numpy_path}, shape: {self.x_train.shape}")


    def save_to_numpy_file(self, numpy_path: str):
        """Save all data to a compressed numpy file."""
        np.savez_compressed(
            numpy_path,
            x_train=self.x_train,
            x_test=self.x_test,
            y_train=self.y_train,
            train_ids=self.train_ids,
            test_ids=self.test_ids,
            num_cont_features=np.array([self.num_cont_features]),
            headers=np.array(self.headers)
        )
        print(f"Saved data to {numpy_path}")


    def _cleaning_pipeline(self, dataset_path: str, metadata_path: str) -> None:
        self.x_train, self.x_test, self.y_train, self.train_ids, self.test_ids = load_csv_data(dataset_path)
        metadata: list[dict] = process_metadata(metadata_path, self.headers)

        self.num_cont_features = 0
        for feature in metadata:
            id = feature["id"]
            index = self.headers.index(id)

            if feature["type"] == "delete":
                self.x_train = np.delete(self.x_train, index, axis=1)
                self.x_test = np.delete(self.x_test, index, axis=1)
                self.headers.pop(index)
                print(f"Deleted feature {id}, new shape: {self.x_train.shape}")
            elif feature["type"] == "continuous":
                self._solve_mapping(feature, index)
                self.num_cont_features += 1
                print(f"Processed continuous feature {id}, new shape: {self.x_train.shape}, now {self.num_cont_features} continuous features processed")
            elif feature["type"] == "categorical":
                self._solve_mapping(feature, index)
                self._one_hot_encoding(feature, index)
                self.headers.pop(index)
                print(f"Processed categorical feature {id}, new shape: {self.x_train.shape}")

        self._remove_homogeneous_columns()
        self.x_train[:, :self.num_cont_features] = standardize(self.x_train[:, :self.num_cont_features])
        self.x_test[:, :self.num_cont_features] = standardize(self.x_test[:, :self.num_cont_features])
        print(f"Standardized columns 0 to {self.num_cont_features - 1}, final shape: {self.x_train.shape}")
        print(f"Finished cleaning data")


    def _one_hot_encoding(self, feature: dict, column_index: int) -> None:
        vocabulary: np.ndarray = feature["vocabulary"]
        print(f"One-hot encoding feature {feature['id']} with vocabulary (len={len(vocabulary)}) {vocabulary}")
        updated = []
        for x in (self.x_train, self.x_test):
            unique = np.unique(x[:, column_index])
            included: Callable = lambda x: np.any(np.isnan(vocabulary)) if np.isnan(x) else x in vocabulary
            if not all(map(included, unique)):
                raise ValueError(
                    f"Feature {feature['id']} contains values not in the vocabulary: {unique[~np.isin(unique, vocabulary)]}")
            inverse: np.ndarray = np.searchsorted(vocabulary, x[:, column_index])
            one_hot: np.ndarray = np.eye(len(vocabulary))[inverse]
            print(f"One-hot encoded feature {feature['id']} into shape {one_hot.shape}")
            x = np.delete(x, column_index, axis=1)
            x = np.hstack((x, one_hot))
            updated.append(x)
        self.x_train, self.x_test = updated
        print(f"One-hot encoding completed for feature {feature['id']}, new shape: {self.x_train.shape}")


    def _remove_homogeneous_columns(self) -> None:
        non_homogeneous_mask = np.std(self.x_train, axis=0) != 0.0
        self.x_train = self.x_train[:, non_homogeneous_mask]
        self.x_test = self.x_test[:, non_homogeneous_mask]


    def _solve_mapping(self, feature: dict, column_index: int) -> None:
        if "mapping" in feature:
            mapping = feature["mapping"]
            for col in (self.x_train[:, column_index], self.x_test[:, column_index]):
                for [key, value] in mapping:
                    if "-" in str(key):
                        num1, num2 = tuple(map(float, key.split("-")))
                        mask = (col >= num1) & (col <= num2)
                    else:
                        mask = np.isnan(col) if np.isnan(key) else col == key
                    if value == "mean":
                        mean_value = np.nanmean(col)
                        col[mask] = mean_value
                        print(f"Mapping feature {feature['id']} value {key} to mean value {mean_value}")
                    else:
                        col[mask] = value
                        print(f"Mapping feature {feature['id']} value {key} to {value}")




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


def process_metadata(json_path: str, headers: list[str]) -> list[dict]:
    """
    Loads and validates the dataset metadata JSON file.
    Ensures each feature has 'id', 'description', and 'type'.
    Checks all ids are in headers.
    Validates and converts mappings: numbers to float, 'NaN' to np.nan, 'mean' to 'mean'.
    Returns the processed metadata as a list of dicts.
    Raises ValueError on validation failure.
    """
    with open(json_path, "r") as f:
        metadata = json.load(f)
    check_all_leaf_strings(metadata)

    ids = set()
    for feature in metadata:
        # Check required fields
        if not all(map(lambda x: x in feature, ("id", "description", "type"))):
            raise ValueError(f"Feature missing required fields {feature}")
        # Check correct type
        if feature["type"] not in ("continuous", "categorical", "delete"):
            raise ValueError(f"Feature '{feature['id']}' has invalid type '{feature['type']}'")
        # If type is categorical, check and process the vocabulary field
        if feature["type"] == "categorical":
            if "vocabulary" not in feature or not isinstance(feature["vocabulary"], list):
                raise ValueError(f"Feature '{feature['id']}' of type 'categorical' must have a 'vocabulary' field as a list.")

            vocabulary = np.array([np.nan if v == "NaN" else float(v) for v in feature["vocabulary"]])
            if len(vocabulary) != len(set(feature["vocabulary"])):
                raise ValueError(f"Vocabulary for feature {feature['id']} contains duplicate values: {feature["vocabulary"]}")
            feature["vocabulary"] = np.sort(vocabulary)

        # Check for duplicate ids
        if feature["id"] in ids:
            raise ValueError(f"Duplicate feature id found: '{feature['id']}'")
        ids.add(feature["id"])
        # Check id in headers
        if feature["id"] not in headers:
            raise ValueError(f"Feature id '{feature['id']}' not found in headers.")
        # Validate mapping if present
        if "mapping" in feature:
            mapping = feature["mapping"]
            if not isinstance(mapping, list):
                raise ValueError(f"Mapping for feature '{feature['id']}' is not a list.")
            for i, pair in enumerate(mapping):
                if not (isinstance(pair, list) and len(pair) == 2):
                    raise ValueError(f"Mapping entry {pair} in feature '{feature['id']}' is not a two-element list.")
                f = lambda x: x if "-" in x or x == "mean" else (np.nan if x == "NaN" else float(x))
                mapping[i] = list(map(f, pair))

    pending_features = set(headers) - ids
    if pending_features:
        print(f"Still pending the features {pending_features}")
    print(f"Processing metadata for {len(ids)} ids: {ids}")
    return metadata


def check_all_leaf_strings(obj, path=None):
    """
    Recursively checks that all dict keys and leaf values in a nested structure are strings.
    Raises ValueError with the path to the offending value if not.
    """
    path = path or []
    if isinstance(obj, dict):
        for k, v in obj.items():
            check_all_leaf_strings(k, path)
            check_all_leaf_strings(v, path + [str(k)])
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            check_all_leaf_strings(v, path + [f"[{i}]"])
    elif not isinstance(obj, str):
        raise ValueError(f"Non-string leaf at {'.'.join(path)}: {repr(obj)} (type {type(obj).__name__})")

