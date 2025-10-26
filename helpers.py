"""Some helper functions for project 1."""

import csv
import numpy as np
import os
import pathlib
import datetime
import json
from typing import List
import logging

logger = logging.getLogger(__name__)


def load_csv_data(data_path, sub_sample=False):
    """
    This function loads the data and returns the respectinve numpy arrays.
    Remember to put the 3 files in the same folder and to not change the names of the files.

    Args:
        data_path (str): datafolder path
        sub_sample (bool, optional): If True the data will be subsempled. Default to False.

    Returns:
        x_train (np.array): training data
        x_test (np.array): test data
        y_train (np.array): labels for training data in format (-1,1)
        train_ids (np.array): ids of training data
        test_ids (np.array): ids of test data
    """
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # sub-sample
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def unique_name(func):
    dir = pathlib.Path("submissions")
    dir.mkdir(exist_ok=True)

    def wrapper(*args, **kwargs):
        prefix = kwargs.pop("prefix", "")
        prefix = prefix + "_" if prefix else ""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        name = str(dir / f"{prefix}submission_{timestamp}.csv")
        return func(*args, name=name, **kwargs)

    return wrapper


@unique_name
def create_csv_submission(ids, y_pred, *, name):
    """
    This function creates a csv file named 'name' in the format required for a submission in Kaggle or AIcrowd.
    The file will contain two columns the first with 'ids' and the second with 'y_pred'.
    y_pred must be a list or np.array of 1 and -1 otherwise the function will raise a ValueError.

    Args:
        ids (list,np.array): indices
        y_pred (list,np.array): predictions on data correspondent to indices
        name (str): name of the file to be created
    """
    # Check that y_pred only contains -1 and 1
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def process_metadata(json_path: str, headers: List[str]) -> List[dict]:
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
        logger.info(f"Still pending the features {pending_features}")
    logger.info(f"Processing metadata for {len(ids)} ids: {ids}")
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
