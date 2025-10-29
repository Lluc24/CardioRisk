"""
Data Cleaning and Preprocessing Module

This module provides a comprehensive pipeline for loading, cleaning, and preprocessing
machine learning datasets. It handles:
- Loading data from CSV files or compressed NumPy archives
- Feature mapping and transformation based on metadata
- One-hot encoding of categorical features
- Standardization of continuous features
- Feature expansion (polynomial features)
- Removal of homogeneous columns
- Adding intercept terms

The main class is Data, which encapsulates all preprocessing operations and maintains
training/test data in a consistent state.
"""

from typing import Callable, Any
import numpy as np
import csv
import os
from helpers import load_csv_data
import json

from implementations import build_poly


class Data:
    """
    Container for dataset preprocessing and feature engineering operations.

    This class manages the complete data cleaning pipeline from raw CSV data to
    preprocessed features ready for machine learning. It handles both continuous
    and categorical features, applying appropriate transformations based on metadata.

    The cleaning pipeline includes:
    1. Loading raw data from CSV files
    2. Applying feature-specific mappings (value replacements, range binning)
    3. One-hot encoding categorical features
    4. Removing homogeneous columns (zero variance)
    5. Standardizing continuous features (zero mean, unit variance)

    Final Dataset Structure:
        After processing, the feature matrix has a specific column ordering:
        - Left side: Continuous features (columns 0 to (num_cont_features-1), leftmost)
                    These preserve their relative order from the original data.
                    Can be expanded to polynomial features via feature_expansion().
        - Right side: One-hot encoded categorical features (rightmost columns)
                     Each categorical feature is converted to multiple binary columns.
                     Categorical features preserve their relative order from the original data.

    Attributes:
        x_train: numpy array of shape (N_train, D) or None. Training feature matrix.
        x_test: numpy array of shape (N_test, D) or None. Test feature matrix.
        y_train: numpy array of shape (N_train,) or None. Training labels.
        train_ids: numpy array of shape (N_train,) or None. Training sample IDs.
        test_ids: numpy array of shape (N_test,) or None. Test sample IDs.
        num_cont_features: int. Number of continuous features (before expansion).
                          Used to track which columns to standardize and expand.
                          Columns [0:num_cont_features] are continuous.
                          Columns [num_cont_features:] are one-hot encoded categorical.
        headers: list of str. Feature names corresponding to columns in x_train/x_test.
    """

    def __init__(self) -> None:
        """Initializes an empty Data container.

        All data attributes are set to None or default values. Use load_from_csv()
        or load_from_numpy_file() to populate the container with actual data.
        """
        self.x_train: np.ndarray | None = None
        self.x_test: np.ndarray | None = None
        self.y_train: np.ndarray | None = None
        self.train_ids: np.ndarray | None = None
        self.test_ids: np.ndarray | None = None
        self.num_cont_features: int = 0
        self.headers: list[str] = []


    def load_from_csv(self, dataset_path: str, metadata_path: str) -> None:
        """Loads and cleans data from CSV files using metadata-driven pipeline.

        Reads training and test data from CSV files, applies all cleaning transformations
        based on the metadata specification, and stores the cleaned data in the object.

        This method orchestrates the entire preprocessing pipeline:
        - Loads raw CSV data (x_train, x_test, y_train, train_ids, test_ids)
        - Processes features according to metadata (delete, continuous, categorical)
        - Applies mappings and transformations
        - One-hot encodes categorical features
        - Removes homogeneous columns
        - Standardizes continuous features

        Args:
            dataset_path: str. Path to directory containing CSV files:
                         x_train.csv, x_test.csv, y_train.csv.
            metadata_path: str. Path to JSON file containing feature metadata
                          (types, mappings, vocabularies).

        Side Effects:
            Populates all instance attributes (x_train, x_test, y_train, etc.)
            and prints progress information to stdout.
        """
        self.headers = get_headers(dataset_path)
        self._cleaning_pipeline(dataset_path, metadata_path)
        print(f"Loaded and cleaned data from {dataset_path} with metadata {metadata_path}, shape: {self.x_train.shape}")


    def add_intercept(self) -> None:
        """Adds a leading column of ones (intercept term) to feature matrices.

        Prepends a column of 1.0 values to both x_train and x_test, enabling
        models to learn a bias/intercept term. This transforms the feature
        matrices from shape (N, D) to (N, D+1).

        The intercept column is always the first column (index 0).

        Side Effects:
            Modifies x_train and x_test in-place by adding a leading column.
            Prints the new shape to stdout.
        """
        intercept_train: np.ndarray = np.ones((self.x_train.shape[0], 1))
        intercept_test: np.ndarray = np.ones((self.x_test.shape[0], 1))
        self.x_train = np.hstack((intercept_train, self.x_train))
        self.x_test = np.hstack((intercept_test, self.x_test))
        print(f"Added intercept to data, new shape: {self.x_train.shape}")


    def feature_expansion(self, degree: int) -> None:
        """Expands continuous features into polynomial features up to specified degree.

        Applies polynomial feature expansion (e.g., x, x^2, x^3, ...) to all continuous
        features while leaving categorical (one-hot encoded) features unchanged.
        This allows models to capture non-linear relationships.

        The expansion is applied only to the first num_cont_features columns (left side),
        which contain the continuous features. Categorical features (right side, the
        remaining columns) are kept unchanged and remain on the right after expansion.

        Args:
            degree: int. Maximum polynomial degree for feature expansion.
                   degree=1 keeps features unchanged, degree=2 adds squared terms, etc.

        Side Effects:
            Modifies x_train and x_test in-place, changing their shapes from
            (N, num_cont_features + num_cat_features) to
            (N, expanded_cont_features + num_cat_features), where expanded_cont_features = num_cont_features * degree
            Continuous features remain on the left, categorical on the right.
            Prints the new shape to stdout.

        Example:
            If num_cont_features=3 and degree=2:
            Before: [x1, x2, x3, cat1, cat2] (shape: N, 5)
            After:  [x1, x1^2, x2, x2^2, x3, x3^2, cat1, cat2]
                    |<---- continuous (left) --->||<- categorical (right) ->|
        """
        # Expand continuous features (first num_cont_features columns)
        x_train_poly: np.ndarray = build_poly(self.x_train[:, :self.num_cont_features], degree)
        x_test_poly: np.ndarray = build_poly(self.x_test[:, :self.num_cont_features], degree)

        # Keep categorical features unchanged (remaining columns)
        x_train_cat: np.ndarray = self.x_train[:, self.num_cont_features:]
        x_test_cat: np.ndarray = self.x_test[:, self.num_cont_features:]

        # Concatenate expanded continuous and categorical features
        self.x_train = np.hstack([x_train_poly, x_train_cat])
        self.x_test = np.hstack([x_test_poly, x_test_cat])
        print(f"Expanded features to degree {degree}, new shape: {self.x_train.shape}")


    def load_from_numpy_file(self, numpy_path: str) -> None:
        """Loads preprocessed data from a compressed NumPy archive file.

        Reads a .npz file containing previously cleaned and preprocessed data,
        restoring all dataset attributes. This is much faster than re-running
        the entire cleaning pipeline from CSV files.

        The .npz file must contain the following arrays:
        - x_train, x_test, y_train, train_ids, test_ids
        - num_cont_features (scalar stored as array)
        - headers (array of strings)

        Args:
            numpy_path: str. Path to .npz file created by save_to_numpy_file().

        Side Effects:
            Populates all instance attributes from the loaded file.
            Prints the loaded data shape to stdout.
        """
        data: np.lib.npyio.NpzFile = np.load(numpy_path)
        self.x_train = data['x_train']
        self.x_test = data['x_test']
        self.y_train = data['y_train']
        self.train_ids = data['train_ids']
        self.test_ids = data['test_ids']
        self.num_cont_features = int(data['num_cont_features'])
        self.headers = data['headers'].tolist()
        print(f"Loaded data from {numpy_path}, shape: {self.x_train.shape}")


    def save_to_numpy_file(self, numpy_path: str) -> None:
        """Saves all preprocessed data to a compressed NumPy archive file.

        Writes all dataset attributes to a .npz file for fast loading later.
        This avoids re-running the time-consuming cleaning pipeline.

        The saved file contains:
        - x_train, x_test, y_train, train_ids, test_ids
        - num_cont_features
        - headers

        Args:
            numpy_path: str. Path where .npz file will be created.
                       Typically ends with .npz extension.

        Side Effects:
            Creates a compressed NumPy archive file at the specified path.
            Prints confirmation message to stdout.
        """
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
        """Executes the complete data cleaning and preprocessing pipeline.

        This is the core internal method that orchestrates all cleaning operations
        in the correct sequence. It processes features according to metadata,
        applying type-specific transformations.

        Pipeline steps:
        1. Load raw CSV data
        2. Process each feature according to its type:
           - delete: Remove the column entirely
           - continuous: Apply mappings, track for standardization
           - categorical: Apply mappings, one-hot encode
        3. Remove homogeneous (zero-variance) columns
        4. Standardize continuous features to zero mean and unit variance

        Final column ordering:
        - Continuous features are placed on the left (columns 0 to num_cont_features-1)
        - One-hot encoded categorical features are placed on the right (remaining columns)
        - Both preserve their relative order from the original metadata

        Args:
            dataset_path: str. Path to directory containing CSV files.
            metadata_path: str. Path to JSON metadata file.

        Side Effects:
            Populates and modifies x_train, x_test, y_train, train_ids, test_ids,
            num_cont_features, and headers attributes.
            Prints detailed progress information to stdout.
        """
        # Load raw CSV data
        self.x_train, self.x_test, self.y_train, self.train_ids, self.test_ids = load_csv_data(dataset_path)

        # Load and validate metadata
        metadata: list[dict] = process_metadata(metadata_path, self.headers)

        # Process each feature according to its type
        self.num_cont_features = 0
        for feature in metadata:
            feature_id: str = feature["id"]
            index: int = self.headers.index(feature_id)

            if feature["type"] == "delete":
                # Remove unwanted features entirely
                self.x_train = np.delete(self.x_train, index, axis=1)
                self.x_test = np.delete(self.x_test, index, axis=1)
                self.headers.pop(index)
                print(f"Deleted feature {feature_id}, new shape: {self.x_train.shape}")

            elif feature["type"] == "continuous":
                # Process continuous features: apply mappings
                self._solve_mapping(feature, index)
                self.num_cont_features += 1
                print(f"Processed continuous feature {feature_id}, new shape: {self.x_train.shape}, now {self.num_cont_features} continuous features processed")

            elif feature["type"] == "categorical":
                # Process categorical features: apply mappings and one-hot encode
                self._solve_mapping(feature, index)
                self._one_hot_encoding(feature, index)
                header = self.headers.pop(index)
                self.headers.append(header)
                print(f"Processed categorical feature {feature_id}, new shape: {self.x_train.shape}")

        # Remove columns with zero variance (all values equal)
        self._remove_homogeneous_columns()

        # Standardize continuous features (first num_cont_features columns)
        self.x_train[:, :self.num_cont_features] = standardize(self.x_train[:, :self.num_cont_features])
        self.x_test[:, :self.num_cont_features] = standardize(self.x_test[:, :self.num_cont_features])
        print(f"Standardized columns 0 to {self.num_cont_features - 1}, final shape: {self.x_train.shape}")
        print(f"Finished cleaning data")


    def _one_hot_encoding(self, feature: dict, column_index: int) -> None:
        """Converts a categorical feature column into one-hot encoded columns.

        Replaces a single categorical column with multiple binary columns (one per
        category). Each row will have exactly one 1.0 and the rest 0.0, indicating
        which category that sample belongs to.

        The vocabulary must be sorted (enforced by process_metadata), which allows
        using np.searchsorted for efficient index lookup.

        Args:
            feature: dict. Feature metadata containing:
                    - 'id': Feature name
                    - 'vocabulary': Sorted numpy array of all possible category values
            column_index: int. Index of the column to encode in x_train and x_test.

        Side Effects:
            Modifies x_train and x_test:
            - Removes the original categorical column
            - Appends len(vocabulary) new binary columns
            Prints progress information to stdout.

        Raises:
            ValueError: If the data contains values not in the vocabulary.

        Example:
            If vocabulary = [1.0, 2.0, 3.0] and column = [2.0, 1.0, 3.0]:
            Original: [[2.0], [1.0], [3.0]]
            Encoded:  [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
        """
        vocabulary: np.ndarray = feature["vocabulary"]
        print(f"One-hot encoding feature {feature['id']} with vocabulary (len={len(vocabulary)}) {vocabulary}")

        updated: list[np.ndarray] = []
        for x in (self.x_train, self.x_test):
            # Validate that all values in the column are in the vocabulary
            unique: np.ndarray = np.unique(x[:, column_index])
            included: Callable[[float], bool] = lambda val: np.any(np.isnan(vocabulary)) if np.isnan(val) else val in vocabulary
            if not all(map(included, unique)):
                invalid_values: np.ndarray = unique[~np.isin(unique, vocabulary)]
                raise ValueError(
                    f"Feature {feature['id']} contains values not in the vocabulary: {invalid_values}")

            # Find index of each value in the sorted vocabulary
            inverse: np.ndarray = np.searchsorted(vocabulary, x[:, column_index])

            # Create one-hot encoded matrix
            one_hot: np.ndarray = np.eye(len(vocabulary))[inverse]
            print(f"One-hot encoded feature {feature['id']} into shape {one_hot.shape}")

            # Remove original categorical column
            x = np.delete(x, column_index, axis=1)

            # Append one-hot encoded columns at the end
            x = np.hstack((x, one_hot))
            updated.append(x)

        # Update instance attributes with modified arrays
        self.x_train, self.x_test = updated
        print(f"One-hot encoding completed for feature {feature['id']}, new shape: {self.x_train.shape}")


    def _remove_homogeneous_columns(self) -> None:
        """Removes columns with zero variance (all values identical).

        Identifies and removes features where all samples have the same value
        (standard deviation = 0). Such features provide no discriminative information
        for machine learning and can cause numerical issues.

        The removal is based on the training set variance. The same columns are
        removed from both training and test sets to maintain alignment.

        Side Effects:
            Modifies x_train and x_test by removing homogeneous columns.

        Example:
            If column 3 has values [5.0, 5.0, 5.0, ...] in training set,
            it will be removed from both x_train and x_test.
        """
        # Compute standard deviation for each column in training set
        non_homogeneous_mask: np.ndarray = np.std(self.x_train, axis=0) != 0.0

        # Remove columns with zero variance from both train and test
        self.x_train = self.x_train[:, non_homogeneous_mask]
        self.x_test = self.x_test[:, non_homogeneous_mask]


    def _solve_mapping(self, feature: dict, column_index: int) -> None:
        """Applies value mappings to a feature column based on metadata.

        Processes feature-specific transformations such as:
        - Replacing specific values (e.g., NaN -> 0, 7 -> 1)
        - Replacing value ranges (e.g., "50-100" -> 2)
        - Replacing with mean value (e.g., missing values -> column mean)

        Mappings are applied in order, which matters when transformations overlap
        (e.g., mapping NaN to 2 then 7 to NaN gives different results than reverse order).

        Args:
            feature: dict. Feature metadata containing optional 'mapping' field.
                    'mapping' is a list of [key, value] pairs where:
                    - key: str range ("50-100"), float, or np.nan
                    - value: float or "mean" (compute column mean)
            column_index: int. Index of the column to transform in x_train and x_test.

        Side Effects:
            Modifies x_train and x_test columns in-place.
            Prints each mapping operation to stdout.

        Example:
            mapping = [["NaN", "mean"], ["50-100", 2.0]]
            - First replaces NaN values with the column mean
            - Then replaces values in range [50, 100] with 2.0
        """
        if "mapping" not in feature:
            return

        mapping: list[list] = feature["mapping"]

        # Apply mappings to both train and test sets
        for col in (self.x_train[:, column_index], self.x_test[:, column_index]):
            for key, value in mapping:
                # Determine which rows match the key
                if "-" in str(key):
                    # Range mapping: "50-100" -> all values in [50, 100]
                    num1: float
                    num2: float
                    num1, num2 = tuple(map(float, key.split("-")))
                    mask: np.ndarray = (col >= num1) & (col <= num2)
                else:
                    # Single value mapping: handles NaN or specific numbers
                    mask: np.ndarray = np.isnan(col) if np.isnan(key) else col == key

                # Apply the mapped value
                if value == "mean":
                    # Replace with column mean (ignoring NaN values)
                    mean_value: np.generic = np.nanmean(col)
                    col[mask] = mean_value
                    print(f"Mapping feature {feature['id']} value {key} to mean value {mean_value}")
                else:
                    # Replace with specified value
                    col[mask] = value
                    print(f"Mapping feature {feature['id']} value {key} to {value}")




def standardize(x: np.ndarray) -> np.ndarray:
    """Standardizes data to zero mean and unit variance.

    Applies z-score normalization: (x - mean) / std for each feature column.
    This ensures all features have the same scale, which improves convergence
    for gradient-based optimization methods.

    Args:
        x: numpy array of shape (N, D). Input data matrix where N is number
           of samples and D is number of features.

    Returns:
        numpy array of shape (N, D). Standardized data where each column
        has mean ≈ 0 and standard deviation ≈ 1.
    """
    means: np.ndarray = np.mean(x, axis=0)
    stds: np.ndarray = np.std(x, axis=0)
    result: np.ndarray = (x - means) / stds
    return result


def get_headers(dataset_path: str) -> list[str]:
    """Extracts feature column headers from the training CSV file.

    Reads the first row of x_train.csv and returns the feature names,
    excluding the 'id' column which is always the first column.

    Args:
        dataset_path: str. Path to directory containing x_train.csv.

    Returns:
        list of str. Feature names in the order they appear in the CSV,
        without the 'id' column.

    Example:
        If x_train.csv header is: "id,age,weight,height"
        Returns: ["age", "weight", "height"]
    """
    with open(os.path.join(dataset_path, "x_train.csv"), 'r') as file:
        reader = csv.reader(file)
        headers: list[str] = next(reader)
    headers = headers[1:]  # Remove the 'id' column
    return headers


def process_metadata(json_path: str, headers: list[str]) -> list[dict]:
    """Loads and validates the dataset metadata JSON file.

    Reads a JSON file containing feature metadata and validates its structure
    and content. Ensures all required fields are present, all values are valid,
    and converts string representations to appropriate types (float, np.nan).

    Validation checks:
    - All leaf values and dict keys must be strings (before conversion)
    - Each feature must have 'id', 'description', and 'type' fields
    - Type must be one of: 'continuous', 'categorical', 'delete'
    - All feature IDs must exist in the headers list
    - All features in headers must be present in metadata
    - No duplicate feature IDs
    - Categorical features must have a 'vocabulary' list
    - Vocabularies must be sorted and contain no duplicates
    - Mappings must be lists of two-element lists
    - Mapping values must be numbers, "NaN", "mean", or ranges ("50-100")

    Args:
        json_path: str. Path to JSON metadata file.
        headers: list of str. Feature names from the CSV header row.
                Used to validate that all feature IDs exist.

    Returns:
        list of dict. Validated and processed feature metadata where:
            - Numbers are converted to float
            - "NaN" strings are converted to np.nan
            - "mean" strings are kept as "mean"
            - Vocabularies are converted to sorted numpy arrays
            - Range strings ("50-100") are kept as strings

    Raises:
        ValueError: If any validation check fails (missing fields, invalid types,
                   duplicate IDs, values not in vocabulary, etc.)
    """
    # Load JSON file
    with open(json_path, "r") as f:
        metadata: list[dict] = json.load(f)

    # Validate that all leaf values and keys are strings
    check_all_leaf_strings(metadata)

    ids: set[str] = set()
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

            # Convert vocabulary strings to floats/NaN and create numpy array
            vocabulary: np.ndarray = np.array([np.nan if v == "NaN" else float(v) for v in feature["vocabulary"]])

            # Check for duplicates
            if len(vocabulary) != len(set(feature["vocabulary"])):
                raise ValueError(f"Vocabulary for feature {feature['id']} contains duplicate values: {feature['vocabulary']}")

            # Sort vocabulary and store as numpy array
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
            mapping: list[list[str]] = feature["mapping"]
            if not isinstance(mapping, list):
                raise ValueError(f"Mapping for feature '{feature['id']}' is not a list.")

            for i, pair in enumerate(mapping):
                if not (isinstance(pair, list) and len(pair) == 2):
                    raise ValueError(f"Mapping entry {pair} in feature '{feature['id']}' is not a two-element list.")

                # Convert string representations to appropriate types
                # Keep ranges ("50-100") and "mean" as strings
                # Convert numbers to float, "NaN" to np.nan
                f: Callable[[str], float | str] = lambda x: x if "-" in x or x == "mean" else (np.nan if x == "NaN" else float(x))
                mapping[i] = list(map(f, pair))

    # Check all features are processed
    pending_features: set[str] = set(headers) - ids
    if pending_features:
        raise ValueError(f"The following features are missing in metadata: {pending_features}")

    return metadata


def check_all_leaf_strings(obj: Any, path: list[str] | None = None) -> None:
    """Recursively validates that all dict keys and leaf values are strings.

    Traverses a nested data structure (dicts and lists) to ensure that:
    - All dictionary keys are strings
    - All leaf values (non-dict, non-list values) are strings

    This is used to validate JSON metadata before type conversion, ensuring
    that all numeric values are provided as strings (e.g., "123" not 123)
    so they can be explicitly converted to the correct type.

    Args:
        obj: Any. The object to validate. Can be dict, list, str, or other types.
        path: list of str or None. Internal parameter tracking the current path
             through the nested structure for error reporting. Default is None.

    Raises:
        ValueError: If any dict key or leaf value is not a string.
                   The error message includes the path to the problematic value.

    Example:
        >>> data = {"id": "age", "value": "123"}  # Valid
        >>> check_all_leaf_strings(data)

        >>> bad_data = {"id": "age", "value": 123}  # Invalid (number not string)
        >>> check_all_leaf_strings(bad_data)
        ValueError: Non-string leaf at value: 123 (type int)
    """
    path = path or []

    if isinstance(obj, dict):
        # Recursively check all keys and values in the dictionary
        for k, v in obj.items():
            check_all_leaf_strings(k, path)  # Keys must be strings
            check_all_leaf_strings(v, path + [str(k)])  # Values checked recursively

    elif isinstance(obj, list):
        # Recursively check all elements in the list
        for i, v in enumerate(obj):
            check_all_leaf_strings(v, path + [f"[{i}]"])

    elif not isinstance(obj, str):
        # Leaf value that is not a string - this is an error
        raise ValueError(f"Non-string leaf at {'.'.join(path)}: {repr(obj)} (type {type(obj).__name__})")

