"""
Helper functions for machine learning project workflows.

This module provides utility functions for:
- Loading data from CSV files
- Creating submission files with unique timestamps
- Data subsetting for faster experimentation

All functions handle file I/O and data formatting for the ML pipeline.
"""

import csv
import numpy as np
import os
import pathlib
import datetime


def load_csv_data(data_path: str, sub_sample: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load training and test data from CSV files in the specified directory.

    Reads three CSV files (x_train.csv, x_test.csv, y_train.csv) from the given
    directory path and returns them as numpy arrays. The first column in each
    file is assumed to be the sample ID.

    Args:
        data_path (str): Path to the directory containing the CSV files.
                        Must contain 'x_train.csv', 'x_test.csv', and 'y_train.csv'.
        sub_sample (bool, optional): If True, subsamples the training data by
                                     selecting every 50th sample. Useful for
                                     faster experimentation. Default is False.

    Returns:
        tuple containing:
            - x_train (np.ndarray): Training features, shape (N_train, D) where
                                   N_train is number of samples, D is features.
            - x_test (np.ndarray): Test features, shape (N_test, D).
            - y_train (np.ndarray): Training labels in {-1, 1}, shape (N_train,).
            - train_ids (np.ndarray): Training sample IDs, shape (N_train,).
            - test_ids (np.ndarray): Test sample IDs, shape (N_test,).

    Note:
        The IDs (first column) are automatically extracted and returned separately
        from the feature matrices.
    """
    # Load training labels from CSV (only second column contains labels)
    y_train = np.genfromtxt(
        os.path.join(data_path, "y_train.csv"),
        delimiter=",",
        skip_header=1,
        dtype=int,
        usecols=1,
    )
    # Load training features from CSV
    x_train = np.genfromtxt(
        os.path.join(data_path, "x_train.csv"), delimiter=",", skip_header=1
    )
    # Load test features from CSV
    x_test = np.genfromtxt(
        os.path.join(data_path, "x_test.csv"), delimiter=",", skip_header=1
    )

    # Extract IDs (first column) and convert to integers
    train_ids = x_train[:, 0].astype(dtype=int)
    test_ids = x_test[:, 0].astype(dtype=int)
    # Remove ID column from feature matrices
    x_train = x_train[:, 1:]
    x_test = x_test[:, 1:]

    # Subsample data if requested (take every 50th sample for faster experiments)
    if sub_sample:
        y_train = y_train[::50]
        x_train = x_train[::50]
        train_ids = train_ids[::50]

    return x_train, x_test, y_train, train_ids, test_ids


def unique_name(func):
    """Decorator that generates unique timestamped filenames for submission files.

    This decorator wraps functions that create submission files, automatically
    generating a unique filename with a timestamp to prevent overwriting previous
    submissions. Files are saved in a 'submissions/' directory.

    Args:
        func: Function to decorate. Should accept 'name' as a keyword argument
              for the output filename.

    Returns:
        Wrapper function that generates timestamped filenames.

    Example:
        @unique_name
        def create_submission(ids, predictions, *, name):
            # Save to file with name
            pass

        # Calling create_submission() will auto-generate:
        # submissions/submission_20231031_143022.csv
        create_submission(ids, preds)

        # Or with a prefix:
        create_submission(ids, preds, prefix="logistic")
        # Creates: submissions/logistic_submission_20231031_143022.csv
    """
    # Create submissions directory if it doesn't exist
    dir = pathlib.Path("submissions")
    dir.mkdir(exist_ok=True)

    def wrapper(*args, **kwargs):
        # Extract optional prefix from kwargs
        prefix = kwargs.pop("prefix", "")
        prefix = prefix + "_" if prefix else ""
        # Generate timestamp string
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create unique filename
        name = str(dir / f"{prefix}submission_{timestamp}.csv")
        # Call original function with generated name
        return func(*args, name=name, **kwargs)

    return wrapper


@unique_name
def create_csv_submission(ids: np.ndarray, y_pred: np.ndarray, *, name: str) -> None:
    """Create a CSV submission file with predictions in competition format.

    Generates a CSV file with two columns (Id, Prediction) suitable for
    submission to Kaggle or AIcrowd competitions. The filename is automatically
    generated with a timestamp via the @unique_name decorator.

    Args:
        ids (np.ndarray): Array of sample IDs, shape (N,).
        y_pred (np.ndarray): Array of predictions in {-1, 1}, shape (N,).
        name (str): Output filename (automatically generated by decorator).

    Raises:
        ValueError: If y_pred contains values other than -1 or 1.

    Side Effects:
        Creates a CSV file in the 'submissions/' directory.

    Example:
        # Automatically creates: submissions/submission_20231031_143530.csv
        create_csv_submission(test_ids, predictions)

        # With prefix: submissions/best_submission_20231031_143530.csv
        create_csv_submission(test_ids, predictions, prefix="best")

    CSV Format:
        Id,Prediction
        1,-1
        2,1
        3,1
        ...
    """
    # Validate predictions are binary (-1 or 1)
    if not all(i in [-1, 1] for i in y_pred):
        raise ValueError("y_pred can only contain values -1, 1")

    # Write predictions to CSV file
    with open(name, "w", newline="") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})
