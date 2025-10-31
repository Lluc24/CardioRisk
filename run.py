"""
Main script for training and evaluating machine learning models on health survey data.

This script orchestrates the complete machine learning pipeline:
1. Data loading and preprocessing from CSV files
2. Model training using cross-validation
3. Threshold optimization for classification
4. Evaluation on local test set
5. Prediction on online test set and submission file generation

The pipeline uses metadata-driven data cleaning to handle missing values,
categorical encoding, and feature standardization.
"""

from dataset import Dataset
from helpers import create_csv_submission
from models import *
from data_cleaning import Data
from methods import cross_validate
import numpy as np

SEED = 1


def load_csv_and_save() -> None:
    """Load raw data from CSV files, clean it, and save to NumPy format.

    This function performs the complete data preprocessing pipeline:
    - Loads raw CSV files from the dataset directory
    - Applies metadata-driven cleaning (handle NaN, mappings, one-hot encoding)
    - Saves the cleaned data to a compressed NumPy file for faster future loading

    The cleaned data includes:
    - Training features (x_train) and labels (y_train)
    - Test features (x_test) and IDs (test_ids)
    - Metadata about continuous vs categorical features

    Side Effects:
        Creates 'cleaned_data.npz' file in the current directory.

    Note:
        This function only needs to be run once. After the first run,
        use load_numpy_file() to load the preprocessed data directly.
    """
    data = Data()
    data.load_from_csv("dataset", "metadata.json")
    data.save_to_numpy_file("cleaned_data.npz")


def load_numpy_file() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Load preprocessed data from NumPy file and prepare for training.

    Loads the cleaned dataset from disk and adds an intercept term
    (leading column of ones) to both training and test feature matrices.

    Returns:
        tuple containing:
            - x_train (np.ndarray): Training features with intercept, shape (N_train, D+1)
            - y_train (np.ndarray): Training labels in {-1, 1}, shape (N_train,)
            - x_test (np.ndarray): Test features with intercept, shape (N_test, D+1)
            - test_ids (np.ndarray): Test sample IDs for submission, shape (N_test,)
            - num_cont_features (int): Number of continuous (non-categorical) features

    Note:
        The intercept is added at column 0. Continuous features are at columns
        1 to num_cont_features+1, and categorical features follow after.
    """
    data = Data()
    data.load_from_numpy_file("cleaned_data.npz")
    # Optional: Uncomment to apply polynomial feature expansion
    # data.feature_expansion(degree=2)
    data.add_intercept()
    return data.x_train, data.y_train, data.x_test, data.test_ids, data.num_cont_features


def main() -> None:
    """Execute the complete ML pipeline: preprocessing, training, evaluation, and submission.

    This function performs the following steps:
    1. Load and preprocess data (create cleaned_data.npz if not exists)
    2. Initialize a Dataset wrapper for cross-validation
    3. Train a Logistic Regression model with cross-validation
    4. Optimize classification threshold using validation sets
    5. Average model parameters (weights, threshold, standardization stats) across folds
    6. Evaluate on local test set (from k-fold split)
    7. Generate predictions for online test set
    8. Create CSV submission file

    The standardization (mean and std) is computed separately for each fold
    during cross-validation and then averaged. The same averaged standardization
    is applied to both local and online test sets.

    Output Files:
        - cleaned_data.npz: Preprocessed data (if not already exists)
        - submissions/submission_YYYYMMDD_HHMMSS.csv: Predictions for online test

    Console Output:
        - Cross-validation metrics for each fold
        - Local test set performance metrics
        - Averaged cross-validation metrics
    """
    # Step 1: Load and preprocess data from CSV (only needed once)
    load_csv_and_save()

    # Step 2: Load preprocessed data and prepare for training
    x_train, y_train, x_test, test_ids, num_cont_features = load_numpy_file()

    # Step 3: Create dataset wrapper for k-fold cross-validation
    dataset = Dataset(x_train, y_train, num_cont_features, seed=SEED)

    # Step 4: Initialize model and perform cross-validation
    model = LogisticRegressionGD(max_iters=400, gamma=0.7)
    metrics = cross_validate(model, dataset, search_threshold_iterations=100)

    # Step 5: Extract and average model parameters from all folds
    # Average learned weights from k-fold cross-validation
    weights = np.mean(metrics.pop('Weights'), axis=0)
    # Average optimal threshold from k-fold cross-validation
    threshold = float(np.mean(metrics.pop('Thresholds')))
    # Average standardization parameters (mean and std) from k-fold cross-validation
    mean = np.mean(metrics.pop('Mean'), axis=0)
    std = np.mean(metrics.pop('Std'), axis=0)

    # Set the averaged weights to the model
    model.w = weights

    # Step 6: Evaluate on local test set (held-out from k-fold)
    x_test_local, y_test_local = dataset.get_test_set()
    # Standardize continuous features using averaged stats from cross-validation
    x_test_local[:, 1:num_cont_features+1] = (x_test_local[:, 1:num_cont_features+1] - mean) / std
    # Generate predictions using averaged threshold
    y_te_local_pred = model.predict(x_test_local, threshold=threshold)
    # Compute and display local test metrics
    local_metrics = model.get_metrics(y_test_local, y_te_local_pred)
    print("Local Test Set Metrics:", local_metrics)

    # Step 7: Predict on online test set for submission
    # Standardize continuous features using same stats as local test
    x_test[:, 1:num_cont_features+1] = (x_test[:, 1:num_cont_features+1] - mean) / std
    # Generate predictions using averaged threshold
    y_te_pred = model.predict(x_test, threshold=threshold)

    # Step 8: Create submission file with timestamp
    create_csv_submission(test_ids, y_te_pred)

    # Display averaged cross-validation metrics
    print(metrics)


if __name__ == '__main__':
    main()
