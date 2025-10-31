"""
Model evaluation and optimization methods.

This module provides high-level functions for:
- Cross-validation: K-fold splitting and evaluation
- Threshold optimization: Finding optimal classification thresholds
- Metrics collection: Comprehensive performance tracking

The methods are designed to work with any model that implements the
ModelBase interface and support standardization and threshold tuning.
"""

import numpy as np

from models import ModelBase
from dataset import Dataset

def cross_validate(model: ModelBase, dataset: Dataset, k_fold: int = 5, add_weights: bool = True, search_threshold_iterations: int = 0, threshold: float = 0.0) -> dict[str, list]:
    """Perform k-fold cross-validation and collect comprehensive metrics.

    Trains the model on k different train/validation splits and evaluates
    performance on each fold. The dataset is split into k parts; in each iteration,
    (k-1) parts are used for training and 1 part for validation. This provides
    a robust estimate of model performance and hyperparameter stability.

    Features:
    - Automatic data standardization per fold (prevents data leakage)
    - Optional threshold optimization on validation set
    - Comprehensive metric collection (loss, accuracy, precision, recall, F1)
    - Weight tracking for ensemble methods

    Args:
        model (ModelBase): The model to train and evaluate. Must implement
                          fit(), score(), predict(), and get_metrics() methods.
        dataset (Dataset): Training data container with k-fold split capability.
        k_fold (int, optional): Number of folds for cross-validation. Default is 5.
                               Each fold uses (k-1)/k data for training and 1/k for validation.
        add_weights (bool, optional): Whether to store learned weights from each fold.
                                     Default is True. Useful for weight averaging or
                                     analyzing weight stability across folds.
        search_threshold_iterations (int, optional): Number of threshold values to test
                                                     for finding optimal classification
                                                     threshold. If 0, uses default threshold.
                                                     Default is 0 (no threshold search).
        threshold (float, optional): Fixed classification threshold to use when
                                    search_threshold_iterations=0. Default is 0.0.

    Returns:
        dict[str, list]: Dictionary containing metrics from all folds. Keys:
            - "Train Loss": Training loss for each fold (list of k floats).
            - "Validation Loss": Validation loss for each fold (list of k floats).
            - "Accuracy": Validation accuracy for each fold (list of k floats).
            - "Recall": Validation recall (sensitivity) for each fold (list of k floats).
            - "False Positive Ratio": Validation FPR for each fold (list of k floats).
            - "Precision": Validation precision for each fold (list of k floats).
            - "F1 Score": Validation F1 score for each fold (list of k floats).
            - "Mean": Standardization mean from each fold (list of k arrays).
            - "Std": Standardization std from each fold (list of k arrays).
            - "Thresholds": Optimal/used threshold for each fold (list of k floats).
            - "Weights": Learned weights from each fold if add_weights=True
                        (list of k arrays of shape (D,)).

    Example:
        >>> model = LogisticRegressionGD(max_iters=400, gamma=0.7)
        >>> dataset = Dataset(x_train, y_train, num_cont_features)
        >>> # With threshold search
        >>> metrics = cross_validate(model, dataset, k_fold=5,
        ...                          search_threshold_iterations=100)
        >>> # Average metrics across folds
        >>> avg_f1 = np.mean(metrics['F1 Score'])
        >>> avg_weights = np.mean(metrics['Weights'], axis=0)

    Note:
        - Data standardization is computed separately for each fold to prevent
          data leakage from validation set into training.
        - Threshold search maximizes F1 score on the validation set.
        - Use the returned Mean and Std arrays to standardize test data consistently.
    """
    # Initialize metrics dictionary to collect results from all folds
    metrics = {
        'Train Loss': [],
        'Validation Loss': [],
        "Accuracy": [],
        "Recall": [],
        "False Positive Ratio": [],
        "Precision": [],
        "F1 Score": [],
        "Mean": [],  # Standardization mean for each fold
        "Std": [],   # Standardization std for each fold
        "Thresholds": [],  # Classification threshold for each fold
    }
    if add_weights:
        metrics["Weights"] = []

    print(f"Starting {k_fold}-fold cross-validation for {model}")

    # Iterate through each fold
    for k, (x_tr, y_tr, x_val, y_val, mean, std) in enumerate(dataset.k_fold_generator(k_fold)):
        # Train model on training fold
        w, train_loss = model.fit(y_tr, x_tr)

        # Evaluate on validation fold
        val_loss = model.score(y_val, x_val)

        # Optimize classification threshold if requested
        if search_threshold_iterations > 0:
            threshold = search_threshold(x_val, y_val, num_thresholds=search_threshold_iterations, model=model)

        # Generate predictions using the threshold
        y_pred = model.predict(x_val, threshold=threshold)
        # Compute classification metrics
        fold_metrics = model.get_metrics(y_val, y_pred)

        # Store metrics for this fold
        metrics['Train Loss'].append(train_loss)
        metrics['Validation Loss'].append(val_loss)
        metrics["Mean"].append(mean)
        metrics["Std"].append(std)
        metrics["Thresholds"].append(threshold)
        if add_weights:
            metrics["Weights"].append(w.copy())
        # Add all classification metrics (accuracy, precision, recall, F1, FPR)
        for key in fold_metrics:
            metrics[key].append(fold_metrics[key])

        # Display fold results
        print(f"Fold {k + 1}/{k_fold}")
        print(f"Train Loss = {train_loss:.6f}")
        print(f"Validation Loss = {val_loss:.6f}")
        print(f"Metrics: {fold_metrics}\n")

    print("\nCross-validation complete.")
    return metrics


def search_threshold(x_val: np.ndarray, y_val: np.ndarray, num_thresholds: int = 100, model: ModelBase = None) -> float:
    """Find the optimal classification threshold by maximizing F1 score.

    Tests multiple threshold values on the validation set to find the one
    that produces the best F1 score. This is useful for imbalanced datasets
    where the default threshold (0.0) may not be optimal.

    The function performs a grid search over thresholds in the range [-1, 1],
    evaluating each by converting model predictions to binary labels and
    computing the F1 score against true labels.

    Args:
        x_val (np.ndarray): Validation feature matrix, shape (N_val, D).
        y_val (np.ndarray): True validation labels in {-1, 1}, shape (N_val,).
        num_thresholds (int, optional): Number of threshold values to test.
                                       Higher values give finer granularity but
                                       increase computation time. Default is 100.
        model (ModelBase): Trained model instance with predict() and get_metrics()
                          methods. Used to generate predictions at each threshold.

    Returns:
        float: The threshold value that maximizes F1 score on the validation set.

    Algorithm:
        1. Generate num_thresholds evenly spaced values in [-1, 1]
        2. For each threshold:
           a. Generate binary predictions using model.predict(x_val, threshold)
           b. Compute F1 score against y_val
        3. Return threshold with highest F1 score

    Example:
        >>> model = LogisticRegressionGD(max_iters=400, gamma=0.7)
        >>> model.fit(y_train, x_train)
        >>> # Find optimal threshold on validation set
        >>> best_thresh = search_threshold(x_val, y_val, num_thresholds=100, model=model)
        >>> # Use optimal threshold for predictions
        >>> y_pred = model.predict(x_test, threshold=best_thresh)

    Note:
        - F1 score balances precision and recall, making it suitable for
          imbalanced datasets
        - The search range [-1, 1] covers typical model output ranges
        - Prints the best threshold and corresponding F1 score
    """
    best_threshold = 0.0
    best_f1_score = 0.0

    # Generate evenly spaced threshold candidates
    thresholds = np.linspace(-1, 1, num=num_thresholds)
    
    # Evaluate each threshold
    for threshold in thresholds:
        # Convert raw predictions to binary labels based on current threshold
        y_pred = model.predict(x_val, threshold=threshold)

        # Calculate F1 score for this threshold
        fold_metrics = model.get_metrics(y_val, y_pred)
        current_f1_score = fold_metrics["F1 Score"]

        # Update best threshold if current F1 score is better
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_threshold = threshold

    print(f"Best threshold found: {best_threshold:.4f} with F1 score: {best_f1_score:.4f}")
    return best_threshold