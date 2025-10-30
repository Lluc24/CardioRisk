import numpy as np

from models import ModelBase
from dataset import Dataset

def cross_validate(model: ModelBase, dataset: Dataset, k_fold: int = 5, add_weights: bool = True, search_threshold_iterations: int = 0, threshold: float = 0.0) -> dict[str, list]:
    """Performs k-fold cross-validation and collects comprehensive metrics.

    Trains the model on k different train/validation splits and evaluates
    performance on each fold. Collects loss metrics, classification metrics,
    and optionally the learned weights from each fold.

    The function prints progress information for each fold and returns
    aggregated metrics across all folds for further analysis (e.g., computing
    mean and standard deviation).

    Args:
        model: ModelBase instance. The model to train and evaluate. Must implement
              fit(), score(), predict(), and get_metrics() methods.
        dataset: Dataset instance. Contains the training data to be split into
                k folds for cross-validation.
        k_fold: int. Number of folds for cross-validation. Default is 5.
               Each fold uses (k-1)/k of data for training and 1/k for validation.
        add_weights: bool. Whether to store learned weights from each fold.
                    Default is True. Useful for analyzing weight stability.

    Returns:
        dict with string keys and list values. Each list has k_fold elements:
            - "Train Loss": list of floats. Training loss for each fold.
            - "Validation Loss": list of floats. Validation loss for each fold.
            - "Accuracy": list of floats. Validation accuracy for each fold.
            - "Recall": list of floats. Validation recall for each fold.
            - "False Positive Ratio": list of floats. Validation FPR for each fold.
            - "Precision": list of floats. Validation precision for each fold.
            - "F1 Score": list of floats. Validation F1 score for each fold.
            - "Weights": list of numpy arrays (if add_weights=True). Learned weights
                        from each fold, each array of shape (D,).
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
        "Mean": [],
        "Std": [],
        "Thresholds": [],
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

        # Generate predictions and compute classification metrics
        if (search_threshold_iterations > 0):
            threshold = search_threshold(x_val, y_val, num_thresholds=search_threshold_iterations, model=model)

        y_pred = model.predict(x_val, threshold=threshold)
        fold_metrics = model.get_metrics(y_val, y_pred)

        # Store metrics for this fold
        metrics['Train Loss'].append(train_loss)
        metrics['Validation Loss'].append(val_loss)
        metrics["Mean"].append(mean)
        metrics["Std"].append(std)
        if add_weights:
            metrics["Weights"].append(w.copy())
        for key in fold_metrics:
            metrics[key].append(fold_metrics[key])

        # Display fold results
        print(f"Fold {k + 1}/{k_fold}")
        print(f"Train Loss = {train_loss:.6f}")
        print(f"Validation Loss = {val_loss:.6f}")
        print(f"Metrics: {fold_metrics}\n")

    print("\nCross-validation complete.")
    return metrics


def search_threshold(x_val, y_val, num_thresholds=100, model: ModelBase = None) -> float:
    """
    Searches for the best threshold to convert raw model predictions into binary class labels (-1 and 1)
    based on maximizing the F1 score.
    
    Args:
        y_raw_predictions (np.ndarray): Raw predictions from the model (continuous values).
        y_true (np.ndarray): True binary class labels (-1 and 1).
        num_thresholds (int): Number of thresholds to evaluate.

    Returns:
        float: The threshold that maximizes the F1 score.
    """
    best_threshold = 0.0
    best_f1_score = 0.0

    # Search thresholds from min to max of raw predictions
    thresholds = np.linspace(-1, 1, num=num_thresholds)
    
    for threshold in thresholds:
        # Convert raw predictions to binary labels based on the current threshold
        y_pred = model.predict(x_val, threshold=threshold)

        # Calculate F1 score
        fold_metrics = model.get_metrics(y_val, y_pred)
        current_f1_score = fold_metrics["F1 Score"]

        # Update best threshold if current F1 score is better
        if current_f1_score > best_f1_score:
            best_f1_score = current_f1_score
            best_threshold = threshold

    print(f"Best threshold found: {best_threshold} with F1 score: {best_f1_score}")
    return best_threshold