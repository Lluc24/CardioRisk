import numpy as np

from dataset import Dataset
from models import ModelBase


def cross_validate(model: ModelBase, dataset: Dataset, k_fold: int = 5):
    metrics = {
        'Train Loss': [],
        'Validation Loss': [],
        "Accuracy": [],
        "Recall": [],
        "False Positive Ratio": [],
        "Precision": [],
        "F1 Score": [],
        "Weights Norm": []
    }

    print(f"Starting {k_fold}-fold cross-validation for {model}")

    for k, (x_tr, y_tr, x_val, y_val) in enumerate(dataset.k_fold_generator(k_fold)):
        w, train_loss = model.fit(y_tr, x_tr)
        val_loss = model.score(y_val, x_val)

        y_pred = model.predict(x_val)
        fold_metrics = model.get_metrics(y_val, y_pred)

        metrics['Weights Norm'].append(np.linalg.norm(w))
        metrics['Train Loss'].append(train_loss)
        metrics['Validation Loss'].append(val_loss)
        for key in fold_metrics:
            metrics[key].append(fold_metrics[key])

        print(f"Fold {k + 1}/{k_fold}")
        print(f"Train Loss = {train_loss:.6f}")
        print(f"Validation Loss = {val_loss:.6f}")
        print(f"Metrics: {fold_metrics}\n")

    print("\nCross-validation complete.")
    return metrics
