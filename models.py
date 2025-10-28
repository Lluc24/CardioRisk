import numpy as np
from abc import ABC, abstractmethod

from implementations import mean_squared_error_gd, mean_squared_error, logistic_regression, sigmoid, logistic_loss, \
    reg_logistic_regression, least_squares, ridge_regression


class ModelBase(ABC):
    """
    Abstract Base Class for all Machine Learning models.
    Enforces the fit, predict, and score interfaces.
    """

    def __init__(self):
        """Initializes the base model properties."""
        self.w = None
        self.loss = None

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """Applies the model function to input data."""
        pass

    @abstractmethod
    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Trains the model with the given data."""
        pass

    def predict(self, tx: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Generates class predictions (-1/1) for new data."""
        y = self.f(tx)
        y = np.where(y <= threshold, -1.0, 1.0)
        return y

    @abstractmethod
    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Calculates the final loss (e.g., MSE or Logistic Loss) on the data."""
        pass

    def get_metrics(self, y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
        """Returns a dictionary of the model's performance metrics."""
        n = len(y_true)
        zeros_mask = y_true == -1.0
        tp = np.sum(y_predicted[~zeros_mask] == 1.0).item()
        fn = np.sum(y_predicted[~zeros_mask] == -1.0).item()
        tn = np.sum(y_predicted[zeros_mask] == -1.0).item()
        fp = np.sum(y_predicted[zeros_mask] == 1.0).item()

        print("TP:", tp, "FN:", fn, "TN:", tn, "FP:", fp)

        accuracy = (tp + tn) / n  # proportion of total predictions that were correct
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # positive cases that were correctly identified
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # negative cases that were incorrectly classified as positive
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # positive predictions that were actually correct
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # harmonic mean of precision and recall
        return {
            "Accuracy": accuracy,
            "Recall": recall,
            "False Positive Ratio": fpr,
            "Precision": precision,
            "F1 Score": f1
        }

    def get_params(self) -> dict:
        """Returns the dictionary of the model's hyper-parameters."""
        # Inspect __init__ to get hyper-parameters of the specific subclass
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and key not in ['w', 'loss']
        }


class LeastSquares(ModelBase):
    """Linear Regression solved using the Normal Equation and Mean Squared Error loss function."""

    def f(self, x: np.ndarray) -> np.ndarray:
        """Applies the linear model to input data."""
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return x @ self.w

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model using the Normal Equation."""
        self.w, self.loss = least_squares(y, tx)
        return self.w, self.loss

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Returns the Mean Squared Error (MSE)."""
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return mean_squared_error(y, tx, self.w)

    def __str__(self):
        return "LeastSquares()"


class MSEGradientDescent(LeastSquares):
    """Linear Regression solved using Gradient Descent (GD) and Mean Squared Error loss function."""

    def __init__(self, max_iters: int = 100, gamma: float = 0.1, initial_w: np.ndarray = None):
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.initial_w = initial_w

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model by running GD."""
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])
        self.w, self.loss = mean_squared_error_gd(
            y, tx, self.initial_w, self.max_iters, self.gamma
        )
        return self.w, self.loss

    def __str__(self):
        return f"MSEGradientDescent(max_iters={self.max_iters}, gamma={self.gamma})"


class RidgeRegression(MSEGradientDescent):
    """Ridge Regression solved using Gradient Descent (GD) and Mean Squared Error loss function."""

    def __init__(self, lambda_: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model by running GD."""
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])
        self.w, self.loss = ridge_regression(y, tx, self.lambda_)
        return self.w, self.loss

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Returns the Ridge Regression Loss."""
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return super().score(y, tx) + self.lambda_ * np.sum(self.w ** 2)

    def __str__(self):
        return f"RidgeRegression(lambda_={self.lambda_}, max_iters={self.max_iters}, gamma={self.gamma})"


class LogisticRegressionGD(ModelBase):
    """Logistic Regression solved using Gradient Descent (GD) and Logistic Loss function."""

    def __init__(self, max_iters: int = 100, gamma: float = 0.7, initial_w: np.ndarray = None):
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.initial_w = initial_w

    def f(self, x: np.ndarray) -> np.ndarray:
        """Applies the logistic model to input data."""
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return sigmoid(x @ self.w)

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model by running GD."""
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])
        y = np.where(y == -1.0, 0.0, 1.0)  # Convert labels from (-1, 1) to (0, 1)
        self.w, self.loss = logistic_regression(
            y, tx, self.initial_w, self.max_iters, self.gamma
        )
        return self.w, self.loss

    def predict(self, tx: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predicts labels (-1 or 1) using the fitted weights."""
        if threshold is None:
            print("Using default threshold of 0.18 for predictions.")
            threshold = 0.18
        return super().predict(tx, threshold)

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Returns the Logistic Loss."""
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        y = np.where(y == -1.0, 0.0, 1.0)  # Convert labels from (-1, 1) to (0, 1)
        return logistic_loss(y, tx, self.w)

    def __str__(self):
        return f"LogisticRegressionGD(max_iters={self.max_iters}, gamma={self.gamma})"


class RegularizedLogisticRegressionGD(LogisticRegressionGD):
    """Regularized Logistic Regression solved using Gradient Descent (GD) and Logistic Loss function."""

    def __init__(self, lambda_: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model by running GD."""
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])
        y = np.where(y == -1.0, 0.0, 1.0)  # Convert labels from (-1, 1) to (0, 1)
        self.w, self.loss = reg_logistic_regression(
            y, tx, self.lambda_, self.initial_w, self.max_iters, self.gamma
        )
        return self.w, self.loss

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Returns the Regularized Logistic Loss."""
        return super().score(y, tx) + self.lambda_ * np.sum(self.w ** 2)

    def __str__(self):
        return f"RegularizedLogisticRegressionGD(lambda_={self.lambda_}, max_iters={self.max_iters}, gamma={self.gamma})"

