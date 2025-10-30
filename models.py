"""
Machine Learning Models Module

This module provides object-oriented wrappers around the core ML implementations.
All models inherit from ModelBase and follow a consistent interface:
- fit(y, tx): Train the model on data
- predict(tx, threshold): Generate class predictions
- score(y, tx): Evaluate the model's loss
- get_metrics(y_true, y_predicted): Calculate performance metrics
- get_params(): Return model hyperparameters

Models included:
- LeastSquares: Linear regression via normal equation
- MSEGradientDescent: Linear regression via gradient descent
- RidgeRegression: Regularized linear regression
- LogisticRegressionGD: Logistic regression via gradient descent
- RegularizedLogisticRegressionGD: Regularized logistic regression
"""

import numpy as np
from abc import ABC, abstractmethod

from implementations import mean_squared_error_gd, mean_squared_error, logistic_regression, sigmoid, logistic_loss, \
    reg_logistic_regression, least_squares, ridge_regression


class ModelBase(ABC):
    """
    Abstract Base Class for all Machine Learning models.

    This class enforces a consistent interface across all ML models and provides
    shared functionality for prediction, evaluation, and metric calculation.
    All models operate on binary classification problems with labels {-1, 1}.

    Attributes:
        w: numpy array of shape (D,). The learned weight vector. None until fit() is called.
        loss: scalar. The final loss value after training. None until fit() is called.
    """

    def __init__(self):
        """Initializes the base model properties.

        Sets weights and loss to None. These will be populated during training.
        """
        self.w = None
        self.loss = None

    @abstractmethod
    def f(self, x: np.ndarray) -> np.ndarray:
        """Applies the model function to input data.

        This is the core prediction function that computes the raw model output
        (before applying any threshold for classification).

        Args:
            x: numpy array of shape (N, D). Input feature matrix.

        Returns:
            numpy array of shape (N,). Raw model predictions.
        """
        pass

    @abstractmethod
    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Trains the model with the given data.

        Learns the optimal weight vector w by minimizing the model's loss function.
        Updates self.w and self.loss.

        Args:
            y: numpy array of shape (N,). Target labels, always in {-1, 1}
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            w: numpy array of shape (D,). The learned weight vector.
            loss: scalar. The final loss value.
        """
        pass

    def predict(self, tx: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Generates class predictions (-1/1) for new data.

        Applies the model function f(x) and converts continuous outputs to
        binary class labels using the specified threshold.

        Args:
            tx: numpy array of shape (N, D). Input feature matrix.
            threshold: scalar. Decision boundary. Predictions <= threshold are
                      classified as -1, predictions > threshold as 1.

        Returns:
            numpy array of shape (N,). Class predictions in {-1, 1}.
        """
        y = self.f(tx)
        y = np.where(y <= threshold, -1.0, 1.0)
        return y

    @abstractmethod
    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Calculates the final loss (e.g., MSE or Logistic Loss) on the data.

        Args:
            y: numpy array of shape (N,). True labels.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            scalar. The loss value for the given data.
        """
        pass

    def get_metrics(self, y_true: np.ndarray, y_predicted: np.ndarray) -> dict[str, float]:
        """Calculates comprehensive classification performance metrics.

        Computes standard binary classification metrics based on the confusion matrix.
        Assumes labels are in {-1, 1} where -1 is negative class and 1 is positive class.

        Args:
            y_true: numpy array of shape (N,). Ground truth labels in {-1, 1}.
            y_predicted: numpy array of shape (N,). Predicted labels in {-1, 1}.

        Returns:
            dict with keys:
                - "Accuracy": (TP + TN) / N. Overall correctness.
                - "Recall": TP / (TP + FN). Sensitivity, true positive rate.
                - "False Positive Ratio": FP / (FP + TN). False alarm rate.
                - "Precision": TP / (TP + FP). Positive predictive value.
                - "F1 Score": Harmonic mean of precision and recall.
        """
        n = len(y_true)

        # Create mask for negative class (-1.0)
        zeros_mask = y_true == -1.0

        # Calculate confusion matrix components
        # True Positives: actual=1, predicted=1
        tp = np.sum(y_predicted[~zeros_mask] == 1.0).item()
        # False Negatives: actual=1, predicted=-1
        fn = np.sum(y_predicted[~zeros_mask] == -1.0).item()
        # True Negatives: actual=-1, predicted=-1
        tn = np.sum(y_predicted[zeros_mask] == -1.0).item()
        # False Positives: actual=-1, predicted=1
        fp = np.sum(y_predicted[zeros_mask] == 1.0).item()

        # Calculate metrics with safe division (avoid divide by zero)
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
        """Returns the dictionary of the model's hyperparameters.

        Extracts all non-private attributes except for 'w' and 'loss', which are
        learned parameters rather than hyperparameters.

        Returns:
            dict. Hyperparameter names mapped to their values.
                 Examples: {"max_iters": 100, "gamma": 0.1, "lambda_": 0.01}
        """
        # Inspect __init__ to get hyperparameters of the specific subclass
        return {
            key: value for key, value in self.__dict__.items()
            if not key.startswith('_') and key not in ['w', 'loss']
        }


class LeastSquares(ModelBase):
    """Linear Regression solved using the Normal Equation.

    Implements linear regression by solving the normal equation: w = (X^T X)^-1 X^T y
    This provides the optimal closed-form solution that minimizes Mean Squared Error.

    The model assumes a linear relationship: y = tx @ w

    Use this model when:
    - You have a small to medium-sized dataset (normal equation is O(D^3))
    - You want the exact optimal solution (no iterations needed)
    - Features are not highly collinear (X^T X should be invertible)
    """

    def f(self, x: np.ndarray) -> np.ndarray:
        """Applies the linear model to input data.

        Computes the linear transformation: f(x) = x @ w

        Args:
            x: numpy array of shape (N, D). Input feature matrix.

        Returns:
            numpy array of shape (N,). Linear predictions.

        Raises:
            RuntimeError: If the model has not been fitted yet (w is None).
        """
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return x @ self.w

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model using the Normal Equation.

        Solves the closed-form solution: w = (X^T X)^-1 X^T y
        This minimizes the Mean Squared Error loss.

        Args:
            y: numpy array of shape (N,). Target values.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            w: numpy array of shape (D,). Optimal weight vector.
            loss: scalar. Mean Squared Error of the optimal solution.
        """
        self.w, self.loss = least_squares(y, tx)
        return self.w, self.loss

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Calculates the Mean Squared Error (MSE) on the data.

        MSE = (1/2N) * sum((y - tx @ w)^2)

        Args:
            y: numpy array of shape (N,). True target values.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            scalar. The Mean Squared Error.

        Raises:
            RuntimeError: If the model has not been fitted yet (w is None).
        """
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return mean_squared_error(y, tx, self.w).item()

    def __str__(self):
        """String representation of the model."""
        return "LeastSquares()"


class MSEGradientDescent(LeastSquares):
    """Linear Regression solved using Gradient Descent (GD).

    Implements linear regression by iteratively updating weights in the direction
    that reduces Mean Squared Error. The update rule is: w = w - gamma * gradient

    The model assumes a linear relationship: y = tx @ w

    Use this model when:
    - You have a large dataset (GD is O(ND) per iteration)
    - You want to control the training process (iterations, step size)
    - You need an iterative approach (e.g., for online learning)

    Attributes:
        max_iters: int. Number of gradient descent iterations to perform.
        gamma: float. Learning rate (step size) for gradient updates.
        initial_w: numpy array of shape (D, ) or None. Initial weight vector.
                  If None, initializes to zeros.
    """

    def __init__(self, max_iters: int = 200, gamma: float = 0.1, initial_w: np.ndarray = None):
        """Initializes the Gradient Descent model.

        Args:
            max_iters: int. Total number of iterations of GD. Default: 100.
            gamma: float. Learning rate (step size). Default: 0.1.
            initial_w: numpy array of shape (D, ) or None. Initial weight vector.
                      If None, will be initialized to zeros during fit(). Default: None.
        """
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.initial_w = initial_w

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model using Gradient Descent.

        Iteratively updates weights: w = w - gamma * gradient
        where gradient = -(1/N) * X^T * (y - X @ w)

        Args:
            y: numpy array of shape (N,). Target values.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            w: numpy array of shape (D,). The learned weight vector after max_iters iterations.
            loss: scalar. Mean Squared Error using the final weight vector.
        """
        # Initialize weights to zeros if not provided
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])

        # Run gradient descent optimization
        self.w, self.loss = mean_squared_error_gd(
            y, tx, self.initial_w, self.max_iters, self.gamma
        )
        return self.w, self.loss

    def __str__(self):
        """String representation of the model."""
        return f"MSEGradientDescent(max_iters={self.max_iters}, gamma={self.gamma})"


class RidgeRegression(MSEGradientDescent):
    """Ridge Regression (L2 Regularized Linear Regression).

    Implements linear regression with L2 regularization to prevent overfitting.
    Solves the normal equation with a regularization term:
    w = (X^T X + 2*lambda*N*I)^-1 X^T y

    The regularized loss is: MSE + lambda * ||w||^2

    Use this model when:
    - Your data has multicollinearity (highly correlated features)
    - You want to prevent overfitting by penalizing large weights
    - You need a stable solution even with singular X^T X matrix

    Attributes:
        lambda_: float. Regularization parameter. Higher values = stronger regularization.
        max_iters: int. Inherited from parent (not used in ridge regression).
        gamma: float. Inherited from parent (not used in ridge regression).
        initial_w: numpy array. Inherited from parent (not used in ridge regression).
    """

    def __init__(self, lambda_: float = 0.1, **kwargs):
        """Initializes the Ridge Regression model.

        Args:
            lambda_: float. Regularization parameter (must be >= 0).
                    lambda=0 reduces to ordinary least squares.
                    Higher values increase regularization strength. Default: 0.1.
            **kwargs: Additional arguments passed to parent class (not used in practice).
        """
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model using Ridge Regression (closed-form solution).

        Solves: w = (X^T X + 2*lambda*N*I)^-1 X^T y
        where I is the identity matrix of size D x D.

        Args:
            y: numpy array of shape (N,). Target values.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            w: numpy array of shape (D,). The learned weight vector.
            loss: scalar. Mean Squared Error (without regularization term) using the learned weights.
        """
        # Initialize weights (not actually used, but keeps interface consistent)
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])

        # Solve ridge regression using closed-form solution
        self.w, self.loss = ridge_regression(y, tx, self.lambda_)
        return self.w, self.loss

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Calculates the Ridge Regression Loss (MSE + regularization term).

        Loss = MSE + lambda * ||w||^2
             = (1/2N) * sum((y - tx @ w)^2) + lambda * sum(w^2)

        Args:
            y: numpy array of shape (N,). True target values.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            scalar. The regularized loss value.

        Raises:
            RuntimeError: If the model has not been fitted yet (w is None).
        """
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        # MSE + L2 regularization penalty
        return super().score(y, tx) + self.lambda_ * np.sum(self.w ** 2)

    def __str__(self):
        """String representation of the model."""
        return f"RidgeRegression(lambda_={self.lambda_}, max_iters={self.max_iters}, gamma={self.gamma})"


class LogisticRegressionGD(ModelBase):
    """Logistic Regression solved using Gradient Descent (GD).

    Implements binary logistic regression by minimizing the logistic loss (negative log-likelihood)
    using gradient descent. The model applies the sigmoid function to linear predictions:
    P(y=1|x) = sigmoid(x @ w) = 1 / (1 + exp(-x @ w))

    The logistic loss is: L(w) = mean(-y * log(p) - (1-y) * log(1-p))
    where p = sigmoid(tx @ w)

    Use this model when:
    - You have a binary classification problem
    - You want probabilistic predictions
    - You need interpretable feature importance (via weights)

    Note: This model expects labels in {0, 1} internally but converts from {-1, 1}

    Attributes:
        max_iters: int. Number of gradient descent iterations to perform.
        gamma: float. Learning rate (step size) for gradient updates.
        initial_w: numpy array of shape (D,) or None. Initial weight vector.
                  If None, initializes to zeros.
    """

    def __init__(self, max_iters: int = 200, gamma: float = 0.7, initial_w: np.ndarray = None):
        """Initializes the Logistic Regression model.

        Args:
            max_iters: int. Total number of gradient descent iterations. Default: 100.
            gamma: float. Learning rate (step size). Default: 0.7.
                  Note: Logistic regression often uses higher learning rates than linear regression.
            initial_w: numpy array of shape (D,) or None. Initial weight vector.
                      If None, will be initialized to zeros during fit(). Default: None.
        """
        super().__init__()
        self.max_iters = max_iters
        self.gamma = gamma
        self.initial_w = initial_w

    def f(self, x: np.ndarray) -> np.ndarray:
        """Applies the logistic model to input data.

        Computes the sigmoid of the linear combination: f(x) = sigmoid(x @ w)
        This gives the predicted probability of the positive class.

        Args:
            x: numpy array of shape (N, D). Input feature matrix.

        Returns:
            numpy array of shape (N,). Predicted probabilities in [0, 1].

        Raises:
            RuntimeError: If the model has not been fitted yet (w is None).
        """
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return sigmoid(x @ self.w)

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model using Gradient Descent.

        Iteratively updates weights: w = w - gamma * gradient
        where gradient = (1/N) * X^T * (sigmoid(X @ w) - y)

        Args:
            y: numpy array of shape (N,). Target labels in {-1, 1}.
                   These are converted to {0, 1} internally for training.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            w: numpy array of shape (D,). The learned weight vector after max_iters iterations.
            loss: scalar. Logistic loss using the final weight vector.
        """
        # Initialize weights to zeros if not provided
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])

        # Convert labels from (-1, 1) to (0, 1) for logistic regression
        y = np.where(y == -1.0, 0.0, 1.0)

        # Run gradient descent optimization
        self.w, self.loss = logistic_regression(
            y, tx, self.initial_w, self.max_iters, self.gamma
        )
        return self.w, self.loss

    def predict(self, tx: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """Predicts labels (-1 or 1) using the fitted weights.

        Applies sigmoid to get probabilities, then classifies based on threshold.

        Args:
            tx: numpy array of shape (N, D). Input feature matrix.
            threshold: float or None. Decision boundary for classification.
                      Probabilities > threshold are classified as 1, otherwise -1.
                      If None, uses default threshold of 0.18. Default: None.

        Returns:
            numpy array of shape (N,). Class predictions in {-1, 1}.

        Note: The default threshold of 0.18 is tuned for this specific problem
              and may need adjustment for different datasets.
        """
        if threshold is None:
            print("Using default threshold of 0.18 for predictions.")
            threshold = 0.18
        return super().predict(tx, threshold)

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Calculates the Logistic Loss (negative log-likelihood).

        Loss = mean(-y * log(sigmoid(tx @ w)) - (1-y) * log(1 - sigmoid(tx @ w)))
             = mean(-y * (tx @ w) + log(1 + exp(tx @ w)))

        Args:
            y: numpy array of shape (N,). True labels in {-1, 1}.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            scalar. The logistic loss value.

        Raises:
            RuntimeError: If the model has not been fitted yet (w is None).
        """
        if self.w is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Convert labels from (-1, 1) to (0, 1) for loss calculation
        y = np.where(y == -1.0, 0.0, 1.0)
        return logistic_loss(y, tx, self.w).item()

    def __str__(self):
        """String representation of the model."""
        return f"LogisticRegressionGD(max_iters={self.max_iters}, gamma={self.gamma})"


class RegularizedLogisticRegressionGD(LogisticRegressionGD):
    """Regularized Logistic Regression solved using Gradient Descent (GD).

    Implements binary logistic regression with L2 regularization to prevent overfitting.
    Minimizes the regularized logistic loss using gradient descent.

    The regularized loss is: L(w) = logistic_loss(w) + lambda * ||w||^2
    where logistic_loss(w) = mean(-y * log(p) - (1-y) * log(1-p))
    and p = sigmoid(tx @ w)

    The gradient includes the regularization term:
    gradient = (1/N) * X^T * (sigmoid(X @ w) - y) + 2 * lambda * w

    Use this model when:
    - You have a binary classification problem with many features
    - You want to prevent overfitting by penalizing large weights
    - You need feature selection (large lambda shrinks unimportant weights toward zero)

    Note: This model expects labels in {0, 1} internally but converts from {-1, 1}

    Attributes:
        lambda_: float. Regularization parameter. Higher values = stronger regularization.
        max_iters: int. Inherited from parent. Number of gradient descent iterations.
        gamma: float. Inherited from parent. Learning rate (step size).
        initial_w: numpy array. Inherited from parent. Initial weight vector.
    """

    def __init__(self, lambda_: float = 0.1, **kwargs):
        """Initializes the Regularized Logistic Regression model.

        Args:
            lambda_: float. Regularization parameter (must be >= 0).
                    lambda=0 reduces to ordinary logistic regression.
                    Higher values increase regularization strength. Default: 0.1.
            **kwargs: Additional arguments passed to parent class (max_iters, gamma, initial_w).
        """
        super().__init__(**kwargs)
        self.lambda_ = lambda_

    def fit(self, y: np.ndarray, tx: np.ndarray):
        """Fits the model using Regularized Gradient Descent.

        Iteratively updates weights: w = w - gamma * (gradient + 2 * lambda * w)
        where gradient = (1/N) * X^T * (sigmoid(X @ w) - y)

        Args:
            y: numpy array of shape (N,). Target labels in {-1, 1}.
                   These are converted to {0, 1} internally for training.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            w: numpy array of shape (D,). The learned weight vector after max_iters iterations.
            loss: scalar. Logistic loss (without regularization term) using the final weight vector.
        """
        # Initialize weights to zeros if not provided
        if self.initial_w is None:
            self.initial_w = np.zeros(tx.shape[1])

        # Convert labels from (-1, 1) to (0, 1) for logistic regression
        y = np.where(y == -1.0, 0.0, 1.0)

        # Run regularized gradient descent optimization
        self.w, self.loss = reg_logistic_regression(
            y, tx, self.lambda_, self.initial_w, self.max_iters, self.gamma
        )
        return self.w, self.loss

    def score(self, y: np.ndarray, tx: np.ndarray) -> float:
        """Calculates the Regularized Logistic Loss.

        Loss = logistic_loss + lambda * ||w||^2
             = mean(-y * (tx @ w) + log(1 + exp(tx @ w))) + lambda * sum(w^2)

        Args:
            y: numpy array of shape (N,). True labels in {-1, 1}.
            tx: numpy array of shape (N, D). Input feature matrix.

        Returns:
            scalar. The regularized logistic loss value.

        Raises:
            RuntimeError: If the model has not been fitted yet (w is None).
        """
        # Logistic loss + L2 regularization penalty
        return super().score(y, tx) + self.lambda_ * np.sum(self.w ** 2)

    def __str__(self):
        """String representation of the model."""
        return f"RegularizedLogisticRegressionGD(lambda_={self.lambda_}, max_iters={self.max_iters}, gamma={self.gamma})"

