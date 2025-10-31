"""
Core machine learning algorithm implementations.

This module provides implementations of fundamental ML algorithms and their
associated loss functions, gradients, and utility functions:

Main Algorithms:
- Gradient Descent (GD) for Linear Regression
- Stochastic Gradient Descent (SGD) for Linear Regression
- Least Squares (Normal Equation)
- Ridge Regression
- Logistic Regression with Gradient Descent
- Regularized Logistic Regression

Balanced Variants:
- Balanced versions of algorithms to handle class imbalance
- Custom loss weighting for minority class

Utility Functions:
- Loss computation (MSE, Logistic Loss)
- Gradient computation
- Mini-batch iteration
- Polynomial feature expansion
- Sigmoid activation

All implementations follow a consistent interface and are designed for
binary classification tasks with labels in {-1, 1}.
"""

from typing import Generator
import numpy as np


################################################################################
# Main Algorithm Implementations
################################################################################


def mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Linear regression using Gradient Descent (GD) with Mean Squared Error loss.

    Iteratively updates weights by moving in the direction of steepest descent
    of the MSE loss function. Uses the full dataset at each iteration.

    Args:
        y: numpy array of shape=(N,). Target values.
        tx: numpy array of shape=(N, D). Feature matrix with N samples and D features.
        initial_w: numpy array of shape=(D,). Initial weights for optimization.
        max_iters: int. Maximum number of gradient descent iterations.
        gamma: float. Learning rate (step size) for gradient descent.

    Returns:
        tuple containing:
            - w (np.ndarray): Final weight vector after max_iters iterations, shape (D,).
            - loss (np.generic): Final MSE loss value (scalar).

    Note:
        This is a batch gradient descent algorithm. For large datasets,
        consider using mean_squared_error_sgd instead.
    """
    w = initial_w
    # Perform gradient descent iterations
    for _ in range(max_iters):
        gradient: np.ndarray = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    # Compute final loss
    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def mean_squared_error_gd_balanced(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Linear regression using GD with class-balanced MSE loss.

    Similar to standard GD but applies higher weight to errors from the
    minority class. Useful for imbalanced datasets where one class has
    significantly fewer samples than the other.

    Args:
        y: numpy array of shape=(N,). Target values in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        initial_w: numpy array of shape=(D,). Initial weights.
        max_iters: int. Number of iterations.
        gamma: float. Learning rate.

    Returns:
        tuple containing:
            - w (np.ndarray): Final weight vector, shape (D,).
            - loss (np.generic): Final balanced MSE loss (scalar).

    Note:
        The balancing factor is computed as ratio of negative to positive samples.
        Errors from the minority class are weighted proportionally higher.
    """
    w = initial_w
    # Perform balanced gradient descent iterations
    for _ in range(max_iters):
        gradient: np.ndarray = compute_balanced_gradient(y, tx, w)
        w = w - gamma * gradient

    # Compute final balanced loss
    loss: np.generic = mean_squared_balanced_error(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Linear regression using Stochastic Gradient Descent (SGD) with MSE loss.

    Updates weights using one random sample at a time (batch_size=1). This can
    be faster than batch GD for large datasets and may help escape local minima.

    Args:
        y: numpy array of shape=(N,). Target values.
        tx: numpy array of shape=(N, D). Feature matrix.
        initial_w: numpy array of shape=(D,). Initial weights.
        max_iters: int. Number of SGD iterations (updates).
        gamma: float. Learning rate.

    Returns:
        tuple containing:
            - w (np.ndarray): Final weight vector, shape (D,).
            - loss (np.generic): Final MSE loss computed on full dataset (scalar).

    Note:
        Each iteration uses a randomly selected single sample. The final loss
        is computed on the entire dataset for consistent comparison.
    """
    batch_size = 1
    w = initial_w
    # Perform stochastic gradient descent iterations
    for _ in range(max_iters):
        # Sample one random data point
        mini_y, mini_tx = next(batch_iter(y, tx, batch_size))
        gradient: np.ndarray = compute_gradient(mini_y, mini_tx, w)
        w = w - gamma * gradient

    # Compute final loss on full dataset
    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, np.generic]:
    """Compute optimal linear regression weights using the Normal Equation.

    Solves the closed-form solution: w = (X^T X)^(-1) X^T y
    This provides the exact optimal solution for linear regression with MSE loss.

    Args:
        y: numpy array of shape (N,). Target values.
        tx: numpy array of shape (N, D). Feature matrix.

    Returns:
        tuple containing:
            - w (np.ndarray): Optimal weight vector, shape (D,).
            - loss (np.generic): MSE loss with optimal weights (scalar).

    Note:
        This is an exact solution (no iterations needed) but requires inverting
        a D×D matrix, which can be expensive for high-dimensional data.
        The matrix X^T X must be invertible (full rank).
    """
    tx_t = tx.T
    # Solve the normal equation: (X^T X) w = X^T y
    a = tx_t @ tx
    b = tx_t @ y
    w: np.ndarray = np.linalg.solve(a, b)

    # Compute MSE loss with optimal weights
    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> tuple[np.ndarray, np.generic]:
    """Compute optimal ridge regression weights with L2 regularization.

    Solves the regularized normal equation: w = (X^T X + λ'I)^(-1) X^T y
    where λ' = 2λN and N is the number of samples.

    Ridge regression adds L2 penalty to prevent overfitting and improve
    numerical stability when X^T X is near-singular.

    Args:
        y: numpy array of shape (N,). Target values.
        tx: numpy array of shape (N, D). Feature matrix.
        lambda_: float. Regularization parameter (controls penalty strength).

    Returns:
        tuple containing:
            - w (np.ndarray): Optimal regularized weight vector, shape (D,).
            - loss (np.generic): MSE loss (without regularization term) (scalar).

    Note:
        - Larger lambda_ values produce smaller weights (more regularization)
        - The returned loss is the unregularized MSE for fair comparison
        - The regularization parameter is scaled by 2N internally
    """
    n, d = tx.shape
    # Scale regularization parameter by 2N
    lambda_prime = 2 * lambda_ * n
    tx_t = tx.T
    # Solve regularized normal equation: (X^T X + λ'I) w = X^T y
    a = tx_t @ tx + lambda_prime * np.eye(d)
    b = tx_t @ y
    w: np.ndarray = np.linalg.solve(a, b)

    # Compute unregularized MSE loss for comparison
    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Logistic regression for binary classification using gradient descent.

    Optimizes the logistic loss (negative log-likelihood) using gradient descent.
    Suitable for binary classification with labels in {-1, 1}.

    Args:
        y: numpy array of shape=(N,). Binary labels in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        initial_w: numpy array of shape=(D,). Initial weights.
        max_iters: int. Number of gradient descent iterations.
        gamma: float. Learning rate.

    Returns:
        tuple containing:
            - w (np.ndarray): Final weight vector, shape (D,).
            - loss (np.generic): Final logistic loss value (scalar).

    Note:
        The logistic loss is: L = mean(-y·(Xw) + log(1 + exp(Xw)))
        This is the negative log-likelihood for binary classification.
    """
    w = initial_w
    # Perform gradient descent iterations
    for _ in range(max_iters):
        gradient: np.ndarray = logistic_gradient(y, tx, w)
        w = w - gamma * gradient

    # Compute final logistic loss
    loss: np.generic = logistic_loss(y, tx, w)
    return w, loss


def logistic_regression_balanced(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Logistic regression with class-balanced loss for imbalanced datasets.

    Similar to standard logistic regression but applies higher weight to the
    minority class to handle class imbalance. The loss from minority class
    samples is scaled up proportionally to the class ratio.

    Args:
        y: numpy array of shape=(N,). Binary labels in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        initial_w: numpy array of shape=(D,). Initial weights.
        max_iters: int. Number of iterations.
        gamma: float. Learning rate.

    Returns:
        tuple containing:
            - w (np.ndarray): Final weight vector, shape (D,).
            - loss (np.generic): Final balanced logistic loss (scalar).

    Note:
        The balancing factor is the ratio of negative to positive samples.
        This helps the model pay more attention to the minority class.
    """
    w = initial_w
    # Perform balanced gradient descent iterations
    for _ in range(max_iters):
        gradient: np.ndarray = logistic_gradient_balanced(y, tx, w)
        w = w - gamma * gradient

    # Compute final balanced logistic loss
    loss: np.generic = logistic_loss_balanced(y, tx, w)
    return w, loss


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Regularized logistic regression with L2 penalty.

    Adds L2 regularization to logistic regression to prevent overfitting.
    The regularization term penalizes large weight values.

    Args:
        y: numpy array of shape=(N,). Binary labels in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        lambda_: float. Regularization parameter (penalty strength).
        initial_w: numpy array of shape=(D,). Initial weights.
        max_iters: int. Number of iterations.
        gamma: float. Learning rate.

    Returns:
        tuple containing:
            - w (np.ndarray): Final weight vector, shape (D,).
            - loss (np.generic): Final logistic loss without regularization term (scalar).

    Note:
        - The gradient includes regularization: ∇L + 2λw
        - Larger lambda_ values produce smaller weights (more regularization)
        - The returned loss excludes the regularization term for fair comparison
    """
    w = initial_w
    # Perform regularized gradient descent
    for _ in range(max_iters):
        # Add L2 regularization term to gradient
        gradient: np.ndarray = logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient

    # Compute unregularized logistic loss
    loss: np.generic = logistic_loss(y, tx, w)
    return w, loss


################################################################################
# Loss Functions
################################################################################


def mean_squared_error(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.generic:
    """Calculate the Mean Squared Error (MSE) loss.

    Computes: MSE = (1/2N) Σ(y_i - x_i^T w)^2

    Args:
        y: numpy array of shape=(N,). Target values.
        tx: numpy array of shape=(N, D). Feature matrix.
        w: numpy array of shape=(D,). Weight vector.

    Returns:
        Scalar MSE loss value.

    Note:
        The factor of 1/2 simplifies the gradient computation.
    """
    n: int = y.shape[0]
    # Compute prediction errors
    error: np.ndarray = y - tx @ w
    # Compute MSE: (1/2N) ||error||^2
    loss: np.generic = 1 / (2 * n) * error.T @ error  # type: ignore
    return loss


def mean_squared_balanced_error(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.generic:
    """Calculate class-balanced Mean Squared Error loss.

    Similar to standard MSE but applies higher weight to errors from the
    minority class. The correction factor is the ratio of class frequencies.

    Args:
        y: numpy array of shape=(N,). Target values in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        w: numpy array of shape=(D,). Weight vector.

    Returns:
        Scalar balanced MSE loss value.

    Note:
        Errors from the minority class are weighted by the class ratio,
        giving them more importance in the overall loss.
    """
    n: int = y.shape[0]
    # Compute class-weighted errors
    error: np.ndarray = compute_balanced_errors(y, tx, w)
    # Compute balanced MSE
    loss: np.generic = 1 / (2 * n) * error.T @ error  # type: ignore
    return loss


################################################################################
# Gradient Functions
################################################################################


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Mean Squared Error loss function.

    The gradient is: ∇L = -(1/N) X^T (y - Xw)

    Args:
        y: numpy array of shape=(N,). Target values.
        tx: numpy array of shape=(N, D). Feature matrix.
        w: numpy array of shape=(D,). Current weight vector.

    Returns:
        Gradient vector of shape (D,), same shape as w.

    Note:
        The negative sign indicates the direction of steepest descent.
    """
    n = y.shape[0]
    # Compute prediction errors
    errors: np.ndarray = y - tx @ w
    # Compute gradient: -(1/N) X^T error
    gradient: np.ndarray = -1 / n * tx.T @ errors
    return gradient


def compute_balanced_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute the gradient of the class-balanced MSE loss function.

    Similar to standard gradient but uses balanced errors that weight
    the minority class more heavily.

    Args:
        y: numpy array of shape=(N,). Target values in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        w: numpy array of shape=(D,). Current weight vector.

    Returns:
        Balanced gradient vector of shape (D,), same shape as w.

    Note:
        The balancing helps gradient descent pay more attention to
        correctly classifying the minority class.
    """
    n = y.shape[0]
    # Compute class-weighted errors
    errors: np.ndarray = compute_balanced_errors(y, tx, w)
    # Compute balanced gradient
    gradient: np.ndarray = -1 / n * tx.T @ errors
    return gradient


def compute_balanced_errors(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Calculate weighted prediction errors to handle class imbalance.

    Computes standard errors (y - Xw) but scales them based on class frequency.
    Errors from the minority class are weighted more heavily to ensure they
    contribute proportionally to the loss despite having fewer samples.

    Args:
        y: numpy array of shape=(N,). Target values in {-1, 1}.
        tx: numpy array of shape=(N, D). Feature matrix.
        w: numpy array of shape=(D,). Weight vector.

    Returns:
        Weighted error vector of shape (N,).

    Note:
        The correction factor is the ratio of negative to positive samples.
        - If negatives are majority (factor > 1): positive errors are scaled up
        - If positives are majority (factor < 1): negative errors are scaled up
    """
    # Compute prediction errors
    error = y - (tx @ w)
    # Calculate class imbalance ratio
    correction_factor = negative_positive_ratio(y)
    
    # Apply higher weight to minority class errors
    if correction_factor > 1:
        # More negatives than positives: weight positive errors more
        error[error >= 0] *= correction_factor
    else:
        # More positives than negatives: weight negative errors more
        error[error <= 0] *= correction_factor
    return error


################################################################################
# Utility Functions
################################################################################

def batch_iter(y: np.ndarray, tx: np.ndarray, batch_size: int, num_batches: int = 1, shuffle: bool = True) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add a random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]

def sigmoid(t: np.ndarray | np.generic) -> np.ndarray | np.generic:
    """Apply the sigmoid function on t.

    Args:
        t: A scalar or numpy array of any size.

    Returns:
        The sigmoid of t.
    """
    return 1 / (1 + np.exp(-t))

def logistic_loss(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.generic:
    """Compute the cost by negative log likelihood.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        The negative log likelihood.
    """
    t = tx @ w
    loss: np.generic = np.mean(-y * t + np.log(1 + np.exp(t)))
    return loss

def logistic_loss_balanced(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.generic:
    """Compute the cost by negative log likelihood with class balancing.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        The negative log likelihood.
    """
    t = tx @ w
    correction_factor = negative_positive_ratio(y)
    loss_vector = -y * t + np.log(1 + np.exp(t))
    
    if correction_factor > 1:
        loss_vector[y == 1] *= correction_factor
    else:
        loss_vector[y == -1] *= correction_factor
        
    loss: np.generic = np.mean(loss_vector)
    return loss


def logistic_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """compute the gradient of loss.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, ).

    Returns:
        a vector of shape (D )
    """
    n = y.shape[0]
    s: np.ndarray = sigmoid(tx@w)
    gradient: np.ndarray = 1 / n * tx.T @ (s - y)
    return gradient

def logistic_gradient_balanced(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """compute the gradient of loss accounting for data imbalance.

    Args:
        y:  shape=(N, )
        tx: shape=(N, D)
        w:  shape=(D, ).

    Returns:
        a vector of shape (D )
    """
    n = y.shape[0]
    correction_factor = negative_positive_ratio(y)
    s: np.ndarray = sigmoid(tx@w)
    neg_indices = np.where(y == -1)[0]
    pos_indices = np.where(y == 1)[0]
    if correction_factor > 1:
        s[pos_indices] *= correction_factor
    else:
        s[neg_indices] *= correction_factor
    gradient: np.ndarray = 1 / n * tx.T @ (s - y)
    return gradient

def build_poly(x: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial basis functions for input data x, for j=1 up to j=degree.

    Args:
        x: numpy array of shape=(num_samples, num_features)
        degree: integer
    Returns:
        numpy array of shape=(num_samples, num_features*degree)
    """
    num_samples, num_features = x.shape
    one_dim: np.ndarray = x.reshape(-1)  # Flatten to 1D array (rows stacked horitzontally)
    poly: np.ndarray = np.vander(one_dim, degree+1, increasing=True)[:, 1:]  # Create Vandermonde matrix and remove the first column (degree 0)
    # We have a matrix where each row is a feature expansion of a single feature.
    # We reshape it to have all features for a single sample in a row.
    poly: np.ndarray = poly.reshape(num_samples, num_features * degree)
    return poly

def negative_positive_ratio(y: np.ndarray) -> float:
    """Compute the ratio of negative to positive samples in y.

    Args:
        y: numpy array of shape=(N, )
    Returns:
        ratio: float
    """
    # count labels in y (assumes labels are -1 and 1)
    neg_count = np.count_nonzero(y == -1)
    pos_count = np.count_nonzero(y == 1)

    # to avoid division by zero
    if neg_count == 0:
        return 1.0

    correction_factor = neg_count / pos_count
    return correction_factor