from typing import Generator
import numpy as np

from dataset import Dataset


def mean_squared_error_gd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """The Gradient Descent (GD) algorithm.

        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N, D)
            initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
            max_iters: a scalar denoting the total number of iterations of GD
            gamma: a scalar denoting the stepsize

        Returns:
            w: The last weight vector of the method
            loss: the value of the loss (MSE) using the last weight vector
        """
    w = initial_w
    for _ in range(max_iters):
        gradient: np.ndarray = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """The Stochastic Gradient Descent algorithm (SGD).

        Args:
            y: numpy array of shape=(N, )
            tx: numpy array of shape=(N, D)
            initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
            max_iters: a scalar denoting the total number of iterations of SGD
            gamma: a scalar denoting the stepsize

        Returns:
            w: The last weight vector of the method
            loss: the value of the loss (MSE) using the last weight vector
    """
    batch_size = 1
    w = initial_w
    for _ in range(max_iters):
        mini_y, mini_tx = next(batch_iter(y, tx, batch_size))
        gradient: np.ndarray = compute_gradient(mini_y, mini_tx, w)
        w = w - gamma * gradient

    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def least_squares(y: np.ndarray, tx: np.ndarray) -> tuple[np.ndarray, np.generic]:
    """Calculate the least squares solution.
           returns mse, and optimal weights.

        Args:
            y: numpy array of shape (N,), N is the number of samples.
            tx: numpy array of shape (N, D), D is the number of features.

        Returns:
            w: optimal weights, numpy array of shape(D,), D is the number of features.
            mse: scalar.
    """
    tx_t = tx.T
    a = tx_t @ tx
    b = tx_t @ y
    w: np.ndarray = np.linalg.solve(a, b)  # Solve the equation Aw = b
    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def ridge_regression(y: np.ndarray, tx: np.ndarray, lambda_: float) -> tuple[np.ndarray, np.generic]:
    """implement ridge regression.

        Args:
            y: numpy array of shape (N, ), N is the number of samples.
            tx: numpy array of shape (N, D), D is the number of features.
            lambda_: scalar.

        Returns:
            w: optimal weights, numpy array of shape(D,), D is the number of features.
            loss: scalar MSE
        """
    n, d = tx.shape
    lambda_prime = 2 * lambda_ * n
    tx_t = tx.T
    a = tx_t @ tx + lambda_prime * np.eye(d)
    b = tx_t @ y
    w: np.ndarray = np.linalg.solve(a, b)  # Solve the equation Aw = b
    loss: np.generic = mean_squared_error(y, tx, w)
    return w, loss


def logistic_regression(y: np.ndarray, tx: np.ndarray, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of the method
        gamma: a scalar denoting the stepsize
    Returns:
        w: The last weight vector of the method
        loss: the value of the loss (logistic loss) using the last weight vector
    """
    w = initial_w
    for _ in range(max_iters):
        gradient: np.ndarray = logistic_gradient(y, tx, w)
        w = w - gamma * gradient

    loss: np.generic = logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y: np.ndarray, tx: np.ndarray, lambda_: float, initial_w: np.ndarray, max_iters: int, gamma: float) -> tuple[np.ndarray, np.generic]:
    """Regularized logistic regression using gradient descent.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        lambda_: scalar, regularization parameter
        initial_w: numpy array of shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of the method
        gamma: a scalar denoting the stepsize

    Returns:
        w: The last weight vector of the method
        loss: the value of the loss (regularized logistic loss) using the last weight vector
    """
    w = initial_w
    for _ in range(max_iters):
        gradient: np.ndarray = logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    loss: np.generic = logistic_loss(y, tx, w)
    return w, loss


########################################################################################3
# Auxiliar Functions
########################################################################################3

def mean_squared_error(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.generic:
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    n: int  = y.shape[0]
    error: np.ndarray = y - tx @ w
    loss: np.generic = 1 / (2 * n) * error.T @ error # type: ignore
    return loss


def compute_gradient(y: np.ndarray, tx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Computes the gradient at w of the Mean Squared Error loss function.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    n = y.shape[0]
    errors: np.ndarray = y - tx @ w
    gradient: np.ndarray = -1 / n * tx.T @ errors
    return gradient


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
