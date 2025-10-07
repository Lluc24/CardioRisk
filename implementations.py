import numpy as np

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
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
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = mean_squared_error(y, tx, w)
    return w, loss

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
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
        gradient = compute_gradient(mini_y, mini_tx, w)
        w = w - gamma * gradient

    loss = mean_squared_error(y, tx, w)
    return w, loss


def least_squares(y, tx):
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
    A = tx_t @ tx
    b = tx_t @ y
    w = np.linalg.solve(A, b)
    loss = mean_squared_error(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

        Args:
            y: numpy array of shape (N, ), N is the number of samples.
            tx: numpy array of shape (N, D), D is the number of features.
            lambda_: scalar.

        Returns:
            w: optimal weights, numpy array of shape(D,), D is the number of features.
            loss: scalar MSE
        """
    N, D = tx.shape
    lambda_prime = 2 * lambda_ * N
    tx_t = tx.T
    A = tx_t @ tx + lambda_prime * np.eye(D)
    b = tx_t @ y
    w = np.linalg.solve(A, b)
    loss = mean_squared_error(y, tx, w)
    return w, loss


########################################################################################3
# Auxiliar Functions
########################################################################################3

def mean_squared_error(y, tx, w):
    """Calculate the loss using MSE

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    (N, )  = y.shape
    error = y - tx @ w
    loss = 1 / (2 * N) * error.T @ error
    return loss


def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N, D)
        w: numpy array of shape=(D, ). The vector of model parameters.

    Returns:
        An numpy array of shape (D, ) (same shape as w), containing the gradient of the loss at w.
    """
    (N, ) = y.shape
    errors = y - tx @ w
    gradient = -1/N * tx.T @ errors
    return gradient


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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
            # Add an random offset to the start of each batch to eventually consider the remainder points
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

def predict_labels(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Generates class predictions given weights, and a test data matrix
    Args:
        x: numpy array of shape=(num_samples, num_features)
        w: numpy array of shape=(num_features, )
    Returns:
        numpy array of shape=(num_samples, ) containing the predicted class labels
    """
    y_pred = x @ w
    y_pred_labels = np.where(y_pred <= 0, -1, 1)
    return y_pred_labels
