import numpy as np
from tqdm import tqdm

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
    for n_iter in tqdm(range(max_iters)):
        gradient = compute_gradient(y, tx, w)
        w = w - gamma * gradient

    loss = mean_squared_error(y, tx, w)
    return w, loss