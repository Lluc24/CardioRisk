import numpy as np
from costs import compute_loss  # list only what you use

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    N, _ = tx.shape
    err = y - tx @ w
    gradient = -1 / N * (tx.T @ err)
    return gradient


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)        
        w = w - gamma*gradient

        # store w and loss
        ws.append(w)
        losses.append(loss)
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws


def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    M = tx.T @ tx # shape (D,D)
    b = tx.T @ y # shape (D,)
    w = np.linalg.solve(M, b) #Should be shape (D,)
    loss = compute_loss(y, tx, w)
    return (w, loss)

def ridge_regression(y, tx, lambda_):
    """implement ridge regression.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 0)
    array([ 0.21212121, -0.12121212])
    >>> ridge_regression(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]), 1)
    array([0.0397092, 0.00319628])
    """
    # Calculating constants and parameters
    N, D = tx.shape

    # Building the system
    M = 1/N * tx.T @ tx + (2 * lambda_ * np.eye(D))
    b = 1/N * tx.T @ y # same as in original least squares

    # Solving the system
    w = np.linalg.solve(M, b)
    return w

def build_model(weights: np.ndarray):
    """Builds the function implementing the prediction using a linear regression with weights weights
    
    Args:
        weights: numpy array of shape (D,), D is the number of features
    
    Returns:
        f: a fuctcion implementing the linear regression predicting function parametrized by weights. Args: datapint np array (N, D). Returns: np array (N,)
    """

    def f(tx: np.ndarray):
        return tx @ weights
    return f