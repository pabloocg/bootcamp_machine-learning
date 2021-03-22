import numpy as np


def predict_(x, theta):
    """
    Computes the prediction vector y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * n.
    theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if len(x) < 1 or x.shape[1] + 1 != theta.shape[0]:
        return None
    xi = np.zeros((x.shape[0], x.shape[1] + 1))
    xi[:, 0], xi[:, 1:] = 1, x
    return np.sum(xi * theta, axis=1)
