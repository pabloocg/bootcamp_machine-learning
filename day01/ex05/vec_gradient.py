import numpy as np


def add_ones(x):
    if not isinstance(x, np.ndarray) or len(x) < 1:
        return None
    if x.shape == (x.shape[0],):
        x = x[:, np.newaxis]
    return np.insert(x, 0, values=1, axis=1)


def gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for loop.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be a numpy.ndarray, a matrix of dimension m * 1.
    y: has to be a numpy.ndarray, a vector of dimension m * 1.
    theta: has to be a numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta is an empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if x.shape != y.shape or theta.shape != (theta.shape[0],):
        return None
    xo = add_ones(x)
    return np.sum(xo.T * (np.sum(xo * theta, axis=1) - y), axis=1) / x.shape[0]
