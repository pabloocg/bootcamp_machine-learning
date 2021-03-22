import numpy as np
from prediction import predict_


def add_ones(x):
    xi = np.zeros((x.shape[0], x.shape[1] + 1))
    xi[:, 0], xi[:, 1:] = 1, x
    return xi


def gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have the compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a matrix of dimension m * n.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimensions n * 1,
    containg the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    if x.shape[0] != y.shape[0] or x.shape[1] + 1 != theta.shape[0]:
        return None
    xi = add_ones(x)
    return (1 / y.shape[0]) * np.sum(xi.T * (predict_(x, theta) - y), axis=1)
