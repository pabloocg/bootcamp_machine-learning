import numpy as np


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
    return (np.sum(xi.T * (np.sum(xi * theta, axis=1) - y), axis=1)) / x.shape[0]


x = np.array([
[ -6, -7, -9],
[ 13, -2, 14],
[ -7, 14, -1],
[-8,-4, 6],
[-5,-9, 6],
[ 1, -5, 11],
[9,-11, 8]])

y = np.array([2, 14, -13, 5, 12, 4, -19])

theta1 = np.array([0, 0, 0, 0])

print(gradient(x, y, theta1))