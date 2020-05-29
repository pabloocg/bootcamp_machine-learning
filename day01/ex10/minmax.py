import numpy as np
from math import sqrt


def minmax(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray using the min-max
    standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    return (x - np.min(x)) / (np.max(x) - np.min(x))


X = np.array([0, 15, -9, 7, 12, 3, -21])
print(minmax(X))
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(minmax(Y))