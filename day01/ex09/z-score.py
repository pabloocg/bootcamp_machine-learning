import numpy as np
from math import sqrt


def zscore(x):
    """
    Computes the normalized version of a non-empty numpy.ndarray using
    the z-score standardization.
    Args:
    x: has to be an numpy.ndarray, a vector.
    Returns:
    x' as a numpy.ndarray.
    None if x is a non-empty numpy.ndarray.
    Raises:
    This function shouldn't raise any Exception.
    """
    return (x - np.mean(x)) / np.std(x)


X = np.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(X))
Y = np.array([2, 14, -13, 5, 12, 4, -19])
print(zscore(Y))