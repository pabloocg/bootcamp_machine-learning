from prediction import predict_
import numpy as np


def cost_(y, y_hat):
    """
    Computes the mean squared error of two non-empty numpy.ndarray, without any for loop.
    The two arrays must have the same dimensions.
    Args:
    y: has to be an numpy.ndarray, a vector. y_hat: has to be an numpy.ndarray, a vector. Returns:
    The mean squared error of the two vectors as a float.
    None if y or y_hat are empty numpy.ndarray.
    None if y and y_hat does not share the same dimensions.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(y) < 1 or len(y_hat) < 1 or y.shape != y_hat.shape:
        return None
    return np.sum(((y_hat - y) **2) / float(y.shape[0]))
