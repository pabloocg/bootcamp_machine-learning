from prediction import predict_
import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.ndarray. Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    if x.shape != (x.shape[0],):
        x_hat = np.sum(x, axis=1)
    else:
        x_hat = x
    for x_, y_ in zip(x_hat, y):
        plt.scatter(x_, y_, color='b', s=30)
    predX = predict_(x, theta)
    plt.plot(x, predX, color='r')
    plt.show()
