import numpy as np


def h(xi, theha):
    return theha[0] + (theha[1] * xi)


def simple_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop.
    The three arrays must have compatible dimensions.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    y: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a 2 * 1 vector.
    Returns:
    The gradient as a numpy.ndarray, a vector of dimension 2 * 1.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    sumt1 = sumt0 = 0.0
    for x_, y_ in zip(x, y):
        sumt0 += h(x_, theta) - y_
        sumt1 += (h(x_, theta) - y_) * x_
    j0, j1 = sumt0 / x.shape[0], sumt1 / x.shape[0]
    return np.array([j0, j1])


x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733])
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554])
#theta1 = np.array([2, 0.7])
theta2 = np.array([1, -0.4])
print(simple_gradient(x, y, theta2))
