import numpy as np
from tools import add_intercept


def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray. Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    if len(x) < 1 or theta.shape != (2,):
        return None
    return np.sum(add_intercept(x) * theta, axis=1)


def main():
    x = np.arange(1,6)
    theta1 = np.array([5, 0])
    ret = predict_(x, theta1)
    print(ret)
    theta2 = np.array([0, 1])
    ret = predict_(x, theta2)
    print(ret)
    theta3 = np.array([5, 3])
    ret = predict_(x, theta3)
    print(ret)
    theta4 = np.array([-3, 1])
    ret = predict_(x, theta4)
    print(ret)
    theta5 = np.array([1])
    ret = predict_(x, theta5)
    print(ret)
    theta6 = np.array([5, 1])
    ret = predict_("", theta6)
    print(ret)


if __name__ == "__main__":
    main()
