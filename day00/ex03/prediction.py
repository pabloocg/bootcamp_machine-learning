import numpy as np


def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray. Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if len(x) < 1 or theta.shape != (2,):
        return None
    return theta[0] + (theta[1] * x)


def main():
    x = np.arange(1,6)
    theta1 = np.array([5, 0])
    ret = simple_predict(x, theta1)
    print("Result:", ret)
    theta2 = np.array([0, 1])
    ret = simple_predict(x, theta2)
    print("Result:", ret)
    theta3 = np.array([5, 3])
    ret = simple_predict(x, theta3)
    print("Result:", ret)
    theta4 = np.array([-3, 1])
    ret = simple_predict(x, theta4)
    print("Result:", ret)
    theta5 = np.array([1])
    ret = simple_predict(x, theta5)
    print("Result:", ret)
    theta6 = np.array([5, 1])
    ret = simple_predict("", theta6)
    print("Result:", ret)


if __name__ == "__main__":
    main()
