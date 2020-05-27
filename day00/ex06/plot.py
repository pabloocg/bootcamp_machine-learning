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
    for x_, y_ in zip(x, y):
        plt.scatter(x_, y_, color='b', s=30)
    predX = predict_(x, theta)
    plt.plot(x, predX, color='r')
    plt.show()


def main():
    x = np.arange(1,6)
    y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
    #theta1 = np.array([4.5, -0.2])
    #theta1 = np.array([-1.5, 2])
    theta1 = np.array([3, 0.3])
    plot(x, y, theta1)


if __name__ == "__main__":
    main()
