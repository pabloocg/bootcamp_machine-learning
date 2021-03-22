from mylinearregression import MyLinearRegression as MyLR
import numpy as np
import matplotlib.pyplot as plt


def plot_(lm, x, y):
    Y_model1 = lm.predict_(x)
    X_model = np.sum(x, axis=1).reshape(-1, 1)

    plt.plot(X_model, Y_model1, '--r', color='g')
    plt.scatter(X_model, y, color='b', s=40, label='Strue(pills)')
    plt.scatter(X_model, Y_model1, color='r', s=20, label='Spredict(pills)')
    plt.xlabel('Quantity of blue pill (in micrograms)')
    plt.ylabel('Space driving score')
    plt.grid(True)
    plt.legend(loc='upper left')

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



X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [34., 55., 89., 144.]])
X = minmax(X)
Y = np.array([[23.], [48.], [218.]])
Y = minmax(Y)
mylr = MyLR(np.array([[1.], [1.], [1.], [1.], [1.]]))

plot_(mylr, X, Y)
mylr.fit_(X, Y)
plot_(mylr, X, Y)
plt.show()
#plot_cost_(mylr)
#mylr.fit_(X, Y)
print(mylr.thetas)
#plot_cost_(mylr)
#plot_(mylr, X, Y)
