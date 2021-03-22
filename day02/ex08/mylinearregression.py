import numpy as np
from math import sqrt


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """
    def __init__(self, thetas: np.array, alpha=0.001, n_cycle=150000):
        self.alpha = alpha
        self.max_iter = n_cycle
        self.thetas = np.array(thetas)

    def gradient(self, x, y):
        """
        Computes a gradient vector from three non-empty numpy.ndarray,
        without any for-loop. The three arrays must have the compatible dimensions.
        Args:
        x: has to be an numpy.ndarray, a matrix of dimension m * n.
        y: has to be an numpy.ndarray, a vector of dimension m * 1.
        Returns:
        The gradient as a numpy.ndarray, a vector of dimensions n * 1,
        containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
        Raises:
        This function should not raise any Exception.
        """
        if x.shape[0] != y.shape[0] or x.shape[1] + 1 != self.thetas.shape[0]:
            return None
        m = x.shape[0]
        xi = self.add_intercept(x)
        predictions = np.dot(xi, self.thetas)
        return np.dot(xi.T, (predictions - y)) / m

    def add_intercept(self, x):
        """
        Adds a column of 1's to the non-empty numpy.ndarray x. Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1. Returns:
        X as a numpy.ndarray, a vector of dimension m * 2.
        None if x is not a numpy.ndarray.
        None if x is a empty numpy.ndarray.
        Raises:
        This function should not raise any Exception.
        """
        if not isinstance(x, np.ndarray) or len(x) < 1:
            return None
        xi = np.zeros((x.shape[0], x.shape[1] + 1))
        xi[:, 0], xi[:, 1:] = 1, x
        return xi

    def fit_(self, x, y):
        """
        Description:
        Fits the model to the training dataset contained in x and y.
        Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        Returns:
        None
        Raises:
        This function should not raise any Exception.
        """
        if x.shape == (x.shape[0],):
            x = x[:, np.newaxis]
        if y.shape == (y.shape[0],):
            y = y[:, np.newaxis]
        cost_history = [0] * self.max_iter
        for i in range(self.max_iter):
            nabJ = self.gradient(x, y)
            self.thetas = self.thetas - self.alpha * nabJ
            cost_history[i] = self.cost_(self.predict_(x), y)
        return cost_history
    
    def predict_(self, x):
        """
        Computes the prediction vector y_hat from two non-empty numpy.ndarray.
        Args:
        x: has to be an numpy.ndarray, a vector of dimension m * n.
        theta: has to be an numpy.ndarray, a vector of dimension (n + 1) * 1.
        Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
        Raises:
        This function should not raise any Exception.
        """
        if len(x) < 1 or x.shape[1] + 1 != self.thetas.shape[0]:
            return None
        return np.dot(self.add_intercept(x), self.thetas)

    def cost_elem_(self, x, y):
        """
        Description:
        Calculates all the elements (1/2*M)*(y_pred - y)^2 of the cost function.
        Args:
        x: has to be an numpy.ndarray, a vector.
        y: has to be an numpy.ndarray, a vector.
        Returns:
        J_elem: numpy.ndarray, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        Raises:
        This function should not raise any Exception.
        """
        if len(x) < 1 or len(y) < 1 or y.shape[0] != x.shape[0]:
            return None
        return ((x - y)**2) / (2 * y.shape[0])

    def cost_(self, x, y):
        """
        Computes the mean squared error of two non-empty numpy.ndarray, without any for loop.
        The two arrays must have the same dimensions.
        Args:
        x: has to be an numpy.ndarray, a vector.
        y: has to be an numpy.ndarray, a vector.
        Returns:
        The mean squared error of the two vectors as a float.
        None if y or x are empty numpy.ndarray.
        None if y and x does not share the same dimensions.
        Raises:
        This function should not raise any Exceptions.
        """
        if len(x) < 1 or len(y) < 1 or y.shape[0] != x.shape[0]:
            return None
        return np.sum(self.cost_elem_(x, y))

    def mse_(self, x, y):
        """
        Description:
        Calculate the MSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a vector of dimension m * 1.
        Returns:
        mse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != x.shape:
            return None
        return np.sum(((x - y)**2) / float(y.shape[0]))

    def rmse_(self, x, y):
        """
        Description:
        Calculate the RMSE between the predicted output and the real output.
        Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a vector of dimension m * 1.
        Returns:
        rmse: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != x.shape:
            return None
        return sqrt(np.sum(((x - y)**2) / float(y.shape[0])))

    def mae_(self, x, y):
        """
        Description:
        Calculate the MAE between the predicted output and the real output.
        Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a vector of dimension m * 1.
        Returns:
        mae: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != x.shape:
            return None
        return np.sum(np.absolute(x - y) / float(y.shape[0]))

    def r2score_(self, x, y):
        """
        Description:
        Calculate the R2score between the predicted output and the output.
        Args:
        y: has to be a numpy.ndarray, a vector of dimension m * 1.
        x: has to be a numpy.ndarray, a vector of dimension m * 1.
        Returns:
        r2score: has to be a float.
        None if there is a matching dimension problem.
        Raises:
        This function should not raise any Exceptions.
        """
        if y.shape != x.shape:
            return None
        return 1 - (np.sum((x - y)**2) / np.sum((x - np.median(y))**2))

    def minmax(self, x):
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
