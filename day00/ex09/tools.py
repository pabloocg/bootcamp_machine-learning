import numpy as np


def add_intercept(x):
    """Adds a column of 1's to the non-empty numpy.ndarray x. Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1. Returns:
    X as a numpy.ndarray, a vector of dimension m * 2.
    None if x is not a numpy.ndarray.
    None if x is a empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or len(x) < 1:
        return None
    if x.shape == (x.shape[0],):
        x = x[:, np.newaxis]
    return np.insert(x, 0, values=1, axis=1)


def main():
    x = np.arange(1,6)
    ret = add_intercept(x)
    print(ret)
    y = np.arange(1,7).reshape((3,2))
    ret = add_intercept(y)
    print(ret)


if __name__ == "__main__":
    main()
