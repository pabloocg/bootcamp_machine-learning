import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from polynomial_model import add_polynomial_features
from mylinearregression import MyLinearRegression as MyLR

x = np.arange(0, 11).reshape(-1, 1)
y = np.array([[ 0.99270298],
[ 1.39270298],
[ 3.88237651],
[ 4.37726357],
[ 4.63389049],
[ 7.79814439],
[ 6.41717461],
[ 8.63429886],
[ 8.19939795],
[10.37567392],
[10.68238222]])

x_ = add_polynomial_features(x, 5)
my_lr = MyLR(np.ones(6).reshape(-1,1), alpha=2.5e-10, n_cycle=1000000)
my_lr.fit_(x_, y)
continuous_x = np.arange(0, 10.01, 0.01).reshape(-1,1)
x_ = add_polynomial_features(continuous_x, 5)
y_hat = my_lr.predict_(x_)
plt.scatter(x, y)
plt.plot(continuous_x, y_hat, color='orange')
plt.show()