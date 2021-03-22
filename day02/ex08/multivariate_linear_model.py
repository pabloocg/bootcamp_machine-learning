import numpy as np
import pandas as pd
from mylinearregression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt


def plot_(ml, x, y, x_banner, leg_loc):
    if x.shape == (x.shape[0],):
        x = x[:, np.newaxis]
    y_hat = ml.predict_(x)
    X_model = np.sum(x, axis=1).reshape(-1, 1)
    plt.grid(True)
    plt.scatter(X_model, y, color='b', s=40, label='Sell price')
    plt.scatter(X_model, y_hat, color='c', s=20, label='Predicted sell price')
    plt.xlabel(x_banner)
    plt.ylabel('y: sell price (in keuros)')
    plt.legend(loc=leg_loc)
    plt.show()


def prepare(data, column_data, x_banner, leg_loc):
    dataX = data[column_data][1:]
    dataY = data['Sell_price'][1:]
    myLR_age = MyLR(np.array([[1.], [1.]]), alpha=2.5e-5)
    myLR_age.fit_(dataX, dataY)
    plot_(myLR_age, dataX, dataY, x_banner, leg_loc)


data = pd.read_csv("../resources/spacecraft_data.csv")
#prepare(data, 'Age', 'x1: age(in years)', 'lower left')
#prepare(data, 'Thrust_power', 'x2: thrust power(in 10Km/s)', 'upper left')
#prepare(data, 'Terameters', 'x3: distance totalizer value of spacecraft (in Tmeters)', 'upper right')
X = np.array(data[['Age','Thrust_power','Terameters']])
Y = np.array(data[['Sell_price']]).reshape((-1, 1))
my_lreg = MyLR(np.array([[300.0], [-10.0], [3.0], [-2.0]]), alpha=2.5e-5, n_cycle=600000)
my_lreg.fit_(X, Y)
prediction = my_lreg.predict_(X)
#real_age = X[:, 0]
real_dist = X[:, 2]
plt.grid(True)
plt.scatter(real_dist, Y, color='b', s=40, label='Sell price')
plt.scatter(real_dist, prediction, color='c', s=10, label='Predicted sell price')
#plt.xlabel('x1: age(in years)')
plt.xlabel('x2: thrust power(in 10Km/s)')
plt.ylabel('y: sell price (in keuros)')
plt.legend(loc='lower left')
plt.show()