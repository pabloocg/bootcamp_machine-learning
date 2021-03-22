import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from my_linear_regression import MyLinearRegression as MyLR
import matplotlib.pyplot as plt



def plot_(lm, x, y):
    Y_model1 = lm.predict_(x)

    plt.plot(x, Y_model1, '--r', color='g')
    plt.scatter(x, y, color='b', s=30, label='Strue(pills)')
    plt.scatter(x, Y_model1, color='g', s=20, label='Spredict(pills)')
    plt.xlabel('Quantity of blue pill (in micrograms)')
    plt.ylabel('Space driving score')
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.show()

def plot_cost_(lm):
    plt.scatter(lm.thetas[0],lm.thetas[1], color='g', s=20)
    plt.ylabel('Cost')
    plt.grid(True)
    plt.show()


data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)

linear_model1 = MyLR(np.array([[89.0], [-8]]))
#plot_(linear_model1, Xpill, Yscore)
#plot_cost_(linear_model1)
linear_model1.fit_(Xpill, Yscore)
print(linear_model1.thetas)
#plot_cost_(linear_model1)
plot_(linear_model1, Xpill, Yscore)
