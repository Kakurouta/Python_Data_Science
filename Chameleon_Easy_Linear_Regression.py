import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set()

def magic_trace():
    random.seed(69)
    x = [random.randint(0,100) for i in range(100)]
    y = [ 30 + x[j] + random.randint(0,40) for j in range(100)]
    random.seed(None)
    move = random.randint(0, 7)
    random.seed(69)
    x2 = [xn for xm, xn in enumerate(x) if (xm + move)%8 == 0]
    y2 = [yn for ym, yn in enumerate(y) if (ym + move)%8 == 0]
    plt.xlabel('X',fontsize=20)
    plt.ylabel('Y',fontsize=20)
    plt.xlim(0, 100)
    plt.ylim(0, 200)
    plt.scatter(x2, y2)
    plt.show()
    return x, y

def chameleon(x,y):
    x = pd.DataFrame(data=x)
    y = pd.DataFrame(data=y)
    x_matrix = x.values.reshape(-1,1)
    y_matrix = y.values.reshape(-1,1)
    reg = LinearRegression()
    reg.fit(x_matrix,y_matrix)
    yhat = reg.coef_*x + reg.intercept_
    plt.xlim(0, 100)
    plt.ylim(0, 200)
    plt.scatter(x, y, c=(x*y), cmap='cool')
    plt.plot(x, yhat, lw=4, c='b')
    plt.show()