import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#use sklearn to do linear regression
from sklearn.linear_model import LinearRegression

#data with two colums: Performance and Price
data_file = 'data.csv'
data = pd.read_csv(data_file)

#panda describe tool for statistical summary
data.describe()

y = data['Performance']
x = data['Price']

#x.shape is (500,) which is a vector, and we need to change its shape to a matrix (500,1). -1 means remaining unchanged
x_matrix = x.values.reshape(-1,1)

#create linear regression blackbox
reg = LinearRegression()

#apply data to the blackbox
reg.fit(x_matrix,y)

#R-score, coefficients, y-intercept
reg.score(x_matrix,y)
reg.coef_
reg.intercept_

#predict Performance values with two prices 1000 and 700
reg.predict([[1000],[700]])

#another way to feed price data used for prediction
newdata = pd.DataFrame(data=[700,800,900,1000,1100], columns=['Price'])
reg.predict(newdata)

#in pandas dataframe we can add a new column by indicating new index
newdata['Predicted_Price'] = reg.predict(newdata)