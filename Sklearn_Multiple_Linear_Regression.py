import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#use sklearn to do linear regression
from sklearn.linear_model import LinearRegression

#finding p-values with scikit-learn
from sklearn.feature_extraction import f_regression

#feature scaling to normal distribution(central limit theorem) average=0 and variation=1
from sklearn.preprocessing import StandardScaler

#data with two colums: Performance and Price
data_file = 'data.csv'
data = pd.read_csv(data_file)

#panda describe tool for statistical summary
data.describe()

y = data['Performance']
x = data[['Price', 'Random Category']]

#x.shape is (500, 3) is matrix now and no need to reshape

#create feature scaling blackbox
scaler = StandardScaler()
#these two lines can also be combined as fit_transform()
scaler.fit(x)
x_scaled = scaler.transform(x)

#create linear regression blackbox
reg = LinearRegression()
#apply data to the blackbox
reg.fit(x_scaled, y)

#R-score, coefficients, y-intercept
reg.score(x_scaled, y)
reg.coef_
reg.intercept_

#more data columns always results in better R^2, but not necessarily better prediction
#in multi-column scenario, we use adjusted R^2. adjusted R^2 = 1 - (1-R^2)*(n-1)/(n-p-1). (n,p) = x.shape
r = reg.score(x_scaled,y)
r_adj = 1-(1-r)*(n-1)/(n-p-1)

#finding p-values (f_regression returns [f-values, p-values])
p_values = f_regression[1]

#make summary table. x.columns.values returns array ['Price', 'Random Category']
reg_summary = pd.DataFrame(data=x.columns.values, columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary['Bias'] = reg.intercept_
reg_summary['p-values'] = p_values.round(3)
#note the feature, scaling, weight, bias terms

#predict Performance values with ['Price', 'Random Category'] values
#remember to scale the input data too
newdata = pd.DataFrame(data=[[1700,2],[1800,1]], columns=x.columns.values)
newdata_scaled = scaler.transform(newdata)
reg.predict(newdata_scaled)


#in pandas dataframe we can add a new column by indicating new index
#use del reg_sum['Intercepts'] to delete column
newdata['Predicted_Price'] = reg.predict(newdata)
