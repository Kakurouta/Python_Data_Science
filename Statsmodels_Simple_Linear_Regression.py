import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#use statsmodels to do linear regression
import statsmodels.api as sm

#data with two colums: Performance and Price
data_file = 'data.csv'
data = pd.read_csv(data_file)

#panda describe tool for statistical summary
data.describe()

#scatter plot
y = data['Performance']
x = data['Price']
plt.scatter(x,y)
plt.xlabel('Performance',fontsize=20)
plt.ylabel('Price',fontsize=20)
plt.show()

#add constant to x, then do simple linear regression
xc = sm.add_constant(x)
results = sm.OLS(y,xc).fit()
#statsmodels uses summary() to show regression model summary
results.summary()


y = data['Performance']
x = data['Price']
#coloring scatter plot points by colormap RedYellowGreen and the hue is decided from data value cross-multiplied
plt.scatter(xc ,y ,c=(data['Performance']*data['Price']),cmap='RdYlGn')
plt.xlabel('Performance',fontsize=20)
plt.ylabel('Price',fontsize=20)
#draw regression line with the coefficient from results.summary() 
yr = 0.0017*xc + 0.275
fig = plt.plot(xc, yr ,lw=4,c='orange',label='Regression Line')
plt.show()