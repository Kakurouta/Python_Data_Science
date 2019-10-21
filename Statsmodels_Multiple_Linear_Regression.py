import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#this upgrades the plot quality of matplotlib
import seaborn as sns
sns.set()

#data with three columns: Performance, Price, Random Category (values: 1, 2, 3), and Onsale (value: yes, no)
#need to deal with dummies (non-number), thus named raw_data then data
raw_data_file = 'data2.csv'
raw_data = pd.read_csv(raw_data_file)

#deal with non-number data
data = raw_data.copy()
data['Onsale'] = data['Onsale'].map({'yes':1, 'no':0})

#panda describe tool for statistical summary
data.describe()

y = data['Performance']
x = data[['Price', 'Random Category', 'Onsale']]
xc = sm.add_constant(x)
results = sm.OLS(y,xc).fit()
results.summary()
#it can be observed that the p-value(P>|t|) Random Category is much greater than 0.05, thus insignificant to the prediction