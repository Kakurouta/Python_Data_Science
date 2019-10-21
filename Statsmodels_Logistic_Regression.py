import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('data.csv')
raw_data.head()

#map binary predictors to numbers
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data['Gender'] = data['Gender'].map({'Female':1, 'Male':0})

y = data['Admitted']
x = data[['Score','Gender']]

#scatter plot
plt.scatter(x,y,color='C0')
plt.xlabel('Score',fontsize=20)
plt.ylabel('Admitted',fontsize=20)
plt.show()

#use statsmodels.api.Logit() to do logistic regression
x1 = sm.add_constant(x)
reg_log = sm.Logit(y,x1) #note that y first
results_log = reg_log.fit()
results_log.summary()

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
results_log.predict()
results_log.pred_table() #confusion matrix

#load test data
test = pd.read_csv('test.csv')
test.head()
test['Admitted'] = test['Admitted'].map({'No':0, 'Yes':1})
test['Gender'] = test['Gender'].map({'Female':1, 'Male':0})

test_actual = test['Admitted'] #y_test
test_data = test.drop(['Admitted'], axis=1) #x_test
test_data = sm.add_constant(test_data)

#self-defined confusion matrix function with 0.5 threshold set
def cm(data,actual_values,model):
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    c = np.histogram2d(actual_values, pred_values,bins=bins)
    print(c)
    coma = c[0]
    accuracy = (coma[0,0]+coma[1,1])/coma.sum()
    return coma, accuracy

cmm = cm(test_data, test_actual, results_log) #returns (confusion matrix(array), accuracy)
