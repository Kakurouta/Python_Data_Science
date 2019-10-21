import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LinearRegression
#find VIF to evaluate multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
#for feature scaling
from sklearn.preprocessing import StandardScaler
#split train/test data
from sklearn.model_selection import train_test_split

#step1: read data
raw_data = pd.read_csv('example.csv')
raw_data.head() #use .head() to swiftly view top few data

#step2: quick check and preprocessing
raw_data.describe(include='all') #include non-numeric columns

#step2-1: drop unnecessary columns
data = raw_data.drop(['Model'], axis=1) #axis=1 means by column

#step2-2: count and drop nulls 
data.isnull().sum() #true null = 1, thus get null count by sum()
data_no_mv = data.dropna(axis=0) #axis=0 means by row. drop rows that contain null

#step2-3: use istribution plot to check and drop outliers
sns.distplot(data_no_mv['Price'])
q = data_no_mv['Price'].quantile(0.99) #find the value at 99 percentage point
data_1 = data_no_mv[data_no_mv['Price']<q] #drop data in dataframe with this easy conditional statement 

#repeat step2-3 with other numeric columns and get data_2, data_3, ...

#step2-4: reindexing
data_cleaned = data_1.reset_index(drop=True) #drop original index and form new index

#step3: check relationship

#step3-1: plotting scatter plot (predictor x vs. depedend variable y)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (15,3))
ax1.scatter(data_cleaned['Year'], data_cleaned['Price'])
ax1.set_title('Price and Year')
ax2.scatter(data_cleaned['EngineV'], data_cleaned['Price'])
ax2.set_title('Price and EngineV')
ax3.scatter(data_cleaned['Mileage'], data_cleaned['Price'])
ax3.set_title('Price and Mileage')

#step3-2: make logrithmic relationship into linear by taking log(y)
log_price = np.log(data_cleaned['Price'])
data_cleaned['log_price']= log_price
data_cleaned = data_cleaned.drop(['Price'],axis=1) #drop old price because we have log_price

#step3-3: find multicolinearity by VIF (among predictors x)
variables = data_cleaned[['Mileage', 'Year', 'EngineV']]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
data_nm = data_cleaned.drop(['Year'],axis=1) #drop high VIF columns 'cause it has high colinearity with other predictors

#step4: transfer dummies to one-hot encoding
data_nm['Body'].unique() #check unique values within a column
data_d = pd.get_dummies(data_nm, drop_first=True) #drop_first must be true since the last category will be implicitly indicated by all 0s

#step5: finishing preprocessing

#step5-1: check current columns and select desired columns
data_d.columns.values
cols = [] #list desired columns here
data_preprocessed = data_d[cols]
data_preprocessed.head()

#step5-2: define target and inputs
target = data_preprocessed['log_price']
inputs = data_preprocessed.drop(['log_price'], axis=1)

#step6: feature scaling
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

#step7: split training and testing data
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, target, test_size=0.2, random_state=365)

#step8: linear regression

#step8-1: create black box and input data
reg = LinearRegression()
reg.fit(x_train, y_train)

#step8-2: predict
y_hat = reg.predict(x_train)

#step8-3-1: compare with true value by scatter plot
plt.scatter(y_train, y_hat)
plt.xlabel('y_train', size=18)
plt.ylabel('y_hat', size=18)

#step8-3: compare with true value by distribution plot of error y~ (=y^-y)
sns.distplot(y_train-y_hat)
plt.title('Residuals PDF', size=18)

#step8-4: list regression parameters
reg.score(x_train,y_train)
reg.intercept_
reg.coef_
reg_summary = pd.DataFrame(inputs.columns.values,columns=['Features'])
reg_summary['Weights']=reg.coef_.reshape(-1,1)
reg_summary

#step9: doing prediction on test set
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha = 0.2)
plt.xlabel('y_test', size=18)
plt.ylabel('y_hat_test', size=18)

#step10-1: recover log_price to price by taking np.exp()
df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Prediction'])
y_test = y_test.reset_index(drop=True) #reset index
df_pf['Target']= np.exp(y_test)

#step10-2: counting difference
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf.describe()

pd.options.display.max_rows = 999 #set max row number for display
pd.set_option('display.float_format', lambda x:'%.2f' %x) #set float display format
df_pf.sort_values(['Difference%']) #sorted df starts from small Difference% value
