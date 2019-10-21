import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
#simple feature scaling
from sklearn.preprocessing import scale

#load data
data = pd.read_csv('Example_C.csv')
data.head()

#preview data by scatter plot
plt.scatter(data['Satisfaction'],data['Loyalty'])
plt.xlabel('Satisfaction',fontsize=20)
plt.ylabel('Loyalty',fontsize=20)

#2-cluster K-Means
x = data.copy()
kmeans = KMeans(2)
x['cluster'] = kmeans.fit_predict(x)
plt.scatter(x['Satisfaction'], x['Loyalty'], c=x['cluster'],cmap='cool')
plt.show()

#simple feature scaling
x = data.copy()
x_scaled = scale(x)
x_scaled

#This is the same as:
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#x_scaled2 = scaler.fit_transform(x)

#Elbow method from 1 to 10 clusters
wcss = []
for i in range(1,10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,10),wcss)

#transfer x_scaled from array to dataframe for readability
x_scaled = pd.DataFrame(data=x_scaled,columns=['Satisfaction','Loyalty'])
x_scaled

#use 4-cluster chosen by Elbow methos
x2 = x_scaled.copy()
kmeans = KMeans(4)
x['cluster'] = kmeans.fit_predict(x_scaled)
plt.scatter(x['Satisfaction'], x['Loyalty'], c=x['cluster'],cmap='rainbow')
plt.show()

#clustermap
sns.clustermap(x_scaled, cmap='mako')