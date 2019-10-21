import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

data = pd.read_csv('clusters.csv')
data.head()

#preview data with scatter plot (location data)
plt.scatter(data['Longitude'],data['Latitude'])
plt.xlim(-180,180) #set axis boundaries
plt.ylim(-90,90)
plt.show()

x = data.iloc[:,1:3] #use iloc to extract [Latitude, Longitude] from [Country, Latitude, Longitude, Language]

#do K-Means
kmeans = KMeans(3) #set total cluster number to be 3
kmeans.fit(x)
idc = kmeans.fit_predict(x) #return an array indicating the cluster id each point belongs to

#add clustering result to dataframe
datac = data.copy() #make a checkpoint by copying
datac['Cluster'] = idc

#plot the clustered data in a scatter plot. color by cluster id
plt.scatter(datac['Longitude'],datac['Latitude'],c=datac['Cluster'],cmap='cool')
plt.xlim(-180,180)
plt.ylim(-90,90)
plt.show()

#How to find the ideal total cluster number? By Elbow Method.

#K-Means inertia
kmeans.inertia_

#iterate for different cluster number and find wcss(Within-Cluster-Sum-of-Squares, K-Means inertia)
wcss = []
for i in range(1,7):
    kmeans = KMeans(i)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
#plot (wcss, inertia) to find elbow point
plt.plot(range(1,7),wcss)
