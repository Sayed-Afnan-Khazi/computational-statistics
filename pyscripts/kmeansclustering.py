from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Reading the data
data = pd.read_csv("datasets/income.csv")

# Displaying the plotted data
plt.scatter(data['Age'],data['Income'])
plt.show()

#Scaling the data
scaler = MinMaxScaler()
scaler.fit(data[['Income']])
data['Income']=scaler.transform(data[['Income']])
scaler.fit(data[['Age']])
data['Age'] = scaler.transform(data[['Age']])

# Finding K means
km = KMeans(n_clusters=3)
km.fit(data[['Age','Income']])
pY = km.predict(data[['Age','Income']])

data['cluster'] = pY

c0 = data[data['cluster']==0]
c1 = data[data['cluster']==1]
c2 = data[data['cluster']==2]


plt.scatter(data['Age'],data['Income'])
plt.scatter(c0['Age'],c0['Income'])
plt.scatter(c1['Age'],c1['Income'])
plt.scatter(c2['Age'],c2['Income'])
plt.show()