import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Dataset/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

#using dendrogram to find the optimal of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x,method='ward')) #method = 'ward' used to minimize the variance
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Ecludian distance')
plt.show()
#so we got 5 clusters

#Fitting the model to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(x)
#Visualization the clusters

plt.scatter(x[y_hc == 0,0],x[y_hc == 0,1],s=100,c='red',label = 'cluster1')
plt.scatter(x[y_hc == 1,0],x[y_hc == 1,1],s=100,c='blue',label = 'cluster2')
plt.scatter(x[y_hc == 2,0],x[y_hc == 2,1],s=100,c='green',label = 'cluster3')
plt.scatter(x[y_hc == 3,0],x[y_hc == 3,1],s=100,c='cyan',label = 'cluster4')
plt.scatter(x[y_hc == 4,0],x[y_hc == 4,1],s=100,c='magenta',label = 'cluster5')

plt.title('Cluster of clients')
plt.xlabel('income')
plt.ylabel('spending(1-100)')
plt.legend()
plt.show()