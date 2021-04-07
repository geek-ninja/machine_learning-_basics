# K mean clustering
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Dataset/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values

#using elbow method to find optimal no. of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    Kmeans = KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0)
    Kmeans.fit(x)
    wcss.append(Kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('The Elbow Method')
plt.xlabel('no. of clusters')
plt.ylabel('wcss')
plt.show()
#We got 5 clusters as optimal 
#Applying k-means to the dataset
Kmeans = KMeans(n_clusters=5,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans = Kmeans.fit_predict(x)

#Visualizing the clusters
plt.scatter(x[y_kmeans == 0,0],x[y_kmeans == 0,1],s=100,c='red',label = 'cluster1')
plt.scatter(x[y_kmeans == 1,0],x[y_kmeans == 1,1],s=100,c='blue',label = 'cluster2')
plt.scatter(x[y_kmeans == 2,0],x[y_kmeans == 2,1],s=100,c='green',label = 'cluster3')
plt.scatter(x[y_kmeans == 3,0],x[y_kmeans == 3,1],s=100,c='cyan',label = 'cluster4')
plt.scatter(x[y_kmeans == 4,0],x[y_kmeans == 4,1],s=100,c='magenta',label = 'cluster5')
plt.scatter(Kmeans.cluster_centers_[:,0],Kmeans.cluster_centers_[:,1],s = 300,c='yellow',label = 'centroids')
plt.title('Cluster of clients')
plt.xlabel('income')
plt.ylabel('spending(1-100)')
plt.legend()
plt.show()