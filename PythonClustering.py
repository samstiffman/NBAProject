# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 12:07:24 2022

@author: samst
"""
#import normalizing
from sklearn import preprocessing
# Pandas for data frames
import pandas as pd
import numpy as np
# Metrics
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
import matplotlib.pyplot as plt
# Clustering
from scipy.cluster.hierarchy import ward, dendrogram
import scipy.cluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import KMeans
## Prep Work

# load in dataset as pandas dataframe
clusteringData = pd.read_csv("cleanedCensusData.csv")
# Normalizing will remove labels so we want to pull them off first then return them
labels = clusteringData.pop("Unnamed: 0")

# Now we just need to normalize the data
min_max_scaler = preprocessing.MinMaxScaler()
scaledData = min_max_scaler.fit_transform(clusteringData)
clusteringData = pd.DataFrame(scaledData, index = labels)



# We will reduce the size of the dataset in twofrom 87-40 to make vizualization possible
sample = clusteringData.sample(n=40)


########### Density Clustering with DBSCAN      ##################
## Since DBSCAN is visual we will use the flattened data set created in R
flattened = pd.read_csv("flattenedClustering.csv", index_col=0)

###### Elbow method to determine Epsilon our neighborhood size
# Ripped directly from https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/
from sklearn.neighbors import NearestNeighbors

nearest_neighbors = NearestNeighbors(n_neighbors=11)
neighbors = nearest_neighbors.fit(flattened)

distances, indices = neighbors.kneighbors(flattened)
distances = np.sort(distances[:,10], axis=0)

fig = plt.figure(figsize=(5, 5))
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
# Conda doesnt include by default
# conda install -c conda-forge kneed
# Kneed will auto find a good epsilon for us
from kneed import KneeLocator

i = np.arange(len(distances))
knee = KneeLocator(i, distances, S=1, curve='convex', direction='increasing', interp_method='polynomial')

fig = plt.figure(figsize=(5, 5))
knee.plot_knee()
plt.xlabel("Points")
plt.ylabel("Distance")

print(distances[knee.knee])
#####
# Set DBSCAN Parameters
neighborhoodSize  = 0.224
minClusterSize = 6

dbscanInstance = DBSCAN(eps = neighborhoodSize, min_samples = minClusterSize)
dbscanInstance.fit(flattened)

clusters = dbscanInstance.labels_

# Visualizing with just 1 index 
plt.scatter(flattened.iloc[:, 0], 
flattened.iloc[:, 1], 
c=dbscanInstance.labels_)
plt.xlabel("Education")
plt.ylabel("Income")
    
########### HEIRARCHICAL CLUSTERING with Ward   ##################

# For this we will use ConsineSim Ward, and also use minkowski p=2, and p=infinity for fun
# Docs: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
# Code below from Professor Ami Gates 


#----------------------------------------------------------
## Hierarchical Clustering using ward and cosine sim


cosdist = 1 - cosine_similarity(sample)
print(np.round(cosdist,3))  
linkage_matrix = ward(cosdist) #define the linkage_matrix 
#using ward clustering pre-computed distances

print(linkage_matrix)
fig = plt.figure(figsize=(20, 20))
plt.title("Heirarchical Clustering with CosSim and Ward")
dn = dendrogram(linkage_matrix ,labels = sample.index)
plt.show()

euclidDist = euclidean_distances(sample)
print(np.round(euclidDist,3))  
linkage_matrix = ward(euclidDist) #define the linkage_matrix 
#using ward clustering pre-computed distances

print(linkage_matrix)
fig = plt.figure(figsize=(20, 20))
plt.title("Heirarchical Clustering with Euclidean Metric and Ward")
dn = dendrogram(linkage_matrix ,labels = sample.index)
plt.show()

l1Dist = manhattan_distances(sample)
print(np.round(l1Dist,3))  
linkage_matrix = ward(l1Dist) #define the linkage_matrix 
#using ward clustering pre-computed distances

print(linkage_matrix)
fig = plt.figure(figsize=(20, 20))
plt.title("Heirarchical Clustering with L1 Metric and Ward")
dn = dendrogram(linkage_matrix ,labels = sample.index)
plt.show()
