#install.packages('caret')
library(caret)
#install.packages("amap")
library(amap)
#install.packages("factoextra") # Fviz
library(factoextra)

#Import Data set
setwd("C:\\Users\\samst\\Desktop\\MLProject\\MLProjectMod2")
cleanedCensusData <- read.csv("cleanedCensusData.csv", header=TRUE)

# Prep for clustering
# First we need to remove all text and label data since we are using numerical distance metrics and labels will cause 
censusForClustering <- cleanedCensusData
rownames(censusForClustering) <- cleanedCensusData[,1]
censusForClustering <- censusForClustering[-1]
# Clustering to occur based on labels
censusForClustering <- predict(preProcess(censusForClustering, method=c("range")),censusForClustering)

write.csv(censusForClustering,"censusClusteringData.csv") 
# Next we want to normalize the data since for example grad school rate < college rate < hs rate always
# Min max scaling might make our data better since high rates are different for each category


# We will need to sample for the viz
set.seed(12345)                             # Set seed for reproducibility
sample <- censusForClustering[sample(1:nrow(censusForClustering),30),]


# Lets start with 3 clusters to test, we will use elbow method later to find a better number

k <- 3

kmeansOutput <- Kmeans(sample, centers=k, method="euclidean")
# order will tell us the clusters
clusters <- order(kmeansOutput$cluster)
centroids = kmeansOutput$centers
# Centroids as percentages
100*centroids
fviz_cluster(kmeansOutput,stand=F, sample, main="Euclidean K=3", repel=TRUE)

# What the theory says is that the high grad rates should cluster with higher income rates, but we cannot see this effect with the high D data
# We can use elbow with sum of squares to find a better k
# Set seed for reproducability
# Very simple elbow implementation calculates weighted sum of squares for k 1-15
fviz_nbclust(
  as.matrix(sample), 
  kmeans, 
  k.max = 15,
  method = "wss",
  diss = get_dist(as.matrix(sample), method = "euclidean")
)
# looks from the above plot that k=13 might be the best k

k <- 13

kmeansOutput <- Kmeans(sample, centers=k, method="euclidean")
# order will tell us the clusters
clusters <- order(kmeansOutput$cluster)
centroids = kmeansOutput$centers
# Centroids as percentages
100*centroids
fviz_cluster(kmeansOutput,stand=F, sample, main="Euclidean K=13", repel=TRUE)
# Visualization really doesnt tell us much Since its too high dimensional

### Now we have a significant issue, the data is not easy to visualize so our results from clustering dont appear meaningful
# To remedy this and since it makes sense with the data we will flatten it into only 2 categories: 
# Educational Attainment and Income
# To do this for income I will take the 5 highest income brackets and subtract the 5 lowest
# For education I will take all the categories and subtract off highschool dropout rate
# both of these will be scaled to be weighted averages by dividing by the number of columnes used
# This is done using the rowMeans function
flattenedData <- data.frame((rowMeans(cleanedCensusData[, 3:7])-cleanedCensusData[, 2])/6,(rowMeans(cleanedCensusData[,12:17]-rowMeans(cleanedCensusData[,8:11])/10)))
# Set labels for flattened data
rownames(flattenedData) <- cleanedCensusData[,1]

# Now the data will be Min max scaled again 
flattenedData <- predict(preProcess(flattenedData, method=c("range")),flattenedData)
write.csv(flattenedData,"flattenedClustering.csv") 
# The exact values of the data can no longer be used to infer much since they have been transformed significantly but we 
# Can still see potential associations between income and education as we desire

# We will need to sample for the viz
set.seed(12345)                             # Set seed for reproducibility
sample2 <- flattenedData[sample(1:nrow(flattenedData),87),]


plot(sample2)
# Start with 3 then use elbow

k <- 3
kmeansOutput <- Kmeans(sample2, centers=k, method="euclidean")
# order will tell us the clusters
clusters <- order(kmeansOutput$cluster)
centroids = kmeansOutput$centers
fviz_cluster(kmeansOutput,stand=F, sample2, main="Flattened Data Euclidean K=3", xlab = "Education", ylab = "Income", repel=TRUE)


k <- 5
kmeansOutput <- Kmeans(sample2, centers=k, method="euclidean")
# order will tell us the clusters
clusters <- order(kmeansOutput$cluster)
centroids = kmeansOutput$centers
fviz_cluster(kmeansOutput,stand=F, sample2, main="Flattened Data Euclidean K=5", xlab = "Education", ylab = "Income", repel=TRUE)



k <- 8
kmeansOutput <- Kmeans(sample2, centers=k, method="euclidean")
# order will tell us the clusters
clusters <- order(kmeansOutput$cluster)
centroids = kmeansOutput$centers
fviz_cluster(kmeansOutput,stand=F, sample2, main="Flattened Data Euclidean K=8", xlab = "Education", ylab = "Income", repel=TRUE)




#We can see we actually get some very good clusters of our data
# We can also clearly see some association between higher education and income 
# Of course we need to remember that this is transformed data not raw

# Centroids as percentages
100*centroids


# We can use elbow with sum of squares to find a better k
# Set seed for reproducability
# Very simple elbow implementation calculates weighted sum of squares for k 1-10


fviz_nbclust(
  as.matrix(sample2), 
  kmeans, 
  k.max = 10,
  method = "wss",
  diss = get_dist(as.matrix(sample2), method = "euclidean")
)
# We can see that 3 truly is the optimal number of clusters


