library(KRLS)
library(igraph)
library(Matrix)
library(ggplot2)
library(ggpubr)
library(clusteval)

spectral_clusters <- function(path, k, sigma){
  data_raw <- read.csv(path,header = FALSE, sep = '\t')
  
  #Select only the numeric data
  data_num <- data_raw[3:ncol(data_raw)]
  
  #Calculate distance matrix (W)
  data_dist <- gausskernel(data_num, sigma)
  
  #Calculate degree matrix (D)
  data_degree <- rowSums(data_dist) - 1
  data_degree <- diag(data_degree)
  
  #Laplacian matrix
  data_l <- sqrt(data_dist) %*% (data_degree - data_dist) %*% sqrt(data_dist)
  
  #Eigen decompo
  data_l_eigen <- eigen(data_l)
  data_l_eigenvalues <- as.data.frame(data_l_eigen$values)
  data_l_eigenvectors <- as.data.frame(data_l_eigen$vectors)
  
  starting_point <- ncol(data_l_eigenvectors) - k + 2
  #Pick the eigenvectors corresponding to the smallest eigenvalues
  data_newspace <- data_l_eigenvectors[starting_point : ncol(data_l_eigenvectors)]
  # data_newspace <- data_l_eigenvectors[1:k]
  new_clusters <- kmeans(data_newspace, centers = k)
  return(new_clusters)
}

#Get the clusters
new_clusters_cho <- spectral_clusters('Project_2/cho.txt', 5, 0.6)
new_clusters_iyer <- spectral_clusters('Project_2/iyer.txt', 11, 0.6)

#Get the data for the plots
data_raw_cho <- read.csv('Project_2/cho.txt',header = FALSE, sep = '\t')
data_raw_iyer <- read.csv('Project_2/iyer.txt',header = FALSE, sep = '\t')

#Select only the numeric data
data_num_cho <- data_raw_cho[3:ncol(data_raw_cho)]
data_num_iyer <- data_raw_iyer[3:ncol(data_raw_iyer)]

#Plot the original data
pca_org_cho <- prcomp(data_num_cho)
pca_org_comps_cho <- as.data.frame(pca_org_cho$x)
pca_org_comps_cho <- pca_org_comps_cho[1:2]
pca_org_comps_cho$centroids <- data_raw_cho$V2
pca_org_comps_cho$new_centroids <- new_clusters_cho$cluster
org_cho <- ggplot(pca_org_comps_cho, aes(x = PC1, y = PC2)) +
  geom_point(aes(col = as.character(centroids)), size = 3)+
  labs(title = "cho.txt")+
  xlab("First component")+
  ylab("Second component")

spectral_cho <- ggplot(pca_org_comps_cho, aes(x = PC1, y = PC2)) +
  geom_point(aes(col = as.character(new_centroids)), size = 3)+
  labs(title = "Spectral clustering of cho.txt")+
  xlab("First component")+
  ylab("Second component")

pca_org_iyer <- prcomp(data_num_iyer)
pca_org_comps_iyer <- as.data.frame(pca_org_iyer$x)
pca_org_comps_iyer <- pca_org_comps_iyer[1:2]
pca_org_comps_iyer$centroids <- data_raw_iyer$V2
pca_org_comps_iyer$new_centroids <- new_clusters_iyer$cluster
org_iyer <- ggplot(pca_org_comps_iyer, aes(x = PC1, y = PC2)) +
  geom_point(aes(col = as.character(centroids)), size = 3)+
  labs(title = "iyer.txt")+
  xlab("First component")+
  ylab("Second component")

spectral_iyer <- ggplot(pca_org_comps_iyer, aes(x = PC1, y = PC2)) +
  geom_point(aes(col = as.character(new_centroids)), size = 3)+
  labs(title = "Spectral clustering of iyer.txt")+
  xlab("First component")+
  ylab("Second component")

ggarrange(org_cho, org_iyer, spectral_cho, spectral_iyer,
          nrow = 2, ncol = 2, 
          label.x = c('cho.txt', 'iyer.txt'),
          label.y = c('Original', 'Clustered'))

#Rand and Jaccard indices
rand.index(data_raw_cho$V2, new_clusters_cho$cluster)
rand.index(data_raw_iyer$V2, new_clusters_iyer$cluster)

#Jaccard index
cluster_similarity(data_raw_cho$V2, new_clusters_cho$cluster, similarity = "jaccard")
cluster_similarity(data_raw_iyer$V2, new_clusters_iyer$cluster, similarity = "jaccard")
