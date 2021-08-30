import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist # euclidian distance function 

class KMeans():
    """k-means clustering algorithm object
    """
    def __init__(self, K):
        """k-means clustering with euclidian distance between points

        Args:
            K (int): number of distinct centroids tu calculate for k-means clustering
        """
        self.K = K
        self.fitted = False

    def fit(self,X, max_iter=100, return_logs=True):
        # choosing Inintial cluster centroids from points
        init_cent_indx = np.random.choice(len(X), self.K, replace=False)
        
        #Randomly choosing Centroids 
        self.centroids = X[init_cent_indx, :] # Inintial cluster centroids
        
        #finding the distance between centroids and all the data points
        self.distances = cdist(X, self.centroids ,'euclidean')
        
        # assigning centroid with the minimum distance
        self.labels = np.array([np.argmin(i) for i in self.distances]) #Step 3
        
        # cost function
        J = 0
        for cluster in range(self.K):
            within_cluster_J = self.distances[self.labels==cluster][:,cluster].mean()
            J += within_cluster_J
        
        cost_log = []
        labels_log = []
        centroid_log = []
        
        # Repeating the algorithm for a defined number of max_iter
        for _ in range(max_iter): 
            
            cost_log.append(J)
            labels_log.append(self.labels)
            centroid_log.append(self.centroids)
            
            self.centroids = []
            for cluster in range(X):
                # Updating Centroids
                temp_cent = X[self.labels==cluster].mean(axis=0) 
                self.centroids.append(temp_cent)

            self.centroids = np.vstack(self.centroids) #Updated Centroids 
            self.distances = cdist(X, self.centroids ,'euclidean')
            self.labels = np.array([np.argmin(i) for i in self.distances])
            
            # calculating cost function(J)
            J = 0
            for cluster in range(self.K):
                within_cluster_J = self.distances[self.labels==cluster][:,cluster].mean()
                J += within_cluster_J
        
        self.log = (cost_log, labels_log, centroid_log)
        self.fitted = True
        if return_logs:
            return self.labels ,self.log
        else:
            return self.labels
        

    def predict(self, points):
        """assign new points to their nearest cluster centroid

        Args:
            points (pd.Dataframe): new points whose labels shall be calculated with respect to fitted clusters

        """
        if not self.fitted:
            raise Exception("not fit")
        self.pred_distances = cdist(points, self.centroids ,'euclidean')
        self.pred_labels = np.array([np.argmin(i) for i in self.pred_distances])

        return self.pred_labels

