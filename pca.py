import pandas as pd
import numpy as np


def PCA(data , k):
    """ 
        calculates the covariance matrix of X, eigen values and eigen vectors of the covariance matrix
        then calculates the K components of X (first K eigen vectors whith the greatest eigen values)

        Parameters
        ----------
        data : array-like, shape (n_samples, n_features)
               Training data, where n_samples is the number of samples and
               n_features is the number of features.

        Returns
        -------
        transformed_X : K components of X (first K eigen vectors whith the greatest eigen values)
        """
    # Normalizing data (Subtrackting the mean of each column from each element of that column)
    data_normalized = (data - data.mean()).copy()
        
    # Calculating Covariance Matrix
    covmat = data_normalized.cov()
    
    # Calculating eigenVectors Matrix (Sorted by the eigenValues)
    eigen_values , eigen_vectors = np.linalg.eig(covmat)
    sorted_indx = np.argsort(eigen_values)[::-1]
    eigen_vectors = eigen_vectors[sorted_indx]
    eigen_values = eigen_values[sorted_indx]
    
    # Transforming Data to new space
    transformed_data = data_normalized.dot(eigen_vectors)
    
    # Returning the first k features with the greatest eigenValue (first k essetial features)
    return(transformed_data.iloc[:,:k])