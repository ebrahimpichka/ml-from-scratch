import numpy as np

class LinearRegression():
    def __init__(self) -> None:
        pass

    def fit(self,X,y):
        """calculates the linear regression coefficients using 

        Args:
            X (np.ndarray): m by n array(matrix) consisted of m samples and n features
            y (np.ndarray): m by 1 array containing labels for each sample
        Return:
            theta (np.ndarray): n+1 regression coefficients
        """
        
        m, n = X.shape
        X = np.hstack([np.ones((m,1)), X])
        self.theta = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y)

        return(self.theta)

    def predict(self, X):
        return(X.dot(self.theta.reshape(-1,1)))