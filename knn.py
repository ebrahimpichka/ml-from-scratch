import numpy as np
import pandas as pd

class KNNClassifier():
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):

        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y,np.ndarray):
            y = pd.DataFrame(y)
        self.X = X
        self.y = y

    def predict(self, points):
        m, n = self.X.shape

        if isinstance(points,np.ndarray):
            points = pd.DataFrame(points)
        
        prediction=[]
        for idx, point in points.iterrows():
            point = point.values
            dists = []
            for index in range(m):
                selected = self.X.iloc[index,:].values
                dists.append(self._euclidean_distance(selected ,point))
            dists = np.array(dists)

            sorted_index = np.argsort(dists)
            K_nearest_x = self.X.iloc[sorted_index[0:self.K],:]
            K_nearest_y = self.y.iloc[sorted_index[0:self.K]]

            label = np.bincount(K_nearest_y.values.ravel()).argmax()
            prediction.append(label)

        self.prediction = pd.Series(prediction)
        return(self.prediction)
        
    def _euclidean_distance(self,u,v):
        return(np.squeeze(np.sqrt(np.dot((u-v).T,(u-v)))))

class KNNRegressor():
    def __init__(self, K):
        self.K = K

    def fit(self, X, y):

        if isinstance(X,np.ndarray):
            X = pd.DataFrame(X)
        if isinstance(y,np.ndarray):
            y = pd.DataFrame(y)
        self.X = X
        self.y = y

    def predict(self, points):
        m, n = self.X.shape

        if isinstance(points,np.ndarray):
            points = pd.DataFrame(points)
        
        prediction=[]
        for idx, point in points.iterrows():
            point = point.values
            dists = []
            for index in range(m):
                selected = self.X.iloc[index,:].values
                dists.append(self._euclidean_distance(selected ,point))
            dists = np.array(dists)

            sorted_index = np.argsort(dists)
            K_nearest_x = self.X.iloc[sorted_index[0:self.K],:]
            K_nearest_y = self.y.iloc[sorted_index[0:self.K]]

            label = np.mean(K_nearest_y.values.ravel())
            prediction.append(label)

        self.prediction = pd.Series(prediction)
        return(self.prediction)
        
    def _euclidean_distance(self,u,v):
        return(np.squeeze(np.sqrt(np.dot((u-v).T,(u-v)))))


if __name__ == '__main__':

    # testing the algorithm

    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor

    X,y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=3)

    my_knn = KNNClassifier(K=6)
    sk_knn = KNeighborsClassifier(n_neighbors=6)

    my_knn.fit(X_train, y_train)
    sk_knn.fit(X_train, y_train)

    print("My KNN results:\n",my_knn.predict(X_test).values.ravel())
    print("SKlearn KNN results:\n",sk_knn.predict(X_test))
