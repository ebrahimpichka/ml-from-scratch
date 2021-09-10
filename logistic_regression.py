import numpy as np
from numpy.core.fromnumeric import trace
from numpy.lib.function_base import gradient


class LogisticRegression():
    def __init__(self) -> None:
        self.fitted = False

    def init_weights(self):
        w = np.random.randn(self.n_features,1)
        b = 0
        self.coeff = {"w": w, "b": b}
        return w,b
        
    def eta(self, x):
        return x + 0.000000000001

    def sigmoid(self, z):
        z = self.eta(z)
        return 1/(1+np.exp(-z))

    def optim_step(self, w, b, X, y):
        m = X.shape[0]

        # Prediction
        y_pred = self.sigmoid(np.dot(X,w)+b)
        y_T = y.T

        cost = (-1/m)*(y_T.dot(np.log(self.eta(y_pred))) + ((1-y_T).dot(np.log(self.eta(1-y_pred)))))
        
        # Grad
        dw = (1/m)*(np.dot(X.T, (y_pred-y)))
        db = (1/m)*(np.sum(y_pred-y))

        grads = {"dw": dw, "db": db}
        
        return grads, cost
    
    
    def fit(self, X, y, learning_rate=0.1, n_iters=100, trace=True):
    
        self.m, self.n_features = X.shape
        w, b = self.init_weights()
        y = y.reshape(-1,1)

        self.costs = []
        for i in range(n_iters):

            grads, cost = self.optim_step(w, b, X, y)
  
            dw = grads["dw"]
            db = grads["db"]

            #weight update
            w = w - (learning_rate * dw)
            b = b - (learning_rate * db)
            if (i % 100 == 0):
                self.costs.append(cost)
                if trace:
                    print("Cost after %i iter: %f" %(i, cost))
            
        #final parameters
        self.coeff = {"w": w, "b": b}
        self.gradient = {"dw": dw, "db": db}
        
        self.fitted = True
        return self.coeff, self.gradient, self.costs
    

    def predict_prob(self, X_pred):
        if not self.fitted:
            raise Exception("not fit")

        w = self.coeff["w"]
        b = self.coeff["b"]

        pred = self.sigmoid(np.dot(X_pred,w)+b)

        return(pred)
        

    def predict(self, X_pred, treshold=0.5):
        probs = self.predict_prob(X_pred)

        class_pred = []
        for prob in probs:
            if prob > treshold:
                class_pred.append(1)
            else:
                class_pred.append(0)
        return(np.array(class_pred))



if __name__ == '__main__':
    
    from sklearn.datasets import load_breast_cancer
    X,y = load_breast_cancer(return_X_y=True)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

    logistic_regression = LogisticRegression()
    coeff, gradient, costs = logistic_regression.fit(X_train, y_train,n_iters=1000,learning_rate=0.01,trace=False)

    pred = logistic_regression.predict(X_test)
    train_pred = logistic_regression.predict(X_train)
    from sklearn.metrics import accuracy_score, auc
    # print(pred)
    # print(y_test)
    print("test accuracy score:",accuracy_score(y_test, pred))
    print("train accuracy score:",accuracy_score(y_train, train_pred))


