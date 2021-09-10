import numpy as np

class NeuralNetwork():
    def __init__(self,hidden_layers_size) -> None:
        self.hidden_layers_size = hidden_layers_size
        self.fitted = False

    def init_weight(self, n_features):
        w1 = np.random.randn(n_features,self.hidden_layers_size)
        w2 = np.random.randn(self.hidden_layers_size,self.hidden_layers_size)
        w3 = np.random.randn(self.hidden_layers_size,1)

        b1 = np.random.randn(self.hidden_layers_size,1)
        b2 = np.random.randn(self.hidden_layers_size,1)
        b3 = np.random.randn(1)

        init_params = {
            'w1' : w1,
            'b1' : b1,
            'w2' : w2,
            'b2' : b2,
            'w3' : w3,
            'b3' : b3,
        }
        return(init_params)

    def eta(self, x):
        ETA = 0.0000000001
        return x+ETA

    def sigmoid(self, z):    
        output = 1 / (1 + np.exp(-self.eta(z)))
        return output

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1.0 - self.sigmoid(z))


    def fit(self, X, y, learning_rate, epochs, trace=True): 
        m,n = X.shape
        y = y.reshape(-1,1)
        init_params = self.init_weight(n)

        w1 = init_params.get('w1')
        b1 = init_params.get('b1')
        w2 = init_params.get('w2')
        b2 = init_params.get('b2')
        w3 = init_params.get('w3')
        b3 = init_params.get('b3')
        
        cache = {}
        self.loss_log = []
        for epoch in range(epochs):
            
            # forward
            z1 = (np.dot(X, w1).T + b1).T
            a1 = self.sigmoid(z1)

            z2 = (np.dot(a1, w2).T + b2).T
            a2 = self.sigmoid(z2)
            
            z3 = (np.dot(a2, w3).T + b3).T
            a3 = self.sigmoid(z3)

        
            yhat = a3
            yhat = self.eta(yhat)
            
            # loss function
            logprobs = np.dot(y.T,(np.log(yhat))) + np.dot(1-y.T,(np.log(1-yhat)))
            loss = (-1/m) * logprobs
            loss = float(np.squeeze(loss))
            self.loss_log.append(loss)


            # backprop
            y_inv = 1 - y
            yhat_inv = 1 - yhat

            dl_wrt_yhat = np.divide(y_inv, self.eta(yhat_inv)) - np.divide(y, self.eta(yhat))
            dl_wrt_z3 = dl_wrt_yhat * self.sigmoid_derivative(z3)

            dl_wrt_w3 = a2.T.dot(dl_wrt_z3)
            dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
            dl_wrt_a2 = dl_wrt_z3.dot(w3.T)
            dl_wrt_z2 = dl_wrt_a2 * self.sigmoid_derivative(z2)

            dl_wrt_w2 = a1.T.dot(dl_wrt_z2)
            dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
            dl_wrt_a1 = dl_wrt_z2.dot(w2.T)
            dl_wrt_z1 = dl_wrt_a1 * self.sigmoid_derivative(z1)

            dl_wrt_w1 = X.T.dot(dl_wrt_z1)
            dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)
            
            # Updating the parameters according to algorithm

            w1 = w1 - learning_rate * dl_wrt_w1
            b1 = b1 - learning_rate * dl_wrt_b1.T
            w2 = w2 - learning_rate * dl_wrt_w2
            b2 = b2 - learning_rate * dl_wrt_b2.T
            w3 = w3 - learning_rate * dl_wrt_w3
            b3 = b3 - learning_rate * dl_wrt_b3.T

            if trace and (epoch%100==0):
                print(f"epoch {epoch} loss: {loss}")
        self.fitted = True
        self.params = {
            'w1' : w1,
            'b1' : b1,
            'w2' : w2,
            'b2' : b2,
            'w3' : w3,
            'b3' : b3,
        }
        return(self.params,self.loss_log)

    def predict_prob(self, X):
        if not self.fitted:
            raise Exception("not fit")

        w1 = self.params.get('w1')
        b1 = self.params.get('b1')
        w2 = self.params.get('w2')
        b2 = self.params.get('b2')
        w3 = self.params.get('w3')
        b3 = self.params.get('b3')

        # forward
        z1 = (np.dot(X, w1).T + b1).T
        a1 = self.sigmoid(z1)

        z2 = (np.dot(a1, w2).T + b2).T
        a2 = self.sigmoid(z2)
        
        z3 = (np.dot(a2, w3).T + b3).T
        a3 = self.sigmoid(z3)
    
        yhat = a3
        return(yhat)
    
    def predict(self, X, treshold=0.5):
        yhat = self.predict_prob(X)
        labels = []

        for prob in yhat:
            if prob > treshold:
                labels.append(1)
            else:
                labels.append(0)
        
        return(np.array(labels))



if __name__ == '__main__':
    
    from sklearn.datasets import load_breast_cancer
    X,y = load_breast_cancer(return_X_y=True)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

    nn = NeuralNetwork(50)
    params,loss_log = nn.fit(X_train, y_train, learning_rate=0.00001, epochs=2000,  trace=True)
    # print(params)
    pred = nn.predict(X_test)
    train_pred = nn.predict(X_train)
    from sklearn.metrics import accuracy_score, auc
    # print(pred)
    # print(y_test)
    print("test accuracy score:",accuracy_score(y_test, pred))
    print("train accuracy score:",accuracy_score(y_train, train_pred))