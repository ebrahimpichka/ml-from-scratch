import numpy as np



def eta(x):
  ETA = 0.000000001
  return x + ETA

def sigmoid(z):    
    output = 1 / (1 + np.exp(-eta(z)))
    return output

def sigmoid_derivative(z):
    return sigmoid(z) * (1.0 - sigmoid(z))

def train(X, y, learning_rate, epochs, hidden_layers_size=100, trace=True): 
    m,n = X.shape
    y = y.reshape(-1,1)

    w1 = np.random.randn(n,hidden_layers_size)
    w2 = np.random.randn(hidden_layers_size,hidden_layers_size)
    w3 = np.random.randn(hidden_layers_size,1)

    b1 = np.random.randn(hidden_layers_size,1)
    b2 = np.random.randn(hidden_layers_size,1)
    b3 = np.random.randn(1)

    cache = {}
    loss_log = []
    for epoch in range(epochs):
        
        # forward
        z1 = (np.dot(X, w1).T + b1).T
        a1 = sigmoid(z1)

        z2 = (np.dot(a1, w2).T + b2).T
        a2 = sigmoid(z2)
        
        z3 = (np.dot(a2, w3).T + b3).T
        a3 = sigmoid(z3)

    
        yhat = a3
        yhat = eta(yhat)
        
        # loss function
        logprobs = np.dot(y.T,(np.log(yhat))) + np.dot(1-y.T,(np.log(1-yhat)))
        loss = (-1/m) * logprobs
        loss = float(np.squeeze(loss))
        loss_log.append(loss)


        # backprop
        y_inv = 1 - y
        yhat_inv = 1 - yhat

        dl_wrt_yhat = np.divide(y_inv, eta(yhat_inv)) - np.divide(y, eta(yhat))
        dl_wrt_z3 = dl_wrt_yhat * sigmoid_derivative(z3)

        dl_wrt_w3 = a2.T.dot(dl_wrt_z3)
        dl_wrt_b3 = np.sum(dl_wrt_z3, axis=0, keepdims=True)
        dl_wrt_a2 = dl_wrt_z3.dot(w3.T)
        dl_wrt_z2 = dl_wrt_a2 * sigmoid_derivative(z2)

        dl_wrt_w2 = a1.T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0, keepdims=True)
        dl_wrt_a1 = dl_wrt_z2.dot(w2.T)
        dl_wrt_z1 = dl_wrt_a1 * sigmoid_derivative(z1)

        dl_wrt_w1 = X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0, keepdims=True)
        
        # Updating the parameters according to algorithm

        w1 = w1 - learning_rate * dl_wrt_w1
        b1 = b1 - learning_rate * dl_wrt_b1.T
        w2 = w2 - learning_rate * dl_wrt_w2
        b2 = b2 - learning_rate * dl_wrt_b2.T
        w3 = w3 - learning_rate * dl_wrt_w3
        b3 = b3 - learning_rate * dl_wrt_b3.T

        if trace and (epoch%10==0):
            print(f"epoch {epoch} loss: {loss}")
        
    params = {
        'w1' : w1,
        'b1' : b1,
        'w2' : w2,
        'b2' : b2,
        'w3' : w3,
        'b3' : b3,
    }
    return(params,loss_log)


if __name__ == '__main__':
    
    from sklearn.datasets import load_breast_cancer
    X,y = load_breast_cancer(return_X_y=True)

    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

    # logistic_regression = LogisticRegression()
    params,loss_log = train(X_train, y_train, learning_rate=0.00001, epochs=500, hidden_layers_size=100, trace=True)
    # print(params)
    # pred = logistic_regression.predict(X_test)
    # train_pred = logistic_regression.predict(X_train)
    # from sklearn.metrics import accuracy_score, auc
    # # print(pred)
    # # print(y_test)
    # print("test accuracy score:",accuracy_score(y_test, pred))
    # print("train accuracy score:",accuracy_score(y_train, train_pred))