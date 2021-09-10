# ml-from-scratch
implementation of machine learning algorithms from scratch in python

## Usage and Inference:

Except PCA, which is implemented in functional method, all other algorithms are implemented in OOP and could be used like scikit-learn fit/predict API.

# **examples:**
  

## Logistic Regression
```
    logistic_regression = LogisticRegression()                   # creating logistic regression object
    coeff, gradient, costs = logistic_regression.fit(            # fitting the algorithm
      X_train,
      y_train,
      n_iters=1500,
      learning_rate=0.3,
      trace=False
      ) 
    pred = logistic_regression.predict(X_test)                   # Inference
    # ------------------
```

## Linear Regression
    
```
    linear_regression = LinearRegression()                       # creating logistic regression object
    params = linear_regression.fit(                              # fitting the algorithm
      X_train,
      y_train,
      ) 
    pred = linear_regression.predict(X_test)                     # Inference
    # ------------------
```

## K Nearest Neighbours
    
```  
    knn = KNNClassifier(K=6)                                     # creating logistic regression object
    knn.fit(                                                     # fitting the algorithm
      X_train,
      y_train,
      ) 
    pred = knn.predict(X_test)                                   # Inference
    # ------------------
```

## feed-forward Neural Networks Classifier (multi-layer perceptron)
    
```
    nn = NeuralNetwork(hidden_size=50)                           # creating logistic regression object
    nn.fit(                                                      # fitting the algorithm
      X_train,
      y_train,
      learning_rate=0.00001,
      epochs=2000,
      trace=False
      ) 
    pred = nn.predict(X_test)                                   # Inference
    # ------------------
```

## Naive Bayes Classifier
    
```
    nb = NaiveBayes()                                            # creating logistic regression object
    nb.fit(                                                      # fitting the algorithm
      X_train,
      y_train,
      ) 
    pred = nb.predict(X_test)                                   # Inference
    # ------------------
```




## TODO:
 <ul>
  <li>implement PCA in OOP</li>
  <li>add decision trees</li>
  <li>create metrics</li>
</ul>
