import numpy as np 

def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    sqrErrors = np.power(predictions[:,0] - y, 2)
    J = 1/(2*m) * np.sum(sqrErrors)
    return J