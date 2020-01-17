import numpy as np 
from utils.computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        sqrErrors = X.dot(theta)[:,0] - y
        temp0 = theta[0,0] - (alpha / m) * np.sum(np.multiply(sqrErrors, X[:,0]))
        temp1 = theta[1,0] - (alpha / m) * np.sum(np.multiply(sqrErrors, X[:,1]))
        theta[0,0] = temp0
        theta[1,0] = temp1

        J_history[i] = computeCost(X, y, theta)

    return theta, J_history
