# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
# 
#      warmUpExercise.py
#      plotData.py
#      gradientDescent.py
#      computeCost.py
#      gradientDescentMulti.py
#      computeCostMulti.py
#      featureNormalize.py
#      normalEqn.py
# 
# 
#  x refers to the population size in 10,000s
#  y refers to the profit in $10,000s
# import libraries

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from utils.warmUpExercise import warmUpExercise
from utils.plotData import plotData
from utils.computeCost import computeCost
from utils.gradientDescent import gradientDescent

# ==================== Part 1: Basic Function ====================
# Complete warmUpExercise.py
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise())

input('Program paused. Press enter to continue.\n')

# ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt('ex1data1.txt', delimiter=',')
X = data[:, 0] 
y = data[:, 1]
m = len(y) # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
plotData(X, y)

input('Program paused. Press enter to continue.\n')

# =================== Part 3: Cost and Gradient descent ===================
X = np.column_stack((np.ones(len(X)), X)) # Add a column of ones to x
theta = np.zeros((X.shape[1], 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')
# compute and display initial cost
J = computeCost(X, y, theta)
print('With theta = [0 ; 0]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, np.array([[-1],[2]]))
print('\nWith theta = [-1 ; 2]\nCost computed = ', J, '\n')
print('Expected cost value (approx) 54.24\n')

input('Program paused. Press enter to continue.\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, J = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('\n', theta[0,0], '\n', theta[1,0], '\n')
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

plt.plot(X, y, 'ro', marker='x', ms=5, label='Data')
plt.plot(X, X.dot(theta), label='Linear Regression')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.legend()
plt.show()

predict1 = np.sum(np.array([1, 3.5]).dot(theta))
print('For population = 35,000, we predict a profit of \n', predict1*10000)
predict2 = np.sum(np.array([1, 7]).dot(theta))
print('For population = 70,000, we predict a profit of ', predict2*10000)

#  ============= Part 4: Visualizing J(theta_0, theta_1) =============
theta0_t = np.linspace(-10, 10, X.shape[0])
theta1_t = np.linspace(-1, 4, X.shape[0])
J = np.zeros((X.shape[0], X.shape[0]))

for i in range(theta0_t.size):
    for j in range(theta1_t.size):
        t = np.array([[theta0_t[i]], [theta1_t[j]]])
        J[i, j] = computeCost(X, y, t)

theta0_t, theta1_t = np.meshgrid(theta0_t, theta1_t)

fig = plt.figure(figsize=(7, 7))
ax = fig.gca(projection='3d')

ax.plot_surface(theta0_t, theta1_t, J, cmap=cm.coolwarm)
ax.set_xlabel(r'$\theta_0$')
ax.set_ylabel(r'$\theta_1$')
ax.set_zlabel(r'J($\theta$)')

# Has some problem, contours are shown but a bit disoriented
CS = ax.contour(theta0_t, theta1_t, J, np.logspace(-2, 3, 20))
ax.scatter(theta[0], theta[1], c='r')

plt.show()