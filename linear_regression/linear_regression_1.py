import numpy as np
import pandas as pd
from numpy.linalg import inv
from numpy import dot

X = np.array([[1, 2, 3],[4,5,6]]).reshape(3,2)
Y = np.array([[1,1,1],[2,2,2],[3,3,3]])
t = np.array([1.,1.])
t = t.reshape(2,1)
Z = np.dot(X, t) * Y
print X,'\n', Y,'\n', Z


iris = pd.read_csv('iris.csv')

# normal equation
temp = iris.iloc[:, 1:4]
temp['x0'] = 1
X = temp.iloc[:,[3,0,1,2]]
Y = iris.iloc[:, 0]
Y = Y.reshape(len(iris), 1)
theta_n = dot(dot(inv(dot(X.T, X)), X.T), Y) # theta = (X'X)^(-1)X'Y
print theta_n

# batch gradient decent
theta_g = np.array([1., 1., 1., 1.])
theta_g = theta_g.reshape(4,1)
alpha = 0.1
temp = theta_g
X0 = X.iloc[:, 0].reshape(150, 1)
X1 = X.iloc[:, 1].reshape(150, 1)
X2 = X.iloc[:, 2].reshape(150, 1)
X3 = X.iloc[:, 3].reshape(150, 1)
J = pd.Series(np.arange(800, dtype = float))
for i in range(1):
# theta j := theta j + alpha*(yi - h(xi))*xi
    # tmp = alpha*np.sum((Y- dot(X, theta_g))*X)/150.
    print '~~~~',X[-4:]
    temp[0] = theta_g[0] + alpha*np.sum((Y- dot(X, theta_g))*X0)/150.
    temp[1] = theta_g[1] + alpha*np.sum((Y- dot(X, theta_g))*X1)/150.
    temp[2] = theta_g[2] + alpha*np.sum((Y- dot(X, theta_g))*X2)/150.
    temp[3] = theta_g[3] + alpha*np.sum((Y- dot(X, theta_g))*X3)/150.
    print '~~~~',X[-4:]
    print theta_g
    J[i] = 0.5*np.sum((Y - dot(X, theta_g))**2)

print theta_g