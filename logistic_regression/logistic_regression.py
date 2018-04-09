# coding=utf-8

"""
https://www.cnblogs.com/sumai/p/5221067.html
"""

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv
import matplotlib.pyplot as plt

iris = pd.read_csv('iris.csv')
dummy = pd.get_dummies(iris['Species'])
iris = pd.concat([iris, dummy], axis =1)
iris = iris.iloc[0:100, 1:]
print iris.shape

# 构建Logistic Regression , 对Species是否为setosa进行分类 setosa ~ Sepal.Length
# Y = g(BX) = 1/(1+exp(-BX))
def logit(x):
    return 1./(1+np.exp(-x))

# 整理出X矩阵 和 Y矩阵
temp = pd.DataFrame(iris.iloc[:, 0])
temp['x0'] = 1
X = temp.iloc[:, [1, 0]]
Y = iris['setosa'].reshape(len(iris), 1)

# 批量梯度上升降
m, n  = X.shape
alpha = 0.0065
theta_g = np.zeros((n,1))
maxCycles = 3000
J = pd.Series(np.arange(maxCycles, dtype = float))

for i in range(maxCycles):
    h = logit(dot(X, theta_g))
    J[i] = -(1/100.)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h)) #计算损失函数值
    error = h - Y #误差
    grad = dot(X.T, error) #梯度
    theta_g -= alpha * grad
print theta_g


# newton method
theta_n = np.zeros((n,1))
maxCycles = 10
C = pd.Series(np.arange(maxCycles, dtype = float)) #损失数值
for i in range(maxCycles):
    h = logit(dot(X, theta_n)) #估计值
    C[i] = -(1/100.)*np.sum(Y*np.log(h)+(1-Y)*np.log(1-h)) #计算损失函数值
    error = h - Y #误差
    grad = dot(X.T, error) #梯度
    A = h*(1-h)* np.eye(len(X))
    H = np.mat(X.T)*A*np.mat(X) #海瑟矩阵, H = X`AX
    theta_n -= inv(H)*grad
print theta_n
C.plot()

plt.show()