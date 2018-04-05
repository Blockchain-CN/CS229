# coding=utf-8

"""
20180405: Luda
"""

import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import inv

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

# 批量梯度下降
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
print J[-5:]