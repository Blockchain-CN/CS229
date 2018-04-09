#coding=utf-8

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

class LinearRegression():
    def __init__(self): # 新建变量
        self.w = None

    def fit(self, X, y):#训练集的拟合
        X = np.insert(X, 0, 1, axis=1)#增加一个维度
        print (X.shape)
        X_ = np.linalg.inv(X.T.dot(X))#公式求解
        self.w = X_.dot(X.T).dot(y)

    def predict(self, X):#测试集的测试反馈
        #h(theta)=theta.T.dot(X)
        # Insert constant ones for bias weights
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred

def mean_squared_error(y_true, y_pred):
#真实数据与预测数据之间的差值（平方平均）
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse

def main():
    #第一步：导入数据
    # Load the diabetes dataset
    diabetes = datasets.load_diabetes()

    # Use only one feature
    X = diabetes.data[:, np.newaxis, 2]
    print (X.shape)

    #第二步：将数据分为训练集以及测试集
    # Split the data into training/testing sets
    x_train, x_test = X[:-20], X[-20:]

    # Split the targets into training/testing sets
    y_train, y_test = diabetes.target[:-20], diabetes.target[-20:]

    #第三步：导入线性回归类（之前定义的）
    clf = LinearRegression()
    clf.fit(x_train, y_train)#训练
    y_pred = clf.predict(x_test)#测试

    #第四步：测试误差计算（需要引入一个函数）
    # Print the mean squared error
    print ("Mean Squared Error:", mean_squared_error(y_test, y_pred))

    #matplotlib可视化输出
    # Plot the results
    plt.scatter(x_test[:,0], y_test,  color='black')#散点输出
    plt.plot(x_test[:,0], y_pred, color='blue', linewidth=3)#预测输出
    plt.show()

if __name__ == "__main__":
    main()