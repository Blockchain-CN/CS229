
# machine learning PRACTICE  
Andrew NG CS229 extracurricular practice  
[Andrew_NG 机器学习课程](http://open.163.com/special/opencourse/machinelearning.html)

## Linear Regression
1. iris.csv      测试数据
2. linear_regression.py      逻辑回归  
    a. Normal Equation      直接求解最小值，但特征样本集必须满秩  
    b. batch gradient decent        保证朝下降最快的方向走  
    c. stohastic gradient decent        向随机方向走  
    [b VS c](https://www.cnblogs.com/sirius-swu/p/6932583.html)  
    [a VS b/c](https://blog.csdn.net/artprog/article/details/51172025)   
    [学习率过大的问题](https://blog.csdn.net/vcvycy/article/details/79520163)
3. local_weight_linear_regression.py        局部逻辑回归  
    [局部加权线性回归](https://blog.csdn.net/caimouse/article/details/60334243)  
    
## Logistic Regression
1. logistic_regression.py       逻辑回归  
    a. batch gradient ascent        批量梯度上升  
    b. Newton method        牛顿方法  
    [逻辑回归-梯度下降／梯度上升](https://blog.csdn.net/xiaoxiangzi222/article/details/55097570)   
    [牛顿方法原理1](https://www.guokr.com/question/461510/)  
    [牛顿方法原理2](https://www.zhihu.com/question/20690553)  
    [Jacobian & Hessian](https://jingyan.baidu.com/article/cb5d6105c661bc005c2fe024.html)  
    [各种收敛方法对比](https://www.cnblogs.com/shixiangwan/p/7532830.html)  
    [Python 代码](https://www.cnblogs.com/sumai/p/5221067.html)      