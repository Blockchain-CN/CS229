
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
1. iris.csv      测试数据
2. logistic_regression.py       逻辑回归
    a. batch gradient ascent        批量梯度上升  
    b. Newton method        牛顿方法  
    [逻辑回归-梯度下降／梯度上升](https://blog.csdn.net/xiaoxiangzi222/article/details/55097570)   
    [牛顿方法原理1](https://www.guokr.com/question/461510/)  
    [牛顿方法原理2](https://www.zhihu.com/question/20690553)  
    [Jacobian & Hessian](https://jingyan.baidu.com/article/cb5d6105c661bc005c2fe024.html)  
    [各种收敛方法对比](https://www.cnblogs.com/shixiangwan/p/7532830.html)  
    [python 代码](https://www.cnblogs.com/sumai/p/5221067.html)

## Softmax Regression
1. train.csv      测试数据(minst)
2. softmax.py       softmax 回归
    a. stohastic gradient decent        随机选取样本来更新所有分类组的权重
    [softmax 原理](https://www.zhihu.com/question/23765351)
    [softmax 算法推导](http://ufldl.stanford.edu/wiki/index.php/Softmax%E5%9B%9E%E5%BD%92)
    [python 代码](https://blog.csdn.net/wds2006sdo/article/details/53699778?utm_source=itdadao&utm_medium=referral)

```
区分学习 discriminative learning algorithm  划线区分两个类即可(lr, loss,softmax)
生成学习 Generative Learning algorithms     需要区分两类的归属概率对比(GDA, naive bayes)

形式化的说，判别学习方法是直接对 p(y|x)进行建模或者直接学习输入空间 到输出空间的映射关系，其中，x 是某类样例的特征，y 是某类样例的分类标记。 而生成学习方法是对 p(x|y)（条件概率）和 p(y)（先验概率）进行建模，然后按 照贝叶斯法则求出后验概率 p(y|x)
```

## GDA Gaussian discriminant analysis
感觉就是我毕业设计用的那个方法的规范版

## Naive Bayes
根据先验概率进行后验概率推断，可以对多分类。
与softmax的区别在于：softmax的各维数据可以是无意义的，例如手写字体的像素值展开。
                   bayes的各维数据必须是有概率关系的，比如说是身高(连续特征可以离散化或者用高斯分布模拟概率)，是否存在(0／1)等特征。
由于先验概率可能存在为0的情况，会导致分母为0，因此需要进行平滑(laplace smooth)，原理就是:
        算先验概率{P(H1)=0/0+7 => P(H1)=1/1+8}   (在求H1分类出现的概率时，给H1出现的计数+1，同理给所有非H1出现的情况也+1，如果分3类就是P(H1)=0/0+1+3=>1/1+2+3)
        和条件概率{P(E1|H1)=0/0+2 => P(E1|H1)=1/1+3}    (在求E1特征在H1类中出现的条件概率时，给E1出现的计数+1，同理给所有非E1出现的情况也+1，如果E1有3个同级特征P(E1|H1)=0/0+1+3+4=>1/1+2+3+5)
        的时候给所有分类中的所有特征全部加1。
1. naive_bayes.py
    [naive bayes](https://blog.csdn.net/syoya1997/article/details/78618885)
    [laplace smooth+代码+原理](https://blog.csdn.net/tanhongguang1/article/details/45016421)
    [python 代码](https://blog.csdn.net/li8zi8fa/article/details/76176597)
    [正态分布贝叶斯](http://www.ruanyifeng.com/blog/2013/12/naive_bayes_classifier.html)