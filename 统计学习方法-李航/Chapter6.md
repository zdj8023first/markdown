# 第六章  逻辑斯蒂回归和最大熵模型

逻辑斯蒂回归logistic regression是统计学习中经典的分类方法

上一章介绍了最大熵问题，是概率模型学习的准则，推广到分类得到最大熵模型 maximum entropy model。

都属于对数线性模型。

- 逻辑斯蒂回归模型
- 最大熵模型
- 学习算法
- 迭代尺度法和拟牛顿法

## 6.1 逻辑斯蒂回归模型

### 6.1.1逻辑斯谛分布 logistic distribution

对于随机变量X，如果X服从以下分布函数和密度函数：
$$
\begin{array}{l}{F(x)=P(X \leqslant x)=\frac{1}{1+\mathrm{e}^{-(x-\mu) / \gamma}}} \\ {f(x)=F^{\prime}(x)=\frac{\mathrm{e}^{-(x-\mu) \gamma}}{\gamma\left(1+\mathrm{e}^{-(x-\mu) / \gamma}\right)^{2}}}\end{array}
$$
感觉有点类似正态分布，并且曲线也有点类似。两个参数分别为位置参数、和形状参数。

其中分布函数属于**逻辑斯蒂函数**，图形为S曲线，即sigmoid curve，曲线关于点（未知参数，1/2）对称



### 6.1.2 二项逻辑斯蒂回归模型

binomial logistic regression model 是一种分类模型（二分类？），其中条件概率分布为参数化的逻辑斯蒂分布

- X取值为实数
- Y取值为0或1

模型的形式化表达如下：
$$
\begin{array}{l}{P(Y=1 | x)=\frac{\exp (w \cdot x)}{1+\exp (w \cdot x)}} \\ {P(Y=0 | x)=\frac{1}{1+\exp (w \cdot x)}}\end{array}
$$
事件的几率odds：

> 发生的概率与不发生的概率的比值，对数几率即取个对数， log odds

对于逻辑斯蒂回归模型中，输出Y=1的对数几率为输入x的线性函数

而：

> 线性函数的值越接近正无穷，概率值越接近1；
>
> 线性函数的值越接近负无穷，概率值越接近0。



### 6.1.3 模型参数估计

这里需要重新复习一下极大似然估计的方法，详细请见下面链接

[极大似然估计](https://blog.csdn.net/zengxiantao1994/article/details/72787849)

似然函数以及对数似然函数为：
$$
\begin{aligned} & \prod_{i=1}^{N}\left[\pi\left(x_{i}\right)\right]^{y_{i}}\left[1-\pi\left(x_{i}\right)\right]^{1-y_{i}} \\ L(w) &=\sum_{i=1}^{N}\left[y_{i} \log \pi\left(x_{i}\right)+\left(1-y_{i}\right) \log \left(1-\pi\left(x_{i}\right)\right)\right] \\ &=\sum_{i=1}^{N}\left[y_{i} \log \frac{\pi\left(x_{i}\right)}{1-\pi\left(x_{i}\right)}+\log \left(1-\pi\left(x_{i}\right)\right)\right] \\ &=\sum_{i=1}^{N}\left[y_{i}\left(w \cdot x_{i}\right)-\log \left(1+\exp \left(w \cdot x_{i}\right)\right]\right.\end{aligned}
$$
通过对对数似然函数求极大值，如梯度下降法或者拟牛顿法。即可求出参数



### 6.1.4 多项逻辑斯蒂回归 multi-nominal logistic regression

将二分类的逻辑斯蒂推广到多分类，即可得到多项逻辑斯蒂回归。



## 6.2 最大熵模型

maximum entropy model 由最大熵原理推导

- 最大熵原理
- 最大熵模型
- 最大熵模型学习

### 6.2.1 最大熵原理

最大熵原理：

> 可以表述为在满足约束条件下的模型集合中选取熵最大的模型



根据上一章介绍的熵的概念，对于离散随机变量，当且仅当X是均匀分布的时候，熵最大。

0 <= H(P) <= log|X|

> 最大熵原理通过熵的最大化表示等可能性，把等可能性变成一个可优化的数值指标

原书图6.2给出了应用最大熵原理进行概率模型选择的几何解释：

单纯性simplex：

> n维欧式空间中 n + 1个仿射无关的点的集合的凸包



### 6.2.2 最大熵模型

将最大熵原理应用到分类问题中去，就得到了最大熵模型

给定训练集，可以确定**联合分布**和**边缘分布**的经验分布，用训练数据集中的频率来表示

用特征函数feature function 描述输入x输出y之间的某个事实，即用特征函数表示约束条件。

如果模型能够获得训练数据中的信息，那么可以假设：

特征函数关于**经验分布P(X,Y)**和**模型以及经验分布P(X)**的期望相等，并将这个等式作为约束条件。

最大熵模型：

>  从满足所有约束条件的模型集合中，条件熵最大的模型，称为最大熵模型

条件熵的具体形式见式6.13



### 6.2.3 最大熵模型的学习

