# 第2章 感知机

感知机 perceptron 是二分类的线性模型，输入为**实例的特征向量**，输出为**类别1或-1**,也就是

> 对应于输入空间（特征空间）将实例划分为正负两类的分离超平面，属于**判别模型**

损失函数是基于误分类（？），利用**梯度下降法**对损失函数进行极小化，求得感知机模型

学习算法分为：

- 原始形式
- 对偶形式

重要性： 神经网络与支持向量机的基础

1957年由Rosenblatt提出



## 2.1 感知机模型

输入x表示实例的特征向量，输出y表示类别，则

> f(x) = sign(w*x + b)

称为感知机，其中w，b为参数， w为权重向量weight vector, b成为偏置bias，w*x就是正常向量的內积，sign是符号函数。

感知机模型的假设空间是

> 定义在特征空间中的所有线性分类模型linear classification model 或者线性分类器linear classifier。

感知机的几何解释：

> 线性方程 w * x + b = 0 对应于特征空间的超平面S

其中 w是S的法向量，b是S的截距，S将特征空间分为两部分，因此称为分离超平面 separating hyperplane.



例子：二维坐标系中的直线，ax + by + c = 0：

- 特征空间是二维平面的点
- 分离超平面是一条直线
- w = (a,b), b = c
- 直线将二维平面分为两部分，直线上面为正，下面为负



感知机学习的过程，就是求得感知机模型中的参数w, b 的过程。



## 2.2 感知机学习策略

###  2.2.1 数据集的线性可分性

对于给定数据集，如果存在某个超平面，可以将数据集的正、负实例点完全正确的划分到超平面的两侧，则称该数据集是线性可分的。



### 2.2.2 感知机学习策略

如果训练数据是线性可分的，那么感知机的学习目标就是找分离超平面 separating hyperplane，也就是确定w,b!

确定学习策略：

> 定义损失函数并最小化

分析：

> 损失函数采用误分类的点，但是这样关于w,b 不是连续可导函数，优化起来很难

感知机的损失函数：

> 误分类点到超平面的总距离

关于输入空间上任意一点到超平面上的距离定义，与二维空间中点到直线，与三维空间中点到平面的定义一样，即

> | w * x + b | / |w|



对于误分类的点有如下性质：

> -y * ( w * x + b ) > 0  , 用来去掉距离中的绝对值符号

然后误分类点到超平面S的距离可以写成

> -y * (w * x + b ) / |w|

然后就可以得出感知机的损失函数

> 所有误分类点到超平面距离的和（如何确定哪些是误分类的呢？？）, 并且在感知机中，将权重向量weight vector 的模给去掉。



小的总结：

> 感知机即是： sign( w * x + b )
>
> 损失函数： L( w , b ) = - sum ( y * (w * x + b ))， 这里求和是对误分类点集合中的所有误分类点求和

学习策略就是：

> 选取使损失函数最小的 w, b .即是感知机模型



##  2.3 感知机学习算法

由上一节得知

> 感知机学习的问题， 转化为了损失函数的最优化问题。

最优化的方法为 **随机梯度下降法**

- 原始形式
- 对偶形式

证明在线性可分的情况下学习算法的收敛性。

### 2.3.1 原始形式

由损失函数的形式知道，感知机的学习算法是误分类驱动的。

采用**随机梯度下降法** stochastic gradient descent

上述损失函数的梯度为(这里去看下对梯度的理解)

> - TD for w = - sum (y *x )
> - TD for b = - sum(y)

随机选择一个误分类点，对 w,b 更新

> - w  =  w + a * y * x
> - b = b + a * y

这里 0 < a <= 1 ， 称为步长，或者学习率 learning rate.



**算法的原始形式**

> 1. 选取初值w , b
> 2. 训练集中选取数据点 (x, y)
> 3. 如果数据点是误分类点，即 y * ( w * x + b) <= 0, 根据学习率更新 w , b 
> 4. 转至2，直到没有误分类点

算法解释：

> 当有个点是误分类点时，就改变 w , b 的值，使该误分类点往超平面更靠近一点，减少误分类点离超平面之间的距离。
>
> 数学意义就是，更新完 y * ( w  * x + b ) 的值会变大，这个可以自己演算一下。



例子：

对于三个点，正实例点( 3, 3) , (4, 3), 负实例点（1, 1），然后求解感知机模型 f( x ) = sign( w * x + b)

对于随机梯度下降法的理解：

> 随机-  每次随机选择一个实例点，如果是误分类，就更新参数值；
>
> 梯度-  导数？
>
> 下降- 每次一次，迭代的方法



这里如果采取的误分类点的顺序不同，最后得到的感知机模型也有可能不一样。



### 2.3.2 算法收敛性 convergence

收敛性

> 对于线性可分的训练数据集，经过有限次的迭代，算法可以得到一个将训练数据集完全正确划分的分离超平面，也就是感知机模型

为了方便，把b bias 并入到 w 中，得到扩充向量，同时对 x 加入 个 1 ，也得到扩充向量。

收敛性（定理Novikoff）证明，详细见pdf P46， 书中页码 P31.

证明主要基于两点

> 1. 存在超平面,对于扩充向量 | w |  = 1 , 并且有上述定义的点到平面的距离 y ( w * x ) >= a >0
> 2. 误分类次数 <= 

定理表明，误分类的次数是有上届的，即有限次的搜索就可以找到分离超平面 separating hyperplane, 尽管可能不唯一。

为了得到唯一的分离超平面，可以增加约束-》 **线性支持向量机**

 ### 2.3.3 对偶形式

基本思想：

> 将 w 和 b 表示为误分类点的线性组合
>
> w = sum ( a * y * x) , b = sum ( a * y) , 

对偶算法详细见pdf，这里简要介绍下算法的思想

> 输入是训练数据集，以及学习率 learning rate
>
> 输出为b，以及类似与w的线性组合的参数向量

跟原始形式相比，对偶形式的实质是一样的，只是用不同的方式解释了迭代过程。

对偶形式的优点：

> 训练实例仅以內积的形式出现，可以预先将实例间的內积计算出来并存到矩阵中，即是Gram 矩阵



例子：同上述例子



