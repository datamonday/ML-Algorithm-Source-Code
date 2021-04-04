# Gaussian discriminat analysis

## 1. 生成式模型

机器学习模型可以分为判别式模型（Discriminative Model）和生成式模型（Generative Model）。其区别在于判别模型是直接从数据特征到标签，而生成模型是从标签到数据特征。具体地，判别式模型通过直接建模后验概率 $P(Y|X)$ 来预测类别 $Y$ ；而生成式模型是先对联合概率分布 $P(X,Y)$ 建模，再求后验概率 $P(Y|X)$ 。即是否使用贝叶斯准则：

$$maxP(Y|X) = \frac{P(X|Y)P(Y)}{P(X)} \thicksim maxP(X|Y)P(Y)$$

其中，$P(Y)$ 称为先验（prior）概率， $P(X|Y)$ 称为类条件概率，又称为“似然”（likehood），分母 $P(X)$ 为用于归一化的证据因子，对于给定的样本，$P(X)$ 与类标记无关，对于所有的类别均相同。为方便推导，可以在优化过程中省略。



---

## 2. 多元高斯分布

多元正态分布也叫多元高斯分布，其两个参数分别是均值向量 $\mu\in\mathbb{R}^n$ 和协方差矩阵 $\Sigma\in\mathbb{R}^{n \times n}$。

协方差矩阵的定义：假设 $X$ 是由 $n$ 个随机变量组成的列向量，并且 $\mu_k$ 是第 $k$ 个元素的期望，即 $\mu_k=E(X_k)$ ，则协方差矩阵可以被定义为：

$$\sum=E\{(X-E(X))(X-E(X)^T)\}=\\   \begin{bmatrix}E[(X_1-\mu_1)(X_1-\mu_1)]& E[(X_1-\mu_1)(X_2-\mu_2)]&\cdots &E[(X_1-\mu_1)(X_n-\mu_n)] \\ E[(X_2-\mu_2)(X_1-\mu_1)]&E[(X_2-\mu_2)(X_2-\mu_2)]&\cdots &E[(X_2-\mu_2)(X_n-\mu_n)]\\ \vdots &\vdots &\ddots &\vdots \\ E[(X_n-\mu_n)(X_1-\mu_1)]&E[(X_n-\mu_n)(X_2-\mu_2)]&\cdots &E[(X_n-\mu_n)(X_n-\mu_n)] \end{bmatrix}$$

矩阵第 $(i,j)$ 个元素表示 $X_i$ 与 $X_j$ 的协方差。协方差矩阵是对称且是半正定的。

- 实对称矩阵：如果有 $n$ 阶矩阵 $A$ ，其矩阵的元素都为实数，且矩阵 $A$ 的转置等于其本身（$a_{ij} = a_{ji}$），则称 $A$ 为实对称矩阵。
- 正定矩阵：设 $M$ 是 $n$ 阶方阵，如果对任何非零向量 $Z$，都有 $Z^TMZ>0$ 成立，则称 $M$ 为正定矩阵。
- 半正定矩阵：半正定矩阵是正定矩阵的推广。设 $A$ 是 $n$ 阶方阵，如果对任何非零向量 $X$，都有 $X^TAX>=0$ 成立，则称 $A$ 为半正定矩阵。

多元高斯分布可以记为 $N(\vec u,\sum)$，其概率密度的具体表达式为：

$$p\big(x,\mu,\Sigma\big)=\frac{1}{(2\pi)^{n/2}\left|\Sigma\right|^{\frac{1}{2}}}exp\Big(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\Big)$$

上式中，$|\Sigma|$ 表示协方差矩阵的行列式，若随机变量 $X$ 服从 $N(\vec u,\Sigma)$，其数学期望是：

$$E(X)=\int\nolimits _xxp(x,\mu,\Sigma) dx=\mu$$

协方差矩阵的定义：

$$cov(X)=\Sigma$$

---

推导：

![image-20210405014741598](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405014741598.png)

---

下面是一些二维高斯分布的概率密度图像：

- 1）均值向量为零向量，单位协方差矩阵乘以不同的系数

  ![这里写图片描述](https://img-blog.csdn.net/20150607224314178)

  以上三幅图的分布的均值向量皆为零向量，协方差矩阵各不相同。其中，左侧的为 $\Sigma=I$ ，即熟知的标准正态分布；中间的为 $\Sigma=0.6I$ ；右侧的为 $\Sigma=2I$。易知， $\Sigma$ 越大，高斯分布越“铺开”， $\Sigma$ 越小，高斯分布越“收缩”。

- 2）均值向量为零向量，协方差矩阵中次对角线元素不同

  ![这里写图片描述](https://img-blog.csdn.net/20150607225900968)

  其协方差矩阵分别为：

  $$\Sigma=\begin{bmatrix}1&0\\0&1\end{bmatrix};\Sigma=\begin{bmatrix}1&0.5\\0.5&1\end{bmatrix};\Sigma=\begin{bmatrix}1&0.8\\0.8&1\end{bmatrix}$$

  其二维图像分别为：

  ![这里写图片描述](https://img-blog.csdn.net/20150607234022098)

  可以看出，增加 $\Sigma$ 的非主对角元素时，概率密度图像沿着45°线($x_1=x_2$)“收缩”。

- 3）均值向量为零向量，协方差矩阵中次对角线元素为负

  ![这里写图片描述](https://img-blog.csdn.net/20150607234450684)

  上面三幅图对应的 $\Sigma$ 分别是：

  $$\Sigma=\begin{bmatrix}1&-0.5\\-0.5&1\end{bmatrix};\Sigma=\begin{bmatrix}1&-0.8\\-0.8&1\end{bmatrix};\Sigma=\begin{bmatrix}3&0.8\\0.8&1\end{bmatrix}$$

  减少主对角元素，概率密度图像在相反的方向上变得“收缩”。
  
- 4）协方差矩阵为单位阵，均指向量不同

  ![这里写图片描述](https://img-blog.csdn.net/20150608000022684)

  上图中的 $\mu$ 分别为：

  $$\mu=\begin{bmatrix}1\\0\end{bmatrix};\mu=\begin{bmatrix}-0.5\\0\end{bmatrix};\mu=\begin{bmatrix}-1\\-1.5\end{bmatrix}$$

  可以看出，改变均指向量后，图像发生了移动。

  


---

## 3. 高斯判别分析

高斯判别分析是典型的生成式模型，其**假设 $P(X|Y)$ 服从高斯分布，$P(Y)$ 服从伯努利分布**，通过训练样本确定高斯分布和伯努利分布的参数，进而通过最大后验概率来进行分类。

高斯判别分析属于机器学习算法中的分类算法，不妨假设样本数据为两种类别，它的大致思想是通过两个先验假设：**一是样本数据的类别 $y$ 在给定的情况下服从伯努利分布，二是不同类别中的样本数据分别服从多元高斯分布**。首先估计出先验概率以及多元高斯分布的均值和协方差矩阵，然后再由贝叶斯公式求出一个新样本分别属于两类别的概率，预测结果取概率值大者。



**假设函数**

![image-20210405015104882](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405015104882.png)

![image-20210405015139566](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405015139566.png)

其中，![[公式]](https://www.zhihu.com/equation?tex=1%28%7By%5E%7B%28i%29%7D%3D1%7D%29)为指示函数，同时假设![[公式]](https://www.zhihu.com/equation?tex=%5CSigma_%7B0%7D%3D%5CSigma_%7B1%7D%3D%5CSigma)，![[公式]](https://www.zhihu.com/equation?tex=%5CSigma)反映一类数据分布的方差，可以看出，最大似然估计的参数值就是基于对样本的一个统计。

下图为一个简单的高斯判别模型示意图：

![这里写图片描述](https://img-blog.csdn.net/20150608104101208)

图中是训练数据集，并且在图上画出了两类数据拟合出的高斯分布的等高线。图中的直线是决策边界，落在直线上的点满足 $p(y=1|x)=0.5$，如果点落在直线上侧，预测 $y=1$，落在直线下侧，预测 $y=0$。

从上图可以看出，高斯判别模型通过建立两类样本的特征模型，**对于二分类问题，通过比较后验概率的大小来得到一个分类边界**。

最小错误贝叶斯决策（Logistic回归）与一维高斯判别模型得到的决策函数也类似于sigmoid函数。



---

## 4. GDA公式推导

已知样本数据含有参数的概率分布，根据统计学的极大似然估计可以推导高斯判别分析模型的损失函数为：

![image-20210405015808698](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405015808698.png)

![image-20210405015830029](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405015830029.png)

![image-20210405015841644](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405015841644.png)

**通过上述公式，所有未知参数都已经估计出，当判断一个新样本 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%28i%29%7D) 时，可分别求出 ![[公式]](https://www.zhihu.com/equation?tex=+p%28y%5E%7B%28i%29%7D%3D0%7Cx%5E%7B%28i%29%7D%29) 和 ![[公式]](https://www.zhihu.com/equation?tex=p%28y%5E%7B%28i%29%7D%3D1%7Cx%5E%7B%28i%29%7D%29) ，取概率更大的那个类**。



---

## 5. GDA与LR比较

- 高斯判别分析假设：![[公式]](https://www.zhihu.com/equation?tex=P%28X%7CY%29)服从高斯分布，![[公式]](https://www.zhihu.com/equation?tex=P%28Y%29)服从伯努利分布；
- Logistic回归假设：![[公式]](https://www.zhihu.com/equation?tex=P%28Y%7CX%2C%5Ctheta%29)服从伯努利分布。

由高斯判别分析可得：

![[公式]](https://www.zhihu.com/equation?tex=p%28y%3D1%7Cx%3B%5Cphi%2C%5Cmu_%7B0%7D%2C%5Cmu_%7B1%7D%2C%5CSigma%29%3D%5Cfrac%7B1%7D%7B1%2Be%5E%7B-%5Ctheta%5E%7BT%7Dx%7D%7D+%5C%5C)

其中，![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta)是参数![[公式]](https://www.zhihu.com/equation?tex=%5Cphi%EF%BC%8C%5Cmu_%7B0%7D%2C%5Cmu_%7B1%7D%2C%5CSigma)的某种函数。即**高斯判别分析是Logistic回归的一种特例**。

**高斯判别模型的假设强于Logistic模型，也就是说Logistic回归模型的鲁棒性更强。这表示在数据量足够大时，更倾向于选择Logistic回归模型。而在数据量较小，且![[公式]](https://www.zhihu.com/equation?tex=P%28X%7CY%29)服从一个高斯分布非常合理时，选择高斯判别分析模型更适合**。

---

> Reference:
>
> 1. [高斯判别分析](https://zhuanlan.zhihu.com/p/95956492)
> 2. [高斯判别分析公式推导](https://zhuanlan.zhihu.com/p/38269530)
> 3. [高斯判别分析算法](https://blog.csdn.net/xiaolewennofollow/article/details/46406447)

