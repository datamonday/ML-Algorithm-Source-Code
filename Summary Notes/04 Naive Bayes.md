# Naive Bayes Classifier

最为广泛的两种分类模型是决策树模型(Decision Tree Model)和朴素贝叶斯模型（Naive Bayesian Model，NBM）。和决策树模型相比，朴素贝叶斯分类器(Naive Bayes Classifier，NBC)发源于古典数学理论，有着坚实的数学基础，以及稳定的分类效率。同时，NBC模型所需估计的参数很少，对缺失数据不太敏感，算法也比较简单。理论上，NBC模型与其他分类方法相比具有最小的误差率。但是实际上并非总是如此，这是因为NBC模型假设属性之间相互独立，这个假设在实际应用中往往是不成立的，这给NBC模型的正确分类带来了一定影响。（来源：[百度百科](https://baike.baidu.com/item/%E6%9C%B4%E7%B4%A0%E8%B4%9D%E5%8F%B6%E6%96%AF/4925905?fr=aladdin)）

## 1. 基本概念

### 1）条件概率

条件概率：在事件B发生的情况下，事件A发生的概率，用 $P(A|B)$ 表示。

![image-20210405185755191](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405185755191.png)

由文氏图可知，在事件 B 发生的情况下，事件A发生的概率为：

$$P(A|B) = \frac{P(A \cap B)}{P(B)} \tag{1}$$

The probability of **event A given event B** equals the probability of **event A and event B** divided by the probability of **event B**.

因此，

$$P(A \cap B) = P(A|B)P(B) \tag{2}$$

同理可得，

$$P(A \cap B) = P(B|A)P(A) \tag{3}$$



联立上式可得，

$$ P(A|B)P(B) = P(B|A)P(A) \tag{4}$$

即，

$$P(A|B) = \frac{P(B|A)P(A)}{P(B)} \tag{5}$$

---

### 2）全概率公式

假定样本空间S，是两个事件A与A'的和。

![image-20210405185908864](C:\Users\34123\AppData\Roaming\Typora\typora-user-images\image-20210405185908864.png)

上图中，墨绿部分是事件A，蓝色部分是事件A'，它们共同构成了样本空间S。在这种情况下，事件B可以划分成两个部分。

$$P(B) = P(B \cap A) + P(B \cap A') \tag{1}$$

由上一节已知，

$$P(B \cap A) = P(B|A)P(A) \tag{2}$$

故，

$$P(B) = P(B|A)P(A) + P(B|A')P(A') \tag{3}$$

上式（3）称为**全概率公式**。其含义是，如果A和A'构成样本空间的一个划分，则事件B的概率=A和A'的概率分别乘以B对这两个事件的条件概率之和。

将这个公式代入上一节的条件概率公式，得到条件概率的另一种写法：

$$P(A|B) = \frac{P(B|A)P(A)}{P(B|A)P(A) + P(B|A')P(A')} \tag{4}$$

---

### 3）贝叶斯准则

对条件概率进行变形可得：

$$P(A|B) = P(A)\frac{P(B|A)}{P(B)} \tag{1}$$

- P(A)称为**"先验概率"（Prior probability）**，即在B事件发生之前，对A事件概率的一个判断。
- P(A|B)称为**"后验概率"（Posterior probability）**，即在B事件发生之后，对A事件概率的重新评估。
- P(B|A)/P(B)称为**"可能性函数"（Likelyhood）**，这是一个调整因子，使得预估概率更接近真实概率。

所以，条件概率可以理解成：`后验概率　＝　先验概率 ｘ 调整因子`

**这就是贝叶斯推断的含义。先预估一个"先验概率"，然后加入实验结果，看这个实验到底是增强还是削弱了"先验概率"，由此得到更接近事实的"后验概率"。**

---

### 4）朴素贝叶斯

“朴素”，朴素贝叶斯对条件概率分布做了**条件独立性假设**。 比如下面的公式，假设有n个特征：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_21.jpg)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_21.jpg)

由于每个特征都是独立的，可以进一步拆分：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_22.jpg)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_22.jpg)

这样就可以进行计算，例如：

某个医院早上来了六个门诊的病人，他们的情况如下表所示：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_23.jpg)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_23.jpg)

现在又来了第七个病人，是一个打喷嚏的建筑工人。请问他患上感冒的概率有多大？

根据贝叶斯定理：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_24.jpg)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_24.jpg)

可得：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_28.png)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_28.png)

根据朴素贝叶斯条件独立性的假设可知，"打喷嚏"和"建筑工人"这两个特征是独立的，因此，上面的等式变为：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_29.jpg)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_29.jpg)

这里可以计算：

[![机器学习实战教程（四）：朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_30.jpg)](https://cuijiahua.com/wp-content/uploads/2017/11/ml_4_30.jpg)

因此，这个打喷嚏的建筑工人，有66%的概率是得了感冒。同理，可以计算这个病人患上过敏或脑震荡的概率。比较这几个概率，就可以知道他最可能得什么病。

这就是贝叶斯分类器的基本方法：**在统计资料的基础上，依据某些特征，计算各个类别的概率，从而实现分类**。

在编程时，如果不需要求出所属类别的具体概率，P(打喷嚏) = 0.5和P(建筑工人) = 0.33的概率可以不用求。

---

假设有样本数据集 $D={d_1,d_2,…,d_n}$，对应样本数据的特征属性集为 $X={x_1,x_2,…,x_d}$，类变量为$Y={y_1,y_2,…,y_m}$，即D可以分为y_m类别。其中 $x_1,x_2,…,x_d$ 相互独立且随机。根据贝叶斯准则： 

$$P(Y|X) =  \frac{(P(Y)P(X|Y))}{(P(X))}$$

其中，P(Y|X)称为后验概率，P(Y)称为先验概率，P(X|Y)称为条件概率，P(X)称为证据归一化因子，对于训练样本而言，P(X)是相同的, 因此在比较后验概率时，只比较上式的分子部分即可。可以推出朴素贝叶斯公式为：

$$P(y_i│x_1,x_2,…,x_d )=  \frac{P(Y)∏_(i=1)^d P(x_i |Y)} {P(X)}$$

---

优点：

- 假设数据集的特征之间是相互独立的，因此算法的逻辑性简单。
- 算法较为稳定。当数据呈现不同的特点时，朴素贝叶斯的分类性能不会有太大的差异，鲁棒性较好。
- 当数据的特征之间相互独立时，算法的分类效果较好。

缺点：

- 条件特征独立性假设往往在现实中不成立，在分类时效果不好。相应的改进有半朴素贝叶斯分类器。

---

## 2. 言论过滤（二分类）

以在线社区留言为例。为了不影响社区的发展，需要构建一个快速过滤器来屏蔽侮辱性言论，如果某条留言使用了负面或者侮辱性的语言，那么就将该留言标志为内容不当。对此问题建立两个类型：侮辱类和非侮辱类，分别使用1和0表示。

把文本看成单词向量或者词条向量，也就是说将句子转换为向量。考虑出现所有文档中的单词，再决定将哪些单词纳入词汇表（词汇集合），然后将每一篇文档转换为词汇表上的向量。为简单起见，先假设已经将本文切分完毕，存放到列表中，并对词汇向量进行分类标注。

---

Reference：

> 1. https://www.mathsisfun.com/data/probability-events-conditional.html
> 2. [朴素贝叶斯基础篇之言论过滤器](https://cuijiahua.com/blog/2017/11/ml_4_bayes_1.html)