# -*- coding: utf-8 -*-
"""
@author: datamonday
"""
import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#################### 说明 ##########################
# 为了可读性, 所有numpy和math库的函数均使用.x()调用, #
# 这种方式在实际测试中相比直接函数x()调用耗时        #
####################################################


# =========================== Data Pre-processing =========================== #

def nomalize(X, axis=-1, order=2):
    """
    归一化 (Normalize) the dataset X. data range [0, 1].
    Notes：这种方式是参考库的作者实现方式
    ----
    param X: np.array.
    param axis: axis=0(1) means columns(rows), axis=-1 means last dim.
    param order: order=2 means l2 norm.
    """
    # 针对矩阵每个行向量求向量的l2范数
    l2 = np.atleast_1d(np.linalg.norm(X, ord=order, axis=axis))
    # 将范数为零的元素置1，防止下一步除法操作溢出（NaN）
    l2[l2 == 0] = 1
    
    # 返回归一化的值，各api作用详见ipynb
    return X / np.expand_dims(l2, axis=axis)    
    
    
def standardize(X):
    """
    标准化 (Standardize) the dataset X. mean=0, std=1.
    ----
    param X: np.array
    """
    X_std = X
    # 按列（axis=0）计算均值和标准差
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(np.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
            
    return X_std


def normalize_sk(X, feature_range=(0,1)):
    """
    使用 sklearn的库函数实现归一化，方便反归一化操作.
    """
    scaler = MinMaxScaler(feature_range=feature_range) 
    print("Your input data type is: ", type(X))
    minmax_X = scaler.fit_transform(X)
    
    return scaler, minmax_X


def standardize_sk(X):
    """
    使用 sklearn的库函数实现标准化，方便反标准化操作.
    """
    scaler = StandardScaler()
    print("Your input data type is: ", type(X))
    standard_X = scaler.fit_transform(X)
    
    return scaler, standard_X


# ========================== Statistical Calculate ========================== #
def mean_squared_error(y_true, y_pred):
    """
    Returns the mean squared error between y_true and y_pred
    """
    mse = np.mean(np.power(y_true - y_pred, 2))
    
    return mse


def calculate_variance(X):
    """
    Return the variance of the features in dataset X
    """
    mean = np.ones(np.shape(X)) * X.mean(axis=0)
    # 样本数
    n_samples = np.shape(X)[0]
    # np.diag() 函数返回方阵的第k条对角线的元素, k=0表示主对角线
    variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean), k=0)
    
    return variance


def calculate_std_dev(X):
    """
    Calculate the standard deviations of the features in dataset X
    """
    std_dev = np.sqrt(calculate_variance(X))
    
    return std_dev


def euclidean_distance(x1, x2):
    """
    Calculates the l2 distance between two vectors
    """
    # 实现1: numpy
    # return np.sqrt(np.sum(np.square(x1 - x2) ) )
    
    # 实现2: scipy
    # return pdist(np.vstack([x1, x2]), metric="euclidean")

    # 实现3: Python原生模块
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
        
    return math.sqrt(distance)

    
def calculate_softmax(X, axis=1):
    """
    Calculate the softmax of input vector X.
    softmax(y_i) = \frac{e^{y_i}}{\sum_j e^{y_i}}
    ----------
    Parameters
    ----------
    X : np.array
    axis : int, optional
        axis=1 means calculate by rows. The default is 1.

    Returns softmax(X)
    -------
    """
    # 输入矩阵每行的最大值
    row_max = X.max(axis=axis) 
    # 增加维度, 否则无法运算
    row_max = row_max.reshape(-1, 1)
    # 每行元素均减去最大值, 防止溢出
    X = X - row_max
    # 分子
    X_exp = np.exp(X)
    # 分母, keep_dim保持输入的维度数量不变(尺寸可能变化)
    X_sum = np.sum(X_exp, axis=axis, keep_dim=True)
    
    return X_exp / X_sum

    
def calculate_entropy(y):
     """
     Calculate the entropy of label array y.
     ----------
     Parameters
     ----------
     y : the sample labels.
     cross entropy = -\sum_{i=1}^{k} c_i * log(p_i)
     """
     # 换底公式
     log2 = lambda x: math.log(x) / math.log(2)
     # 所有标签
     unique_labels = np.unique(y)
     # 熵
     entropy = 0
     for label in unique_labels:
         count = len(y[y == label])
         # 某一类的概率
         p = count / len(y)
         # 交叉熵计算公式
         entropy += -p * log2(p)
         
     return entropy
     
 
def calculate_covariance_matrix(X, Y=None):
    """
    Calculate the covariance matrix for the dataset X
    """
    if not Y:
        Y = X
    n_samples = np.shape(X)[0]
    # 方差
    covariance_matrix = (1 / n_samples) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    
    return np.array(covariance_matrix, dtype=float)

    
def calculate_correlation_matrix(X, Y=None):
    """
    Calculate the correlation matrix for the dataset X
    ----------
    
    """
    if not Y:
        Y = X
    n_samples = np.shape(X)[0]
    # 方差
    # 相关系数计算公式的分子, 协方差Cov(X, Y)
    covariance = (1 / n_samples) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))
    # 相关系数的分母 方差开根号 sqrt(D(X)), sqrt(D(Y))
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_Y = np.expand_dims(calculate_std_dev(Y), 1)
    # 相关系数计算公式
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_Y.T))
    
    return np.array(correlation_matrix, dtype=float)


# ============================ Evaluation Metrics =========================== #
def accuray_score(y_true, y_pred):
    """ 
    Compare y_true to y_pred and return the accuracy
    """
    try:
        accuracy = np.sum(y_true == y_pred, axis=0) / len(y_true)
    except ValueError as e:
        print("Please ensure input params must not be zero or None!", e)
    else:
        return accuracy
    
    
def precision_score():
    pass


def recall_score():
    pass


def f_score():
    pass
