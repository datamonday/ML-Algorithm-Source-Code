# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import random
import operator
import math
from copy import deepcopy
import matplotlib.pyplot as plt
# # 将网格线置于曲线之下
# plt.rcParams['axes.axisbelow'] = False
plt.style.use('fivethirtyeight') # 'ggplot'

from PlotFunctions import plot_random_init_iris_sepal, plot_random_init_iris_petal, plot_cluster_iris_sepal, plot_cluster_iris_petal


from sklearn.datasets import load_iris


def load_iris_data():
    data = load_iris()
    # iris数据集的特征列
    features = data['data']
    # iris数据集的标签
    target = data['target']
    # 增加维度1，用于拼接
    target = target[:, np.newaxis]
    
    target_names = data['target_names']
    target_dicts = dict(zip(np.unique(target), target_names))
    
    feature_names = data['feature_names']
    
    # 浅拷贝，防止原地修改
    feature_names = data['feature_names'].copy() # deepcopy(data['feature_names'])
    feature_names.append('label')
    
    df_full = pd.DataFrame(data = np.concatenate([features, target], axis=1), 
                           columns=feature_names)
    # 保存数据集
    df_full.to_csv(str(os.getcwd()) + '/iris_data.csv', index=None)
    
    columns = list(df_full.columns)
    features = columns[:len(columns)-1]
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    
    return df_full, df, class_labels, target_dicts


# 初始化隶属度矩阵 U
def init_fuzzy_matrix(n_sample, c):
    """
    初始化隶属度矩阵，注意针对一个样本，三个隶属度的相加和=1
    ----
    param n_sample: 样本数量
    param c: 聚类数量
    """
    # 针对数据集中所有样本的隶属度矩阵，shape = [n_sample, c]
    fuzzy_matrix = []
    
    for i in range(n_sample):
        # 生成 c 个随机数列表, random.random()方法随机生成[0,1)范围内的一个实数。
        random_list = [random.random() for i in range(c)]
        sum_of_random = sum(random_list)
        # 归一化之后的随机数列表
        # 单个样本的模糊隶属度列表
        norm_random_list = [x/sum_of_random for x in random_list]
        # 选择随机参数列表中最大的数的索引
        one_of_random_index = norm_random_list.index(max(norm_random_list))
        
        for j in range(0, len(norm_random_list)):
            if(j == one_of_random_index):
                norm_random_list[j] = 1
            else:
                norm_random_list[j] = 0
                
        fuzzy_matrix.append(norm_random_list)
    
    return fuzzy_matrix


# 计算FCM的聚类中心
def cal_cluster_centers(df, fuzzy_matrix, n_sample, c, m):
    """
    param df: 数据集的特征集，不包含标签列
    param fuzzy_matrix: 隶属度矩阵
    param c: 聚类簇数量
    param m: 加权指数
    """
    # *字符称为解包运算符
    # zip(*fuzzy_amtrix) 相当于将fuzzy_matrix按列展开并拼接，但并不合并！
    # list(zip(*fuzzy_amtrix)) 包含 列数 个元组。
    fuzzy_mat_ravel = list(zip(*fuzzy_matrix))
    
    cluster_centers = []
    
    # 遍历聚类数量次
    for j in range(c):
        # 取出属于某一类的所有样本的隶属度列表（隶属度矩阵的一列）
        fuzzy_one_dim_list = list(fuzzy_mat_ravel[j])
        # 计算隶属度的m次方
        m_fuzzy_one_dim_list = [p ** m for p in fuzzy_one_dim_list]
        # 隶属度求和，求解聚类中心公式中的分母
        denominator = sum(m_fuzzy_one_dim_list)
        
        # 
        numerator_list = []
        
        # 遍历所有样本，求分子
        for i in range(n_sample):
            # 取出一个样本
            sample = list(df.iloc[i])
            # 聚类簇中心的分子部分，样本与对应的隶属度的m次方相乘
            mul_sample_fuzzy = [m_fuzzy_one_dim_list[i] * val for val in sample]
            numerator_list.append(mul_sample_fuzzy)
        # 计算分子，求和
        numerator = map(sum, list(zip(*numerator_list)))
        cluster_center = [val/denominator for val in numerator]
        cluster_centers.append(cluster_center)
        
    return cluster_centers

# 更新隶属度矩阵，参考公式 (8)
def update_fuzzy_matrix(df, fuzzy_matrix, n_sample, c, m, cluster_centers):
    # 分母的指数项
    order = float(2 / (m - 1))
    # 遍历样本
    for i in range(n_sample):
        # 单个样本
        sample = list(df.iloc[i])
        # 计算更新公式的分母：样本减去聚类中心
        distances = [np.linalg.norm(  np.array(list(  map(operator.sub, sample, cluster_centers[j])  ))  ) \
                     for j in range(c)]
        for j in range(c):
            # 更新公式的分母
            denominator = sum([math.pow(float(distances[j]/distances[val]), order) for val in range(c)])
            fuzzy_matrix[i][j] = float(1 / denominator)
            
    return fuzzy_matrix  #, distances


# 获取聚类中心
def get_clusters(fuzzy_matrix, n_sample):
    # 隶属度最大的那一个维度作为最终的聚类结果
    cluster_labels = []
    for i in range(n_sample):
        max_val, idx = max( (val, idx) for (idx, val) in enumerate(fuzzy_matrix[i]) )
        cluster_labels.append(idx)
    return cluster_labels


# 模糊c均值聚类算法
def fuzzy_c_means(df, fuzzy_matrix, n_sample, c, m, max_iter, init_method='random'):
    """
    param init_random: 聚类中心的初始化方法
            - random: 从样本中随机选择c个作为聚类中心
            - multi_normal: 多元高斯分布采样
    """
    # 样本特征数量
    n_features = df.shape[-1]
    # 初始化隶属度矩阵
    fuzzy_matrix = init_fuzzy_matrix(n_sample, c)
    # 初始化迭代次数
    current_iter = 0
    # 初始化聚类中心
    init_cluster_centers = []
    cluster_centers = []
    # 初始化样本聚类标签的列表，每次迭代都需要保存每个样本的聚类
    max_iter_cluster_labels = []
    # 选择初始化方法
    if init_method == 'multi_normal':
        # 均值列表
        mean = [0] * n_features
        # 多元高斯分布的协方差矩阵，对角阵
        cov = np.identity(n_features)
        for i in range(0, c):
            init_cluster_centers.append(  list( np.random.multivariate_normal(mean, cov) )  )
#     else:
#         init_cluster_centers = [[0.1] * n_features ] * c
        
    print(init_cluster_centers)
    
    while current_iter < max_iter:
        if current_iter == 0 and init_method == 'multi_normal':
            cluster_centers = init_cluster_centers
        else:
            cluster_centers = cal_cluster_centers(df, fuzzy_matrix, n_sample, c, m)
        fuzzy_matrix = update_fuzzy_matrix(df, fuzzy_matrix, n_sample, c, m, cluster_centers)
        cluster_labels = get_clusters(fuzzy_matrix, n_sample)
        max_iter_cluster_labels.append(cluster_labels)
        
        current_iter += 1
        
        print('-' * 32)
        print("Fuzzy Matrix U:\n")
        print(np.array(fuzzy_matrix))
        
    return cluster_centers, cluster_labels, max_iter_cluster_labels


if __name__ == '__main__':
    df_full, df, class_labels, target_dicts = load_iris_data()
    
    
    # 簇数量，鸢尾花数据集有3类
    c = 3
    # 最大迭代次数，防止无限循环
    max_iter = 20
    # 数据量
    n_sample = len(df)
    # 加权指数m，有论文建议 [1.5, 2.5] 范围之间比较好
    m = 1.7
    
    fuzzy_matrix = init_fuzzy_matrix(n_sample, c)
    centers, labels, acc = fuzzy_c_means(df, 
                                     fuzzy_matrix, 
                                     n_sample, 
                                     c, 
                                     m, 
                                     max_iter, 
                                     init_method='multi_normal') # multi_normal, random
    
    
    plot_random_init_iris_sepal(df_full)
    plot_random_init_iris_petal(df_full)
    plot_cluster_iris_sepal(df_full, labels, centers)
    plot_cluster_iris_petal(df_full, labels, centers)
    
    
    
    
    
    
    
    
