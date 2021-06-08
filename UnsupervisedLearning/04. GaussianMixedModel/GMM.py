# -*- coding: utf-8 -*-
"""
@author: datamonday
"""
import os
import sys
import math
import numpy as np
import pandas as pd
sys.path.append('D:\\Github\\ML-Algorithm-Source-Code\\utils')
from utils import normalize, calculate_covariance_matrix

class GaussianMixtureModel():
    """

    高斯混合聚类 (概率模型)，软聚类方法，认为数据从多个高斯分布叠加生成
    -----------
    Parameters:
    -----------
    """
    def __init__(self, k=2, max_iterations=2000, tolerance=1e-8):
        """
        Parameters
        ----------
        k : int
            The number of clusters the algorithm will form. The default is 2.
        max_iterations : int
            The number of iterations the algorithm will run for if it does. The default is 2000.
        tolerance : float
            If the difference of the results from one iteration to the next is
            smaller than this value we will say that the algorithm has converged. The default is 1e-8.
        """
        # 聚类数量
        self.k = k
        # 需要优化的参数
        self.parameters = []
        # 最大迭代次数
        self.max_iterations = max_iterations
        # 最小误差
        self.tolerance = tolerance
        self.responsibilities = []
        # 后验概率(responsibility), 每一个高斯分量对于观察值的贡献
        self.responsibility = None
        # 样本所属聚类
        self.sample_assignments = None
        
    def init_random_gaussians(self, X):
        """
        随机初始化高斯分布
        Parameters
        ----------
        X : np.array
        """
        # 样本数量
        n_samples = np.shape(X)[0]
        # 先验概率
        self.priors = (1 / self.k) * np.ones(self.k)
        
        for i in range(self.k):
            params = {}
            # np.random.choice() 从给定的一维数据生成随机样本
            params['mean'] = X[np.random.choice(range(n_samples))]
            params['cov'] = calculate_covariance_matrix(X)
            self.parameters.append(params)
            
    def multivariate_gaussian(self, X, params):
        """
        构造似然 (Likelihood) 函数, 多元高斯分布
        """
        # 数据集特征数量
        n_features = np.shape(X)[1]
        
        mean = params["mean"]
        covar = params["cov"]
        
        # np.linalg.det()计算数组的行列式
        determinant = np.linalg.det(covar)
        likelihoods = np.zeros(np.shape(X)[0])
        
        for i, sample in enumerate(X):
            dims = n_features
            
            coeff = (1.0 / (math.pow( (2.0 * math.pi), dims/2) * math.sqrt(determinant) ))
            # np.linalg.pinv() 计算矩阵的伪逆(pseudo-inverse) or (Moore-Penrose, 矩阵广义逆)
            exponent = math.exp(-0.5 * 
                                (sample - mean).T.dot(np.linalg.pinv(covar)).dot((sample - mean)) )
            
            likelihoods[i] = coeff * exponent
            
        return likelihoods
    
    def get_likelihoods(self, X):
        """
        Calculate the likelihood over all samples
        """
        n_samples = np.shape(X)[0]
        
        likelihoods = np.zeros((n_samples, self.k))
        for i in range(self.k):
            likelihoods[:, i] = self.multivariate_gaussian(X, self.parameters[i])
            
        return likelihoods
    
    def expectation(self, X):
        """
        Calculate the responsibility
        """
        # Calculate probabilities of X belonging to the different clusters
        weighted_likelihoods = self.get_likelihoods(X) * self.priors
        sum_likelihoods = np.expand_dims(
            np.sum(weighted_likelihoods, axis=1), axis=1)
        # Determine responsibility as P(X|y)*P(y)/P(X)
        self.responsibility = weighted_likelihoods / sum_likelihoods
        # Assign samples to cluster that has largest probability
        self.sample_assignments = self.responsibility.argmax(axis=1)
        # Save value for convergence check
        self.responsibilities.append(np.max(self.responsibility, axis=1))

    def maximization(self, X):
        """
        Update the parameters and priors
        """
        # Iterate through clusters and recalculate mean and covariance
        for i in range(self.k):
            resp = np.expand_dims(self.responsibility[:, i], axis=1)
            mean = (resp * X).sum(axis=0) / resp.sum()
            covariance = (X - mean).T.dot((X - mean) * resp) / resp.sum()
            self.parameters[i]["mean"], self.parameters[
                i]["cov"] = mean, covariance

        # Update weights
        n_samples = np.shape(X)[0]
        self.priors = self.responsibility.sum(axis=0) / n_samples

    def converged(self, X):
        """
        Covergence if || likehood - last_likelihood || < tolerance
        """
        if len(self.responsibilities) < 2:
            return False
        diff = np.linalg.norm(
            self.responsibilities[-1] - self.responsibilities[-2])
        # print ("Likelihood update: %s (tol: %s)" % (diff, self.tolerance))
        return diff <= self.tolerance

    def predict(self, X):
        """
        Run GMM and return the cluster indices
        """
        # Initialize the gaussians randomly
        self.init_random_gaussians(X)

        # Run EM until convergence or for max iterations
        for _ in range(self.max_iterations):
            self.expectation(X)    # E-step
            self.maximization(X)   # M-step

            # Check convergence
            if self.converged(X):
                break

        # Make new assignments and return them
        self.expectation(X)
        
        return self.sample_assignments
 
def load_iris_data():
    from sklearn.datasets import load_iris
    data = load_iris()
    # iris数据集的特征列
    features = data['data']
    # iris数据集的标签
    target = data['target']
    # 增加维度1，用于拼接
    target_2d = target[:, np.newaxis]
    
    target_names = data['target_names']
    target_dicts = dict(zip(np.unique(target_2d), target_names))
    
    feature_names = data['feature_names']
    
    # 浅拷贝，防止原地修改
    feature_names = data['feature_names'].copy() # deepcopy(data['feature_names'])
    feature_names.append('label')
    
    df_full = pd.DataFrame(data = np.concatenate([features, target_2d], axis=1), 
                           columns=feature_names)
    # 保存数据集
    # df_full.to_csv(str(os.getcwd()) + '/iris_data.csv', index=None)
    
    columns = list(df_full.columns)
    features = columns[:len(columns)-1]
    class_labels = list(df_full[columns[-1]])
    df = df_full[features]
    
    return df_full, df, class_labels, target, target_dicts
       
if __name__ == '__main__':
    df_full, df, class_labels, target, target_dicts = load_iris_data()
    
    dataset = normalize(df.values)
    
    gmm = GaussianMixtureModel(k=3)
    results = gmm.predict(dataset)
    print(target)
    print(results)
    
    