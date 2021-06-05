import sys
import numpy as np
import pandas as pd
from copy import deepcopy
from matplotlib import pyplot as plt


from DistanceMetric import calculate_distance_multi_dims, calculate_distance
from PlotFunctions import plot_cluster_data, generate_colors, plot_k_means_centroids, plot_cluster_process_2d, plot_cluster_result_2d


def do_init_centroids(X, k, method='random'):
    """
    初始化聚类中心
    ----
    X: array, dataset
    k: int, cluster number
    method: str, 'random'(k-means); 'k-means++'
    ----
        KMeans++改进了KMeans算法选择初始质心的方式。
        其核心思想是：在选择一个聚类中心时，距离已有的聚类中心越远的点，被选取作为聚类中心的概率越大。
    """
    # 样本个数
    n_samples  = X.shape[0]
    # 样本特征数
    n_features = X.shape[1]
    
    # 生成样本索引
    indexs = np.arange(0, n_samples)
    # 打乱顺序
    # 此函数仅沿多维数组的第一个轴对数组进行打乱。子数组的顺序改变，但内容不变。
    np.random.shuffle(indexs)
    # 初始化聚类簇中心，shape=(k, n_features)
    centroids = np.zeros((k, n_features))
    
    # 类型转换，统一格式为 numpy.array
    if type(X) == pd.core.frame.DataFrame:
        X = X.to_numpy()
        
    if method == 'k-means++':
        # 从数据集中随机选择一个样本点作为第一个聚类中心
        centroids[0, ] = X[indexs[0], :]
        print(centroids.shape)
        
        # 从剩余样本中选择 k - 1 个聚类中心
        for centroid in range(k - 1):
            # 定义一个列表存储离聚类中心最近的样本点
            dists = []
            
            for i in range(n_samples):
                # 单一样本
                point = X[i, :]
                # 初始化距离
                min_dist = sys.maxsize
                
                # 计算 point 与之前的每一个聚类中心的距离 
                # 选择质心并存储最小距离
                for j in range(len(centroids)):
                    # temp_dist = calculate_distance_multi_dims(point, centroids[j], axis=0)
                    temp_dist = calculate_distance(point, centroids[j], method='euclidean', p=None)
                    # 存储最小距离
                    min_dist = min(min_dist, temp_dist)
                    
                dists.append(min_dist)
                
            # 遍历完样本之后，选择距离最大的数据点作为下一个质心
            max_dist = np.argmax(np.array(dists))
            next_centroid = X[max_dist, :]
            # 存储第二个及其之后的聚类中心
            centroids[centroid+1, :] = next_centroid
            
            # dists 清零
            dists = []
            
    # 随机初始化：即随机从样本中选择 k 个样本点作为初始聚类中心
    else:
        # 取打乱顺序之后的前 k 个样本作为初始聚类中心
        top_k_index = indexs[:k]
        # 用这k个样本的值作为初始化的簇中心
        centroids = X[top_k_index, :]

    return centroids


def k_means(X, n_cluster, init_method='random', n_iter=100, plot_process=False):
    init_centroids = do_init_centroids(X, k=n_cluster, method=init_method)
#     print(init_centroids.shape)
    
    # 类型转换，统一格式为 numpy.array
    if type(X) == pd.core.frame.DataFrame:
        X = X.to_numpy()
        
    # 用于保存聚类中心更新前的值
    old_centroids = np.zeros(init_centroids.shape)
#     print(old_centroids.shape)
    
    # 更新后的聚类中心的值
    new_centroids = deepcopy(init_centroids)
    
    # 用于保存数据所属的簇
    n_samples = len(X)
    clusters = np.zeros(n_samples)
    
    # 迭代标识符，计算新旧聚类中心的距离
    distance_flag = calculate_distance_multi_dims(init_centroids, old_centroids, axis=1)
    
    if n_iter:
        current_iter = 1
        iteration_flag = (current_iter < n_iter)
    # 去掉最大循环次数限制
    else:
        iteration_flag = True
        
    # 若聚类中心不再变化或者迭代次数超过n_iter次(可取消)，则退出循环
    while distance_flag.any() != 0 and iteration_flag:
        # 1. 计算每个样本点所属的簇（距离最近的簇）
        for i in range(n_samples):
            # 样本与k个聚类中心的距离
            distances = calculate_distance_multi_dims(X[i], new_centroids, axis=1)
            # 当前样本与k个聚类中心的最近距离
            cluster = np.argmin(distances)
            # 记录当前样本点所属的聚类中心
            clusters[i] = cluster
            
        # 2. 更新聚类中心
        # 记录更新前的聚类中心
        old_centroids = deepcopy(new_centroids)
        
        # 属于同一个簇的样本点放到一个数组中，然后按照列的方向取平均值
        for i in range(n_cluster):
            points = [X[j] for j in range(len(X)) if clusters[j] == i]
            new_centroids[i] = np.mean(points, axis=0)
            
        # 3. 判断是否满足迭代停止条件
        if current_iter % 5 == 0:
            print(f"[INFO] Iteration {current_iter}：distance_flag = {distance_flag}.")
        
        distance_flag = calculate_distance_multi_dims(new_centroids, old_centroids, axis=1)
        current_iter += 1
        
        if plot_process:    # 如果绘制图像
            plt = plot_cluster_process_2d(X, new_centroids,old_centroids) # 画聚类中心的移动过程
    
    if plot_process:    # 显示最终的绘制结果
        plt.show()
        
    # 返回每个样本所属的类以及更新后的聚类中心
    return clusters, new_centroids
    
    
if __name__ == '__main__':
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=800, n_features=3, centers=4)
    # Importing the dataset
    data = pd.read_csv('xclara.csv')
    print(data.shape)
    
    plot_cluster_data(data)
    plot_cluster_data(X, show_dims=3)
    
    # K-Means
    centroids = do_init_centroids(X, k=3, method='random')
    print(centroids)
    plot_k_means_centroids(X, centroids)
    clusters, centroids = k_means(X, n_cluster=3, init_method='random', n_iter=100, plot_process=True)
    plot_cluster_result_2d(X, clusters, centroids)
    
    # K-Means++
    centroids = do_init_centroids(X, k=3, method='k-means++')
    print(centroids)
    plot_k_means_centroids(X, centroids)
    clusters, centroids = k_means(X, n_cluster=3, init_method='k-means++', n_iter=100, plot_process=True)
    plot_cluster_result_2d(X, clusters, centroids)
    # skelarn K-MEANS
    from sklearn.cluster import KMeans
    

    # Number of clusters
    kmeans = KMeans(n_clusters=3)
    # Fitting the input data
    kmeans = kmeans.fit(X)
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # Centroid values
    centroids = kmeans.cluster_centers_

    # Comparing with scikit-learn centroids
    print("Centroid values")
    print("Scratch")
    print(centroids) # From Scratch
    print("sklearn")
    print(centroids) # From sci-kit learn