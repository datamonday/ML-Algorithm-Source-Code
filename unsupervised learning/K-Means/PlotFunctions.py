from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
plt.rcParams['figure.figsize'] = (16, 9)
plt.rcParams['figure.dpi'] = 200
plt.style.use('ggplot')

def plot_cluster_data(X, show_dims=2, n_cluster=2, y=None):
    """
    二维/三维可视化聚类数据集
    ----
    param y: 各样本的所属类别
    """
    try:
        n_features = X.shape[1]
        
    except ValueError as e:
        print("[Error] Data X contains at least two features!", e)
            
    else:
        # 类型转换，统一格式为 numpy.array
        if type(X) == pd.core.frame.DataFrame:
            X = X.to_numpy()
            
        fig = plt.figure(tight_layout=True)
        gs = gridspec.GridSpec(1, 1)
        
        if (n_features >= 2) and (show_dims == 2):
            ax = fig.add_subplot(gs[0, 0])
            ax.scatter(X[:, 0], X[:, 1], c=y)
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
        
        elif (n_features >= 3) and (show_dims >= 3):
            ax = Axes3D(fig)
            ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y)
        
        else:
            print("[Error] Please check params!")
            
        plt.show()

def generate_colors(n, name='hsv'):
    '''
    Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.
    '''
    return plt.cm.get_cmap(name, n)
	

def plot_k_means_centroids(X, centroids):
    '''
    绘制 k-means++ 算法的簇中心的初始化选择过程
    '''
    # 类型转换，统一格式为 numpy.array
    if type(X) == pd.core.frame.DataFrame:
        X = X.to_numpy()
        
    plt.figure(tight_layout=True)
    plt.scatter(X[:, 0], X[:, 1], marker = '.', color = 'gray', 
                label = 'data points')
    plt.scatter(centroids[:-1, 0], centroids[:-1, 1], color = 'black', 
                label = 'previously selected centroids')
    plt.scatter(centroids[-1, 0],  centroids[-1, 1],  color = 'red', 
                label = 'next centroid')
    plt.title('Select % d th centroid'%(centroids.shape[0]))
      
    plt.legend()
    plt.show()
	
	
# 聚类中心的移动过程        
def plot_cluster_process_2d(X, new_centroids, old_centroids):
    n_cluster = len(new_centroids)
    cmap = generate_colors(n_cluster)
    
    # 原数据的散点图
    plt.scatter(X[:,0], X[:,1], cmap=cmap)
    
    # 上一次聚类中心
    plt.plot(old_centroids[:, 0], 
             old_centroids[:, 1], 
             'rx', markersize=10, linewidth=5.0)  
    
    # 当前聚类中心
    plt.plot(new_centroids[:, 0],
             new_centroids[:, 1],
             'rx', markersize=10, linewidth=5.0)
    
    # 遍历每个类，画类中心的移动直线
    for j in range(new_centroids.shape[0]): 
        p1 = new_centroids[j, :]
        p2 = old_centroids[j, :]
        plt.plot([p1[0], p2[0]], 
                 [p1[1], p2[1]], 
                 "->", linewidth=2.0)
    return plt
    

# 绘制聚类结果
def plot_cluster_result_2d(X, clusters, centroids):
    # 类型转换，统一格式为 numpy.array
    if type(X) == pd.core.frame.DataFrame:
        X = X.to_numpy()
        
    n_cluster = len(centroids)
    cmap = generate_colors(n_cluster)
    
    fig, ax = plt.subplots()
    for i in range(n_cluster):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, cmap=cmap)
        
    ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black')