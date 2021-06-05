import numpy as np
from scipy.spatial.distance import pdist

def calculate_distance_multi_dims(sample, centroid, axis=1, method='euclidean'):
    """
    计算样本与聚类中心的欧氏距离（默认）
    ----
    axis = 1，单次计算多个样本之间的欧氏距离
    axis = 0，单次计算两个样本之间的欧氏距离
    """
    return np.linalg.norm(sample - centroid, axis=axis)


def calculate_distance(sample, centroid, method='euclidean', p=None):
    """
    距离度量
    ----
    method:
        1. 欧氏距离 euclidean
        2. 闵可夫斯基距离 minkowski
        3. 曼哈顿距离 cityblock
        4. 切比雪夫距离 chebyshev
        5. 余弦相似度 cosine
        6. 汉明距离 hamming
        7. 兰氏距离 canberra
        8. 马氏距离 mahalanobis
    """
    sc = np.vstack([sample, centroid])
    # ###################### 1. 欧氏距离 ###################### 
    if method == 'euclidean':
        return pdist(sc, metric="euclidean")
    
    # ###################### 2. 闵可夫斯基距离 ######################
    elif method == 'minkowski':
        if not p:
            print(f"[Error] You mush pass the param p when choose method = {method}!")
        else:
            return pdist(sc, metric="minkowski", p=p)
    
    # ###################### 3. 曼哈顿距离 ######################
    elif method == '':
        return pdist(sc, metric="cityblock")
    
    # ###################### 4. 切比雪夫距离 ######################
    elif method == 'chebyshev':
        return pdist(sc, metric="chebyshev")
    
    # ###################### 5. 余弦相似度距离 ######################
    elif method == 'cosine':
        return 1 - pdist(sc, metric="cosine")
    
    # ###################### 6. 汉明距离 ######################
    elif method == 'hamming':
        return pdist(sc, metric="hamming")
    
    # ###################### 7. 兰氏距离 ######################
    elif method == 'canberra':
        return pdist(sc, metric="canberra")
    
    # ###################### 8. 马氏距离 ######################
    elif method == 'mahalanobis':
        return pdist(sc, metric="mahalanobis")
    
    # ###################### 9. 欧氏距离 ######################
    else:
        print("[Error] You must choose choice 1 to 8 method!")