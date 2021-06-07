from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.random.seed(1234)
from sklearn import metrics


def knn_classify(train_x, test_x, train_y, k):
    """
    Params:
        train_x:training dataset,ndarry
        test_x:test dataset,ndarry
        train_y:training label,ndarry
        k:knn key paramters,int
    """
    # 训练集的行数
    train_rows = train_x.shape[0]

    ## 求欧氏距离
    # 在列向量方向上重复test_x共1次（横向），行向量方向上重复test_x共train_x次（纵向）
    # 即将单个测试样本test_x重复复制（二维数组中按照行沿列堆叠），最后的diff_matrix的shape与训练集相同
    # 每个测试样本的不同特征值与训练集中的样本的特征值对应相减
    # np.tile(A, reps):Construct an array by repeating A the number of times given by reps
    diff_matrix = np.tile(test_x, (train_rows, 1)) - train_x

    # 二维特征相减后，取平方
    sq_diff_matrix = diff_matrix ** 2

    # sum()所有元素相加，sum(0)列相加，sum(1)行相加；次数表示特征之间的距离和
    sq_distances = sq_diff_matrix.sum(axis=1)

    # 开方，完成欧氏距离的运算，此时训练集中的所有样本与测试样本的距离均保存在distances中
    distances = sq_distances ** 0.5

    # 返回distances中元素从小到大排序后的索引值
    sorted_distance_indices = distances.argsort()

    # 定义一个字典，保存类别
    class_count = {}

    for i in range(k):
        # 取出前k个元素的类别
        top_k_label = train_y[sorted_distance_indices[i]]

        # #dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别出现的次数
        class_count[top_k_label] = class_count.get(top_k_label, 0) + 1

        # key 根据字典的值进行排序, reverse降序排序
        sorted_class_count = sorted(class_count.items(),
                                    key=lambda x: class_count.values(),
                                    reverse=True)

        return sorted_class_count[0][0]


if __name__ == '__main__':
    from sklearn.datasets import load_iris
    import seaborn as sns

    data = load_iris()
    # 标签（3类）
    iris_target = data.target
    # 数据集
    iris_features = pd.DataFrame(data=data.data, columns=data.feature_names)
    # 特征与标签组合的散点可视化
    sns.pairplot(data=iris_features, diag_kind='hist', hue='target')
    plt.show()

    # train and test
    self_knn_pred = []
    for test_x in iris_features.values:
        self_knn = knn_classify(train_x=iris_features, test_x=test_x, train_y=iris_target, k=3)
        self_knn_pred.append(self_knn)

    knn_true = iris_target
    self_knn_pred = np.array(self_knn_pred)
    print(self_knn_pred)

    acc = metrics.accuracy_score(knn_true, self_knn_pred)
    recall = metrics.recall_score(knn_true, self_knn_pred, average='macro')
    print("Acc: ", acc, "\nRecall: ", recall)

