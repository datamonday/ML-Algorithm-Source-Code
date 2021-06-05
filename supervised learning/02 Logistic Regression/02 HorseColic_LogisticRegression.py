# 基础函数库
import numpy as np
# sigmoid function for ndarrays
from scipy.special import expit
import matplotlib.pyplot as plt
import random

# 设置随机种子保证每次测试结果相同
np.random.seed(2021)


def loadDataset(filepath):
    X = []
    Y = []
    datafile = open(filepath)
    lines = datafile.readlines()
    for line in lines:
        lineArray = line.strip().split()
        X.append([1.0, float(lineArray[0]), float(lineArray[1])])
        Y.append(int(lineArray[2]))
    datafile.close()

    return X, Y


def plotDataset(X, Y, hyperplane=False, weights=None):
    """
    按照第一维特征进行可视化（2D plot）
    Params:
        weights:训练好的权重，用于绘制决策边界
    """
    X = np.array(X)
    # 样本个数
    cols = X.shape[0]
    # 正样本
    x1 = []
    y1 = []
    # 负样本
    x2 = []
    y2 = []

    for i in range(cols):
        if int(Y[i] == 1):
            x1.append(X[i, 1])
            y1.append(X[i, 2])
        else:
            x2.append(X[i, 1])
            y2.append(X[i, 2])
    figure = plt.figure(dpi=300)
    ax = figure.add_subplot(111)
    ax.scatter(x1, y1, s=20, c='red', marker='s', alpha=0.6)
    ax.scatter(x2, y2, s=20, c='green', alpha=0.6)
    plt.title('Dataset')
    plt.xlabel('x')
    plt.ylabel('y')

    # 如果需要绘制决策边界，则执行如下操作
    if hyperplane:
        x = np.arange(-3.0, 3.0, 0.1)  # shape=(60,)
        y = np.array((-weights[0] - weights[1] * x) / weights[2]).reshape(-1, )
        print(x.shape, y.shape)
        ax.plot(x, y)
        plt.title('Results')
        plt.xlabel('X1')
        plt.ylabel('X2')

    plt.show()


# # 警告：RuntimeWarning: overflow encountered in exp return 1.0 / (1+exp(-X))
# def sigmoid(X):
#     if X.any() >= 0:
#         return 1.0 / (1+exp(-X))
#     else:
#         return exp(X) / (1+exp(X))


# 改正
def sigmoid(X):
    return expit(X)


def gradAscent(trainX, trainY, return_weights_only=True, alpha=0.001, epochs=500):
    """
    Params:
        trainX:样本
        trainY:样本标签
        alpha:学习率
        epochs:训练轮数
        weights:求得的回归系数数组(最优参数θ)
    """
    # 将输入数据格式化为矩阵
    xMatrix = np.mat(trainX)
    yMatrix = np.mat(trainY).T
    # 得到训练集的样本个数rows和特征数cols
    rows, cols = np.shape(trainX)
    # 初始化权重
    weights = np.ones((cols, 1))
    # 权重迭代变化值
    weights_array = np.array([])
    # 迭代循环
    for epoch in range(epochs):
        h = xMatrix * weights
        y_pred = sigmoid(h)
        error = yMatrix - y_pred
        w_grad = xMatrix.T.dot(error)
        weights = weights + alpha * w_grad
        weights_array = np.append(weights_array, weights)
    weights_array = weights_array.reshape(epochs, cols)
    if return_weights_only:
        return weights
    else:
        return weights, weights_array


def stocGradAscent(trainX, trainY, return_weights_only=True, alpha=0.001, n_iter=200):
    """
    Params:
        trainX:样本
        trainY:样本标签
        alpha:学习率
        n_iter:迭代次数
        weights:求得的回归系数数组(最优参数θ)
        return_weights_only:仅返回权重系数，不返回迭代更新过程中权重的变化
    """
    X = np.array(trainX)

    # 得到训练集的样本个数rows和特征数cols
    rows, cols = np.shape(trainX)
    # 初始化权重
    weights = np.ones(cols)
    # 存储每次更新的回归系数
    weights_array = np.array([])
    # 迭代循环
    for j in range(n_iter):
        # 生成样本索引
        xIndex = list(range(rows))
        for i in range(rows):
            # 降低alpha的大小，每次减小1/(j+i)
            alpha = 4 / (1.0 + j + i) + 0.01
            # 随机选取一个样本
            randIndex = int(random.uniform(0, len(xIndex)))  # 随机索引
            randX = X[xIndex[randIndex]]  # 随机样本
            randY = trainY[xIndex[randIndex]]  # 随机样本标签
            # 随机选取一个样本，计算h
            h = sum(randX * weights)
            y_pred = sigmoid(h)
            error = randY - y_pred
            # 计算梯度
            w_grad = error * randX
            weights = weights + alpha * w_grad

            if not return_weights_only:
                # 添加回归系数到数组
                weights_array = np.append(weights_array, weights, axis=0)
                # 删除已经使用的样本

            del (xIndex[randIndex])

    if not return_weights_only:
        # 重塑权重存储数组维度
        weights_array = weights_array.reshape(n_iter * rows, cols)
        return weights, weights_array
    else:
        return weights


def classifyVector(sample, weights, activation=sigmoid, threshold=0.5):
    """
    根据训练好的权重向量，对输入样本sample根据阈值，进行分类
    """
    y_pred = activation(sum(sample * weights))
    if y_pred > threshold:
        return 1.0
    else:
        return 0.0


def colicTest(method='GD'):
    """
    使用训练集训练，之后使用得到的权重在测试集上测试，并返回准确率
    Params:
        method:使用哪种方式，梯度下降or随机梯度下降
    """
    trainData = open('horseColicTraining.txt')
    testData = open('horseColicTest.txt')
    X = []
    Y = []
    for line in trainData.readlines():
        # 读取一行，原始数据集中使用tab对元素之间进行间隔
        currLine = line.strip().split('\t')
        # 定义一个列表，用于保存当前行的特征数据
        lineArr = []
        # 遍历当前这一行，并将响应的特征数据添加到对应的训练集数组存储
        # 减一是因为python索引从0开始
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        X.append(lineArr)
        # 每行的最后一个元素为样本标签
        Y.append(float(currLine[-1]))

    X = np.array(X)
    # Y = np.array(Y)

    # 注意，不需要绘制权重变化曲线图时，只需要返回权重即可
    if method == 'GD':
        trainWeights = gradAscent(X, Y, return_weights_only=True)
    if method == 'SGD':
        trainWeights = stocGradAscent(X, Y, return_weights_only=True)

    # 在测试集上测试
    # 定义变量，用于保存分对样本的个数
    trueCount = 0
    # 定义变量，用于保存测试样本的总个数
    testCount = 0

    for line in testData.readlines():
        testCount += 1
        currLine = line.strip().split('\t')
        lineArr = []
        # 将当前行构造成一个样本（数组）
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))

        # 对当前样本，根据训练好的权重向量根据阈值进行分类，通过classifyVector函数实现
        lineArr = np.array(lineArr)

        if method == 'SGD':
            test_pred = classifyVector(lineArr, trainWeights)
        else:
            test_pred = classifyVector(lineArr, trainWeights[:, 0])
        true_label = int(currLine[-1])

        if test_pred == true_label:
            trueCount += 1

    # testCount 为float类型
    accuracy = float(trueCount / testCount) * 100
    print("正确率为：%.4f" % accuracy)

    return accuracy


def multiTest(method='SGD', n_tests=10):
    accArr = []
    accSum = 0.0
    for k in range(n_tests):
        acc = colicTest(method)
        accSum += acc
        accArr.append(acc)
        print("Test ID: {}, Acc: {:.4f}".format(k+1, acc))
    print('-' * 32)
    print("Avg Acc by %d: %.4f" %(n_tests, accSum/float(n_tests)))


if __name__ == '__main__':
    colicTest(method='GD')
    colicTest(method='SGD')
    # 多次训练测试
    # multiTest(method='SGD', n_tests=10)

# if __name__ == '__main__':
#     filepath = 'TestBinaryData.txt'
#     X, Y = loadDataset(filepath)
#     plotDataset(X, Y)
#     w = gradAscent(X, Y)
#     print(w)
#     plotDataset(X, Y, True, w)
#
# if __name__ == '__main__':
#     filepath = 'TestBinaryData.txt'
#     X, Y = loadDataset(filepath)
#     plotDataset(X, Y)
#     w, w_array = stocGradAscent(X, Y, return_weights_only=False)
#     print(w)
#     plotDataset(X, Y, hyperplane=True, weights=w)