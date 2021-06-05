import numpy as np
from functools import reduce
import re
import random
random.seed(2020)
from sklearn.naive_bayes import MultinomialNB


def load_dataset():
    """
    加载评论数据集，假设数据集已经按照单词切分好
    :return: 返回数据集和标签
    """
    # 切分的样本
    post_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]

    # 类别标签向量，1代表侮辱类 ，0代表非侮辱类
    class_vector = [0, 1, 0, 1, 0, 1]

    return post_list, class_vector


def create_vocab_list(dataset):
    """
    将切分的样本整理成不重复的词汇表(词向量)
    :param dataset: 切分的实验样本
    :return: 词汇表
    """
    # 创建一个空的不重复的列表
    vocab_set = set([])
    for doc in dataset:
        # 取并集
        vocab_set = vocab_set | set(doc)
    return list(vocab_set)
    # return np.array(list(vocab_set))


def set_word2vec(vocab_list, input_data):
    """
    根据vocab_list词汇表，将input_data向量化，向量的每个元素为1或0
    :type vocab_list: list
    :param vocab_list: createVocabList返回的列表
    :param input_data: 切分的词条列表
    :return: 文档向量 (词向量)
    """
    # 初始化向量为零向量
    word_vector = [0] * len(vocab_list)

    for word in input_data:
        if word in vocab_list:
            # 如果输入数据中的词汇在词汇表中，则词汇向量对应的元素置一
            word_vector[vocab_list.index(word)] = 1
        else:
            print(f"the word {word} is not in VocabularyList!")

    return word_vector


def train_naive_bayes(train_matrix, train_y, laplace=True):
    """
    朴素贝叶斯分类器训练
    :param train_matrix: 训练样本
    :param train_y: 训练样本标签
    :param laplace: 拉普拉斯平滑
    :return: 返回预测的两类概率向量以及文档中属于侮辱性的概率
    """
    # 计算训练的文档数目
    n_docs = len(train_matrix)
    # 计算每篇文档的词条数
    n_words_per_doc = len(train_matrix[0])
    # 文档属于侮辱类的概率
    prob_abusive = sum(train_y) / float(n_docs)
    # 创建数组，用于存储单词属于0和1类的概率，np.zeros初始化为0】
    prob_0 = np.zeros(n_words_per_doc)
    prob_1 = np.zeros(n_words_per_doc)
    # 分母初始化为 0.0
    prob_0_denominator = 0.0
    prob_1_denominator = 0.0

    if laplace:
        # 分母初始化为 2.0 (二分类)
        prob_0_denominator = 2.0
        prob_1_denominator = 2.0

    for i in range(n_docs):
        # 统计属于侮辱类的条件概率所需的数据，即 P(w0|1),P(w1|1),P(w2|1)···
        if train_y[i] == 1:
            prob_1 += train_matrix[i]
            prob_1_denominator += sum(train_matrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)···
        else:
            prob_0 += train_matrix[i]
            prob_0_denominator += sum(train_matrix[i])
    # 词向量中，单词属于1类（非侮辱性）的概率向量
    prob_1_vector = prob_1 / prob_1_denominator
    # 词向量中，单词属于0类的概率向量
    prob_0_vector = prob_0 / prob_0_denominator

    # 返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率
    return prob_0_vector, prob_1_vector, prob_abusive


def navie_bayes_classifer(input_vector, prob_0_vector, prob_1_vector, prob_abusive, log=True):
    """
    贝叶斯分类器
    :param input_vector: 待分类的词向量
    :param prob_0_vector: 属于0类的概率向量
    :param prob_1_vector: 属于1类的概率向量
    :param prob_abusive: 词向量属于1类的概率
    :param log: 防止造成下溢
    :return: 0或1
    """
    # reduce() 函数会对参数序列中元素进行累积。
    prob_1 = reduce(lambda x, y : x * y, input_vector * prob_1_vector) * prob_abusive
    prob_0 = reduce(lambda x, y : x * y, input_vector * prob_0_vector) * prob_abusive

    if log:
        # 对应元素相乘 logA * B = logA + logB，所以这里加上log(pClass1)
        prob_1 = sum(input_vector * prob_1_vector) + np.log(prob_abusive)
        prob_0 = sum(input_vector * prob_0_vector) + np.log(1.0 - prob_abusive)

    print("prob_1:", prob_1)
    print("prob_0:", prob_0)
    if prob_1 > prob_0:
        return 1
    else:
        return 0


def test_nave_bayes(test_vocab):
    post_list, class_vector = load_dataset()
    vocab_list = create_vocab_list(post_list)
    train_matrix = []
    for post_in_doc in post_list:
        train_matrix.append((set_word2vec(vocab_list, post_in_doc)))
    prob_0_vector, prob_1_vector, prob_abusive = train_naive_bayes(train_matrix, class_vector)

    test_vector = np.array(set_word2vec(vocab_list, test_vocab))
    if navie_bayes_classifer(test_vector, prob_0_vector, prob_1_vector, prob_abusive):
        print(test_vocab, '属于侮辱类')  # 执行分类并打印分类结果
    else:
        print(test_vocab, '属于非侮辱类')


if __name__ == '__main__':
    # post_list, class_vector = load_dataset()
    # print("post_list:\n", post_list)
    #
    # vocab_list = create_vocab_list(post_list)
    # print("vocab_list:\n", vocab_list)
    # print("vocab_list.shape:", len(vocab_list))
    #
    # train_matrix = []
    # for post_in_doc in post_list:
    #     train_matrix.append((set_word2vec(vocab_list, post_in_doc)))
    # print("train_matrix:\n", train_matrix)
    # print("train_matrix.shape:", np.array(train_matrix).shape)
    #
    # # --------------------- train Naive Bayes Classifier ---------------------
    # prob_0_vector, prob_1_vector, prob_abusive = train_naive_bayes(train_matrix, class_vector)
    # print("prob_0_vector:\n", prob_0_vector)
    # print("prob_1_vector:\n", prob_1_vector)
    #
    # print("class_vector:", class_vector)
    # # prob_abusive是所有侮辱类的样本占所有样本的概率，从class_vector中可以看出，一用有3个侮辱类，3个非侮辱类。所以侮辱类的概率是0.5
    # print("prob_abusive:", prob_abusive)

    # ------------------- Naive Bayes Classifier predict ---------------------
    # 会发现，算法无法进行分类，p0和p1的计算结果都是0，显然结果错误，需要进行改进——拉普拉斯平滑(Laplace Smoothing)！
    # 另外一个遇到的问题就是下溢出，这是由于太多很小的数相乘造成的。通过求对数可以避免下溢出或者浮点数舍入导致的错误。
    test_vocab1 = ['love', 'my', 'dalmation']
    test_nave_bayes(test_vocab1)

    test_vocab2 = ['stupid', 'garbage']
    test_nave_bayes(test_vocab2)


def bag_word2vec(vocab_list, input_set):
    """
    根据 vocab_list词汇表，构建词袋模型
    :param vocab_list: creat_vocab_list 返回的词汇表（列表）
    :param input_set: 切分的词条列表
    :return: 文档向量（词袋模型）
    """
    vocab_vector = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            vocab_vector[vocab_list.index(word)] += 1

    return vocab_vector


def str_to_list(text):
    """
    接收一个大字符串并将其解析为字符串列表
    :param text: 大字符串
    :return: 字符串列表
    """
    # 将特殊符号作为切分标志进行字符串切分，即非字母、非数字
    list_of_tokens = re.split(r'\W+', text)

    return [token.lower() for token in list_of_tokens if len(token) > 2]


def spam_classifier(sklearn=True):
    """
    垃圾邮件分类
    ham：废垃圾邮件；spam：垃圾邮件
    :param sklearn: 使用sklearn的api进行测试
    """
    rootdir = 'D:/Github/ML-Algorithm-Source-Code/'
    spam_filepath = rootdir + 'dataset/email/spam/'
    ham_filepath = rootdir + 'dataset/email/ham/'

    doc_list = []
    class_list = []
    full_text = []

    # 遍历 25个 txt 文件
    for i in range(1, 26):
        # 读取每个垃圾邮件，并字符串转换成字符串列表
        word_list = str_to_list(open(spam_filepath + '%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(1)
        word_list = str_to_list(open(ham_filepath + '%d.txt' % i, 'r').read())
        doc_list.append(word_list)
        full_text.append(word_list)
        class_list.append(0)

    # 创建词汇表，不重复
    vocab_list = create_vocab_list(doc_list)
    dataset = list(range(50))
    test_x = []

    # 从50个邮件中，随机挑选出40个作为训练集，10个做测试集
    # 随机选取10个，构造测试集
    for i in range(10):
        rand_index = int(random.uniform(0, len(dataset)))
        test_x.append(dataset[rand_index])
        del(dataset[rand_index])

    train_x = []
    train_y = []

    # 遍历训练集
    for doc_index in dataset:
        # 将生成的词袋模型添加到训练矩阵中
        train_x.append(set_word2vec(vocab_list, doc_list[doc_index]))
        # 将类别添加到训练集类别标签向量中
        train_y.append(class_list[doc_index])
    # 训练朴素贝叶斯模型
    prob_0_vector, prob_1_vector, prob_spam = train_naive_bayes(np.array(train_x), np.array(train_y))

   # 正确分类计数
    true_count = 0

    # 遍历测试集
    for doc_index in test_x:
        word_vector = set_word2vec(vocab_list, doc_list[doc_index])
        if navie_bayes_classifer(np.array(word_vector), prob_0_vector, prob_1_vector, prob_spam) == class_list[doc_index]:
            true_count += 1
            print("error test set：", doc_list[doc_index])

    print('-' * 32)
    print("Self NB test acc：%.2f%%" % ((float(true_count) / len(test_x)) * 100))

    if sklearn:
        test_x = []
        test_y = []
        # 随机选取10个，构造测试集
        for i in range(10):
            rand_index = int(random.uniform(0, len(dataset)))
            test_x.append(set_word2vec(vocab_list, doc_list[rand_index]))
            test_y.append(class_list[rand_index])
            del (dataset[rand_index])

        clf = MultinomialNB()
        clf.fit(np.array(train_x), np.array(train_y).ravel())
        # clf_pred = clf.predict(np.array(test_x))
        test_acc = clf.score(np.array(test_x), np.array(test_y).ravel())
        print("sklearn NB test acc: ", test_acc)


if __name__ == '__main__':
    spam_classifier()
