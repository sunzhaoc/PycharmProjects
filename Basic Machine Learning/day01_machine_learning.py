from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
import jieba
import pandas as pd


def datasets_demo():
    """
    sklearn数据集使用
    :return:
    """
    # 获取数据集
    iris = load_iris()
    print("鸢尾花数据集的返回值: \n", iris)
    # 返回值是一个继承自字典的Bench
    print("鸢尾花的特征值: \n", iris["data"])
    print("鸢尾花的目标值: \n", iris.target)
    print("鸢尾花特征的名字: \n", iris.feature_names)
    print("鸢尾花你目标的名字: \n", iris.target_names)
    print("鸢尾花的描述: \n", iris.DESCR)

    # 数据集划分
    x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=22)
    print("训练集的特征值: \n", x_train, x_train.shape)

    return None


def dict_demo():
    """
    字典特征抽取
    :return:
    """
    data = [{'city': '北京', 'temperature': 100},
            {'city': '上海', 'temperature': 60},
            {'city': '深圳', 'temperature': 30}]

    # 1.实例化一个转换器类
    transfer = DictVectorizer(sparse=False)

    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new)
    print("特征名字: \n", transfer.get_feature_names())

    return None


def count_demo():
    """
    文本特征抽取 CountVectorizer
    :return:
    """
    data = ["Life is short, i like like python", "life is too long, i dislike python"]

    # 1.实例化一个转换器类,
    transfer = CountVectorizer(stop_words=["is", "too"])

    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())
    print("特征名字: \n", transfer.get_feature_names())

    return None


def count_chinese_demo():
    """
    中文文本特征抽取 CountVectorizer
    :return:
    """
    data = ["我 爱 北京 天安门", "天安门 上 太阳 升"]

    # 1.实例化一个转换器类
    transfer = CountVectorizer()

    # 2.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new: \n", data_new.toarray())
    print("特征名字: \n", transfer.get_feature_names())

    return None


def cut_word(text):
    """
    进行中文分词:"我爱北京天安门" -> "我 爱 北京 天安门"
    :param text:
    :return:
    """
    return " ".join(list(jieba.cut(text)))


def count_chinese_demo2():
    """
    中文文本特征抽取,自动分词
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷,明天更残酷,后天更美好,但绝对大部分是死在明天晚上,所以每个人都不要放弃今天",
            "我们看到的从很远星系来的光是在百万年之前发出的,这样当我们看到宇宙时,我们是在看它的过去",
            "如果只用一种方式了解某样事物,你就不会真正了解它,了解事物真正含义的秘诀取决于如何将其与我们所了解的事物相联系"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)

    # 1.实例化一个转换器类
    transfer = CountVectorizer()

    # 2.调用fit_transform()
    data_final = transfer.fit_transform(data_new)
    print("data_new: \n", data_final.toarray())
    print("特征名字: \n", transfer.get_feature_names())

    return None


def tfidf_demo():
    """
    用TF-IDF的方法进行文本特征抽取
    :return:
    """
    # 将中文文本进行分词
    data = ["一种还是一种今天很残酷,明天更残酷,后天更美好,但绝对大部分是死在明天晚上,所以每个人都不要放弃今天",
            "我们看到的从很远星系来的光是在百万年之前发出的,这样当我们看到宇宙时,我们是在看它的过去",
            "如果只用一种方式了解某样事物,你就不会真正了解它,了解事物真正含义的秘诀取决于如何将其与我们所了解的事物相联系"]
    data_new = []
    for sent in data:
        data_new.append(cut_word(sent))
    # print(data_new)

    # 1.实例化一个转换器类
    transfer = TfidfVectorizer()

    # 2.调用fit_transform()
    data_final = transfer.fit_transform(data_new)
    print("data_new: \n", data_final.toarray())
    print("特征名字: \n", transfer.get_feature_names())

    return None


def minmax_demo():
    """
    归一化
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2.实例化一个转换器类
    transfer = MinMaxScaler()

    # 3.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


def stand_demo():
    """
    标准化
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("dating.txt")
    data = data.iloc[:, :3]
    print("data:\n", data)

    # 2.实例化一个转换器类
    transfer = StandardScaler()

    # 3.调用fit_transform()
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new)

    return None


def variance_demo():
    """
    过滤低方差特征
    :return:
    """
    # 1.获取数据
    data = pd.read_csv("train.csv")

    data = data.iloc[:, 2:5]
    # print("data\n", data)
    # 2.实例化一个转换器类
    transfer = VarianceThreshold(threshold=0.0001)

    # 3.调用fit_transform
    data_new = transfer.fit_transform(data)
    print("data_new:\n", data_new, data_new.shape)
    return None


if __name__ == "__main__":

    # 代码1:sklearn数据集使用
    # datasets_demo()
    # 代码2:字典特征抽取
    # dict_demo()
    # 代码3:文本特征抽取:CountVectorizer
    # count_demo()
    # 代码4:中文文本特征抽取:CountVectorizer
    # count_chinese_demo()
    # 代码5:中文文本特征抽取,自动分词
    # count_chinese_demo2()
    # 代码6:中文分词
    # print(cut_word("我爱北京天安门"))
    # 代码7:用TF-IDF的方法进行文本特征抽取
    # tfidf_demo()
    # 代码8:归一化
    # minmax_demo()
    # 代码9:标准化
    # stand_demo()
    # 代码10:低方差特征过滤
    variance_demo()
