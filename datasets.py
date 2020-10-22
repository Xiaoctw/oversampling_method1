from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

file_name = 'transfusion.data'
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.under_sampling import EditedNearestNeighbours

def pre_transfusion_data():
    '''
    dbscan的默认参数：eps=0.15, min_pts=3
    334 178经过数据清洗
    outline_radio=0.2
    :return:
    '''
    file_name = 'transfusion.csv'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df = pd.read_csv(data_file,header=None)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y=LabelEncoder().fit_transform(Y)
    return X,Y

def pre_glass_data():
    """
    默认参数
    0.15 3
    :return:
    """
    file_name = 'glass.data'
    data_file = Path(__file__).parent / '多分类数据集' / file_name
    df = pd.read_csv(data_file)
    matrix = df.values
    X, Y = matrix[:, 1:-1], matrix[:, -1]
    X = MinMaxScaler().fit_transform(X)
    Y = LabelEncoder().fit_transform(Y)
    return X, Y

def pre_breast_cancer():
    '''
    eps 0.5   min_pts 3
    filter=True
    :return:
    '''
    file_name = 'breast-cancer-wisconsin.csv'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df = pd.read_csv(data_file,header=None)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    return X, Y

def pre_wine():
    '''
    默认参数 0.36 2
    :return:
    '''
    # file_name='wine.csv'
    # data_file = Path(__file__).parent / '多分类数据集' / file_name
    # df = pd.read_csv(data_file)
    # cols = df.columns
    # matrix=df.values
    # X,Y=matrix[:, :-1], matrix[:, -1]
    # Y = LabelEncoder().fit_transform(Y)
    # return X, Y
    file_name='wine.data'
    data_file = Path(__file__).parent / '多分类数据集' / file_name
    df = pd.read_csv(data_file)
    cols = df.columns
    matrix=df.values
    X,Y=matrix[:, 1:], matrix[:, 0]
    X=MinMaxScaler().fit_transform(X)
    Y = LabelEncoder().fit_transform(Y)
    return X, Y


def pre_adult_data():
    '''
    dbscan的默认参数：eps=1.6, min_samples=3
    该数据中有大量的噪声点
    :return:
    '''
    file_name = 'adult.data'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df = pd.read_csv(data_file, header=None,nrows=2000)
    cols = df.columns
    df = pd.concat((pd.get_dummies(df[cols[:-1]]), df[cols[-1]]), axis=1)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    X,Y=EditedNearestNeighbours().fit_sample(X,Y)
    # indices = np.arange(len(X))
    # np.random.shuffle(indices)
    # train_X, test_X = X[indices[:len(X) // 10 * 8]], X[indices[len(X) // 10 * 8:]]
    # train_Y, test_Y = Y[indices[:len(X) // 10 * 8]], Y[indices[len(X) // 10 * 8:]]
    return X, Y


def pre_haberman():
    '''
    该数据集 225 81 偏斜
    eps 0.14 min_pts 3
    :return:
    '''
    file_name = 'haberman.data'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df=pd.read_csv(data_file,header=None)
    matrix=df.values
    X,Y=matrix[:,:-1],matrix[:,-1]
    X=MinMaxScaler().fit_transform(X)
    Y=LabelEncoder().fit_transform(Y)
    return X,Y


def pre_bank_data():
    file_name = 'bank.csv'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df = pd.read_csv(data_file, ';')
    cols = df.columns
    # df = pd.concat((pd.get_dummies(df[cols[:-1]]), df[cols[-1]]), axis=1)
    # matrix = df.values
    # X, Y = matrix[:, :-1], matrix[:, -1]
    X = df[cols[:-1]]
    X = pd.get_dummies(X).values
    Y = df[cols[-1]].values
    Y = LabelEncoder().fit_transform(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    # print(Counter(Y))
    return X, Y


def pre_automobile_data():
    '''
    默认参数 半径1.8  个数3
    :return:
    '''
    file_name = Path(__file__).parent / '多分类数据集' / 'automobile.csv'
    data = pd.read_csv(file_name).values
    X = data[:, :-1]
    Y = data[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    return X, Y


def pre_ecoli():
    # 处理多分类数据集
    """
    默认参数：
    半径0.12 3
    :return:
    """
    file_name = 'ecoli.csv'
    data_file = Path(__file__).parent / '多分类数据集' / file_name
    df = pd.read_csv(data_file)
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    X=MinMaxScaler().fit_transform(X)
    Y = LabelEncoder().fit_transform(Y)
    return X, Y


if __name__ == '__main__':
    # train_X, train_Y = pre_ecoli()
    # print(Counter(train_Y))
    X, Y = pre_transfusion_data()
    print(Counter(Y))


