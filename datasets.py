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
    '''334 178经过数据清洗
    dbscan的默认参数：eps=0.15, min_pts=3
    outline_radio=0.2
    Counter({0: 570, 1: 178})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data
    :return:
    '''
    file_name = 'transfusion.csv'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df = pd.read_csv(data_file,header=None)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    X=MinMaxScaler().fit_transform(X)
    Y=LabelEncoder().fit_transform(Y)
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'transfusion.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X,Y

datasets=['adult','automobile','bank','breast-cancer-wisconsin','ecoli',
          'glass','haberman','transfusion','transfusion','wine','yeast']

def load_data(dataset):
    """
    transfusion:
    dbscan的默认参数：eps=0.15, min_pts=3
    不限制multiple_k
    outline_radio=0.5
    Counter({0: 570, 1: 178})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/blood-transfusion/transfusion.data
    glass:
    0.15 3
    Counter({1: 76, 0: 69, 5: 29, 2: 17, 3: 13, 4: 9})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data
    breast-cancer-wisconsin:
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
    eps 0.5   min_pts 3
    outline_radio=0.7
    Counter({2: 458, 4: 241})
    filter=True
    wine:
    默认参数 0.36 2
    Counter({1: 71, 0: 58, 2: 48})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    adult:
    Counter({0: 1029, 1: 499})
    dbscan的默认参数：eps=1.6, min_samples=3
    outline_radio=0.7
    该数据中有大量的噪声点
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    haberman：
    该数据集 225 81 偏斜
    eps 0.14 min_pts 3
     outline_radio=0.6,
     noise_radio=0.5,
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
    bank：
     Counter({0: 39922, 1: 5289})
    http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
    automobile：
    默认参数 半径1.8  个数3
    outline_radio=0.4
    noise_radio=0.2
    Counter({1: 48, 2: 46, 3: 29, 0: 20, 4: 13})
    ecoli：
    半径0.14 3
    noise_rate=0.7
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data
    Counter({'cp': 143, 'im': 77, 'pp': 52, 'imU': 35, 'om': 20, 'omL': 5, 'imS': 2, 'imL': 2})
    yeast：
    eps=0.13,min_pts=3
    outline_radio=0.2,
    noise_radio=0.9
    不限制边界点生成数量
    数据集：Counter({0: 463, 7: 429, 6: 244, 5: 163, 4: 51, 3: 44, 2: 35, 9: 30, 8: 20, 1: 5})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data
    """
    assert dataset in datasets
    file_name=dataset+'.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df = pd.read_csv(data_file)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    return X,Y


def pre_glass_data():
    """
    默认参数
    0.15 3
    Counter({1: 76, 0: 69, 5: 29, 2: 17, 3: 13, 4: 9})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data
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
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
    eps 0.5   min_pts 3
    Counter({2: 458, 4: 241})
    filter=True
    :return:
    '''
    file_name = 'breast-cancer-wisconsin.csv'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df = pd.read_csv(data_file,header=None)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'breast-cancer-wisconsin.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X, Y

def pre_wine():
    '''
    默认参数 0.36 2
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
    print(Counter(Y))
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
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'wine.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X, Y


def pre_adult_data():
    '''
    dbscan的默认参数：eps=1.6, min_samples=3
    Counter({0: 1029, 1: 499})
    该数据中有大量的噪声点
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
    :return:
    '''
    file_name = 'adult.data'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
   # df = pd.read_csv(data_file, header=None,nrows=2000)
    df=pd.read_csv(data_file,header=None)
    print('读取数据完成')
    cols = df.columns
    df = pd.concat((pd.get_dummies(df[cols[:-1]]), df[cols[-1]]), axis=1)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
   # X,Y=EditedNearestNeighbours().fit_sample(X,Y)
    # indices = np.arange(len(X))
    # np.random.shuffle(indices)
    # train_X, test_X = X[indices[:len(X) // 10 * 8]], X[indices[len(X) // 10 * 8:]]
    # train_Y, test_Y = Y[indices[:len(X) // 10 * 8]], Y[indices[len(X) // 10 * 8:]]
    # data = np.zeros((X.shape[0], X.shape[1] + 1))
    # data[:, :-1], data[:, -1] = X, Y
    # df = pd.DataFrame(data, columns=None)
    # file_name = 'adult.csv'
    # data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
   # df.to_csv(data_file, index=False)
    return X, Y


def pre_haberman():
    '''
    该数据集 225 81 偏斜
    eps 0.14 min_pts 3
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
    :return:
    '''
    file_name = 'haberman.data'
    data_file = Path(__file__).parent / '二分类数据集' / file_name
    df=pd.read_csv(data_file,header=None)
    matrix=df.values
    X,Y=matrix[:,:-1],matrix[:,-1]
    X=MinMaxScaler().fit_transform(X)
    Y=LabelEncoder().fit_transform(Y)
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'haberman.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X,Y


def pre_bank_data():
    """
    Counter({0: 39922, 1: 5289})
    http://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip
    :return:
    """
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
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'bank.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X, Y


def pre_automobile_data():
    '''
    默认参数 半径1.8  个数3
    Counter({1: 48, 2: 46, 3: 29, 0: 20, 4: 13})
    :return:
    '''
    file_name = Path(__file__).parent / '多分类数据集' / 'automobile.csv'
    data = pd.read_csv(file_name).values
    X = data[:, :-1]
    Y = data[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'automobile.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X, Y


#重点关注
def pre_ecoli():
    # 处理多分类数据集
    """
    默认参数：
    半径0.12 3
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data
    Counter({'cp': 143, 'im': 77, 'pp': 52, 'imU': 35, 'om': 20, 'omL': 5, 'imS': 2, 'imL': 2})
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
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'ecoli.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X, Y

def pre_yeast():
    '''
    eps=0.13,min_pts=3
    outline_radio=0.2,
    noise_radio=0.9
    数据集：Counter({0: 463, 7: 429, 6: 244, 5: 163, 4: 51, 3: 44, 2: 35, 9: 30, 8: 20, 1: 5})
    链接：http://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data
    :return:
    '''
    # file_name = 'yeast.data'
    # data_file = Path(__file__).parent / '多分类数据集' / file_name
    # data=pd.read_csv(data_file,sep='\\s+',header=None)
    # cols=data.columns
    # X,Y=data[cols[1:-1]].values,data[cols[-1]].values
    # X = MinMaxScaler().fit_transform(X)
    # Y = LabelEncoder().fit_transform(Y)
    # X = X[Y != 8]
    # Y = Y[Y != 8]
    # X = X[Y != 1]
    # Y = Y[Y != 1]
    # data=np.zeros((X.shape[0],X.shape[1]+1))
    # data[:,:-1],data[:,-1]=X,Y
    # df=pd.DataFrame(data,columns=cols[1:])
    # file_name='yeast.csv'
    # data_file = Path(__file__).parent / '多分类数据集' / file_name
    # df.to_csv(data_file,index=False)
    # print(Counter(Y))
    file_name = 'yeast.csv'
    data_file = Path(__file__).parent / '多分类数据集' / file_name
    data = pd.read_csv(data_file)
    cols = data.columns
    X,Y=data[cols[:-1]].values,data[cols[-1]].values
    X = MinMaxScaler().fit_transform(X)
    Y = LabelEncoder().fit_transform(Y)
    data = np.zeros((X.shape[0], X.shape[1] + 1))
    data[:, :-1], data[:, -1] = X, Y
    df = pd.DataFrame(data, columns=None)
    file_name = 'yeast.csv'
    data_file = Path(__file__).parent / '预处理完毕的数据集' / file_name
    df.to_csv(data_file, index=False)
    return X, Y


if __name__ == '__main__':
    # X,Y=load_data('automobile')
    X,Y=pre_adult_data()
    print(X.shape)
    print(Counter(Y))
    # print(X.shape)
    # print(Y.shape)



