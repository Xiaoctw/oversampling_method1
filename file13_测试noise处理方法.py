import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from file6_测试多分类数据集 import *
import dbscan_based
from datasets import *


def plot_data(train_X, train_Y):
    train_X = PCA(n_components=2).fit_transform(train_X)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    plt.colorbar()
    plt.show()


dic1 = {
    'automobile': (1.8, 3),
    'ecoli': (0.12, 3),
    'glass': (0.15, 3),
    'wine': (0.32, 2),
    'yeast': (0.13, 3)
}

file_name = 'ecoli'
eps, min_pts = dic1[file_name]

if __name__ == '__main__':
    df=pd.read_csv('ecoli.csv')
    X1,Y1=load_data(file_name)
    matrix = df.values
    X, Y = matrix[:, :-1], matrix[:, -1]
    Y = LabelEncoder().fit_transform(Y)
    print(Counter(Y))
    print(Counter(Y1))

# KNN:0.805, 0.815, 0.801, 0.803, 0.745
#   0.811, 0.802, 0.794, 0.765, 0.727

# tree:0.797,0.786,0.781,0.773,0.726
#     0.796,0.775,0.762,0.761,0.719
