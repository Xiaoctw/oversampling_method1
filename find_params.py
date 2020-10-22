import sys
import random
from functools import reduce

import numpy as np
from collections import Counter
from sklearn import svm
from datasets import *
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import algorithm_vae
from sklearn.preprocessing import label_binarize
from algorithm_MCCCR import *
import dbscan_based
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score


def find_params_dbscan(train_X, train_Y, c):
    #0.36 2
    classifier = DBSCAN(eps=0.12, min_samples=3)
    X = train_X[train_Y == c]
    C = classifier.fit_predict(X)
    print('簇的个数:{}'.format(max(C) + 1))
    print(Counter(C))
    X = PCA(n_components=2).fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=C)
    plt.show()


def find_dists():
    return


def plot_data(train_X, train_Y):
    train_X = PCA(n_components=2).fit_transform(train_X)
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y)
    plt.colorbar()
    plt.show()


def cal_multi_class_matrics(y_label, y_predict):
    """
    :param y_label: 实际标签
    :param y_predict: 预测标签，one-hot编码形式
    :return:
    """
    # 这里计算的是每个类各种评估指标的加权平均值
    conf_mat = confusion_matrix(y_label, y_predict)
    # print(conf_mat)
    y_one_hot = label_binarize(y_label, np.arange(np.unique(y_label).shape[0]))
    precision = precision_score(y_label, y_predict, average='macro')
    recall = recall_score(y_label, y_predict, average='macro')
    f1 = f1_score(y_label, y_predict, average='macro')
    mGM_val=mGM(conf_mat)
    cna=CNA(conf_mat)
    # print('auc_score:{}'.format(auc))
    return np.array([round(precision, 3), round(recall, 3), round(f1, 3),round(mGM_val,3),round(cna,3)])



def mGM(conf_matrix):
    m = len(conf_matrix)
    val = reduce((lambda x, y: x * y), [conf_matrix[i][i] / sum(conf_matrix[i]) for i in range(m) if sum(conf_matrix[i])!=0])
    return val ** (1 / m)

def CNA(conf_matrix):
    m=len(conf_matrix)
    val=reduce((lambda x,y:x+y),[conf_matrix[i][i]/max(sum(conf_matrix[i]),sum(conf_matrix[:,i])) for i in range(m) if max(sum(conf_matrix[i]),sum(conf_matrix[:,i]))!=0])
    return val/m


# def plot_tran_Data(X1,Y1,X2,Y2,X3,Y3):
#     plt.rcParams['figure.figsize'] = (10.0, 3.0)
#     fig=plt.figure()
#     ax1=fig.add_subplot(1,3,1)
#     X1 = PCA(n_components=2).fit_transform(X1)
#     ax1.scatter(X1[:, 0], X1[:, 1], c=Y1)
#    # ax1.colorbar()
#     ax2 = fig.add_subplot(1, 3, 3)
#     X2 = PCA(n_components=2).fit_transform(X2)
#     ax2.scatter(X2[:, 0], X2[:, 1], c=Y2)
#   #  ax2.colorbar()
#     ax3 = fig.add_subplot(1, 3, 2)
#     X3 = PCA(n_components=2).fit_transform(X3)
#     ax3.scatter(X3[:, 0], X3[:, 1], c=Y3)
#    # ax3.colorbar()
#     plt.show()


if __name__ == '__main__':
    # train_X, test_X, train_Y, test_Y = pre_adult_data()
    # find_params_dbscan(train_X,train_Y)
    train_X,train_Y=pre_transfusion_data()
    print(Counter(train_Y))
    #plot_data(train_X,train_Y)
    print(train_X.shape)
    #find_params_dbscan(train_X,train_Y,1)
    X1,Y1=dbscan_based.DbscanBasedOversample(eps=0.15,min_pts=3,multiple_k=0.8).fit_sample(train_X,train_Y)
    X2, Y2 = dbscan_based.DbscanBasedOversample(eps=0.15, min_pts=3,).fit_sample(train_X, train_Y)
    #plot_tran_Data(train_X,train_Y,X1,Y1,X2,Y2)
    #train_X,train_Y=CCR().fit_sample(train_X,train_Y)
    # print(Counter(train_Y))
   # plot_data(X1,Y1)
    # plot_data(train_X,train_Y)


