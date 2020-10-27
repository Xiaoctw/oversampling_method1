import numpy as np
import matplotlib.pyplot as plt
from algorithms_knn import *
from algorithm_svm import *
from algorithm_MCCCR import *
import dbscan_based
import datasets
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import EditedNearestNeighbours


def make_data(num1, num2, num3, num4):
    mean1 = [3.8, 3.8]
    mean2 = [2.8, 3.2]
    mean3 = [2.4, 2.2]
    mean4 = [1.8, 2.1]
    mean5 = [3, 5]
    mean6 = [1, 3]
    mean7 = [3.4, 2]
    cov = [[0.1, 0], [0, 0.1]]
    X1 = np.random.multivariate_normal(mean1, cov, num1 // 4)
    X2 = np.random.multivariate_normal(mean2, cov, num1 // 4)
    X3 = np.random.multivariate_normal(mean3, cov, num1 // 4)
    X4 = np.random.multivariate_normal(mean4, cov, num1 // 4)
    X5 = np.random.multivariate_normal(mean5, cov, num2)
    X6 = np.random.multivariate_normal(mean6, cov, num3 // 2)
    X7 = np.random.multivariate_normal(mean7, cov, num3 // 2)
    X8 = []
    for _ in range(num4):
        X8.append([random.random() * 2 + 2.5, random.random() * 2 + 2.5])
    X8 = np.concatenate(X8).reshape(-1, 2)
    X9=np.random.multivariate_normal([2.2,2.5],cov,2)
    X = np.concatenate([X1, X2, X3, X4, X5, X6, X7, X8,X9])
    Y1 = np.tile([1], num1)
    Y2 = np.tile([2], num2)
    Y3 = np.tile([3], num3)
    Y4 = np.tile([4], num4)
    Y5 = np.tile([3], 2)
    Y = np.concatenate([Y1, Y2, Y3, Y4,Y5])
    return X, Y


def plot_data(X, Y):
    # train_X = PCA(n_components=2).fit_transform(train_X)
    plt.rcParams['figure.figsize'] = (27.0, 5.0)
    fig = plt.figure()
    ax0 = fig.add_subplot(1, 5, 1)
    ax0.scatter(X[:, 0], X[:, 1], c=Y)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    X1, Y1 = SMOTE().fit_sample(X, Y)
    ax1 = fig.add_subplot(1, 5, 2)
    ax1.scatter(X1[:, 0], X1[:, 1], c=Y1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    X2, Y2 = BorderlineSMOTE(kind='borderline-1').fit_sample(X, Y)
    ax2 = fig.add_subplot(1, 5, 3)
    ax2.scatter(X2[:, 0], X2[:, 1], c=Y2)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    enn = EditedNearestNeighbours()
    X3, Y3 = enn.fit_sample(X, Y)
    smo = SMOTE(k_neighbors=5)
    X3, Y3 = smo.fit_sample(X3, Y3)
    ax3 = fig.add_subplot(1, 5, 4)
    ax3.scatter(X3[:, 0], X3[:, 1], c=Y3)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    X4, Y4 = ADASYN(n_neighbors=3).fit_sample(X, Y)
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.scatter(X4[:, 0], X4[:, 1], c=Y4)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    X5, Y5 = dbscan_based.MultiDbscanBasedOverSample(eps=0.3, min_pts=5).fit_sample(X, Y)
    ax5 = fig.add_subplot(1, 5, 5)
    ax5.scatter(X5[:, 0], X5[:, 1], c=Y5)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def find_params_dbscan(train_X, train_Y, c):
    # 0.36 2
    classifier = DBSCAN(eps=0.3, min_samples=5)
    X = train_X[train_Y == c]
    C = classifier.fit_predict(X)
    print('簇的个数:{}'.format(max(C) + 1))
    print(Counter(C))
    plt.scatter(X[:, 0], X[:, 1], c=C)
    plt.show()


if __name__ == '__main__':
    X, Y = make_data(200, 40, 40, 8)
    plot_data(X,Y)