import numpy as np
import matplotlib.pyplot as plt

from algorithms_knn import *
from algorithm_dbscan import *
from algorithm_svm import *
import dbscan_based
from imblearn.over_sampling import SMOTE

mean = [0, 0]
cov = [[0.5, 0], [0.5, 0.5]]
cov3=[[1,0],[1,1]]
mean2 = [2, 2]
cov2 = [[0.2, 0], [0, 0.2]]
mean3 = [-2, -2]
num_maj = 600
num_min = 50
num_noise=3
#num_train = 800
X1 = np.random.multivariate_normal(mean, cov, num_maj)
X2 = np.random.multivariate_normal(mean2, cov2, num_min // 2)
X3 = np.random.multivariate_normal(mean3, cov2, num_min // 2)
X4=np.random.multivariate_normal(mean,cov,num_noise)
Y1 = np.tile([0], num_maj)
Y2 = np.tile([1], num_min // 2)
Y3 = np.tile([1], num_min // 2)
Y4=np.tile([1], num_noise)
X = np.concatenate([X1, X2, X3,X4])
Y = np.concatenate([Y1, Y2, Y3,Y4])

def find_params_dbscan(train_X, train_Y, c):
    #0.36 2
    classifier = DBSCAN(eps=0.6, min_samples=10)
    X = train_X[train_Y == c]
    C = classifier.fit_predict(X)
    print('簇的个数:{}'.format(max(C) + 1))
    print(Counter(C))
    plt.scatter(X[:, 0], X[:, 1], c=C)
    plt.show()

if __name__ == '__main__':
    plt.scatter(X[:,0],X[:,1],c=Y)
    plt.show()
    #find_params_dbscan(X,Y,c=1)
    X1,Y1=dbscan_based.DbscanBasedOversample(eps=0.8,min_pts=5).fit_sample(X,Y)
    X2,Y2=SMOTE(k_neighbors=10).fit_sample(X,Y)
    plt.rcParams['figure.figsize'] = (13.0, 4.0)
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 3, 1)
    ax1.scatter(X[:, 0], X[:, 1], c=Y)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(X1[:, 0], X1[:, 1], c=Y1)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(X2[:, 0], X2[:, 1], c=Y2)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()
