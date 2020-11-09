import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE
from algorithm_MCCCR import *
from find_params import *
import dbscan_based
import warnings

warnings.filterwarnings('ignore')
mean = [0, 0]
cov = [[0.8, 0], [0.7, 1.4]]
mean2 = [3, 3.5]
mean3 = [-3.2, -2.4]
cov2 = [[0.5, 0], [0, 0.6]]
num_maj = 100
num_min = 20
X1 = np.random.multivariate_normal(mean, cov, num_maj)
X2 = np.random.multivariate_normal(mean2, cov2, num_min)
X3 = np.random.multivariate_normal(mean3, cov2, num_min)
Y1 = np.tile([0], num_maj)
Y2 = np.tile([1], num_min)
Y3 = np.tile([1], num_min)
X4 = np.array([[0, 1], [0.3, 2]])
Y4 = np.array([1, 1, ])
X = np.concatenate([X1, X2, X3, X4])
Y = np.concatenate([Y1, Y2, Y3, Y4])


def find_colors(train_Y):
    cs = []
    for val in train_Y:
        if val == 0:
            cs.append('c')
        else:
            cs.append('r')
    return cs


# find_params_dbscan(X,Y,1)

X1, Y1 = CCR(energy=4).fit_sample(X, Y)
X2, Y2 = dbscan_based.DbscanBasedOversample(eps=1, min_pts=3, outline_radio=0.9).fit_sample(X, Y)

plt.rcParams['figure.figsize'] = (15.0, 5.0)
fig = plt.figure()
ax0 = fig.add_subplot(1, 3, 1)
ax0.scatter(X[:, 0], X[:, 1], c=find_colors(Y))
ax0.set_title('Original dataset')
plt.axis('off',)
plt.xticks([])
plt.yticks([])
ax1 = fig.add_subplot(1, 3, 2)
ax1.scatter(X1[:, 0], X1[:, 1], c=find_colors(Y1))
ax1.set_title('CCR')
plt.axis('off')
plt.xticks([])
plt.yticks([])
ax2 = fig.add_subplot(1, 3, 3)
ax2.scatter(X2[:, 0], X2[:, 1], c=find_colors(Y2))
ax2.set_title('ODG')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.show()
