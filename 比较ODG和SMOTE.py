import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import BorderlineSMOTE,SMOTE
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

X1, Y1 = SMOTE(k_neighbors=40).fit_sample(X, Y)
X2,Y2=CCR(energy=3).fit_sample(X,Y)
X3, Y3 = dbscan_based.DbscanBasedOversample(eps=1, min_pts=3, outline_radio=0.9).fit_sample(X, Y)

plt.rcParams['figure.figsize'] = (5.0, 5.0)
plt.scatter(X[:, 0], X[:, 1], c=find_colors(Y))
plt.axis('off',)
plt.xticks([])
plt.yticks([])
plt.savefig('myplot20_0.png')
#ax0.set_title('Original dataset')
plt.show()
plt.scatter(X1[:, 0], X1[:, 1], c=find_colors(Y1))
#ax1.set_title('CCR')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('myplot20_1.png')
plt.show()
plt.scatter(X2[:, 0], X2[:, 1], c=find_colors(Y2))
#ax2.set_title('ODG')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('myplot20_2.png')
plt.show()
plt.scatter(X3[:, 0], X3[:, 1], c=find_colors(Y3))
#ax2.set_title('ODG')
plt.axis('off')
plt.xticks([])
plt.yticks([])
plt.savefig('myplot20_3.png')
plt.show()
