from algorithms_knn import *
from algorithm_dbscan import *
from algorithm_svm import *
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[0.5, 0], [0.5, 1]]
mean2 = [2, 2]
cov2 = [[0.5, 0], [0, 0.6]]
mean3 = [-2, -2]
num_maj = 400
num_min = 50
#num_train = 800
X1 = np.random.multivariate_normal(mean, cov, num_maj)
X2 = np.random.multivariate_normal(mean2, cov2, num_min // 2)
X3 = np.random.multivariate_normal(mean3, cov2, num_min // 2)
Y1 = np.tile([0], num_maj)
Y2 = np.tile([1], num_min // 2)
Y3 = np.tile([1], num_min // 2)
X = np.concatenate([X1, X2, X3])
Y = np.concatenate([Y1, Y2, Y3])
# method=SVMBasedOversample(alpha=0.05,n_steps=50)
method = KNNNormalDistributionOverSample()
X1, Y1 = method.fit_sample(X, Y)
plt.rcParams['figure.figsize'] = (8.0, 10.0)
fig = plt.figure()
ax1 = plt.subplot(2, 1, 1)
ax1.scatter(X[:, 0], X[:, 1], c=Y)
ax2 = plt.subplot(2, 1, 2)
ax2.scatter(X1[:, 0], X1[:, 1], c=Y1)
print(Counter(Y1))
plt.show()
