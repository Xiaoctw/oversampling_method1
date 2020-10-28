import seaborn as sns
import matplotlib.pyplot as plt
import dbscan_based
from file4_对多种模型结果进行测试 import *

from datasets import *

file_name = 'haberman'
X, Y = load_data(file_name)

dic1 = {'transfusion': (0.15, 3),
        'adult': (1.6, 3),
        'breast-cancer-wisconsin': (0.5, 3),
        'haberman': (0.14, 3)}

eps_es = np.linspace(0.05, 5, 21)
# print(eps

min_pts = [2, 3, 5, 7]
precisions, recall, f1, auc_score = [], [], [], []
for eps in eps_es:
    model=KNeighborsClassifier(n_neighbors=3)
    list1 = compare_different_oversample_method(model, dbscan_based.DbscanBasedOversample(eps=eps, min_pts=3,
                                                                                          filter_majority=False), X, Y)
    # np.array([round(precision, 3), round(recall, 3), round(f1, 3),round(auc_score,3)])
    precisions.append(list1[0])
    recall.append(list1[1])
    f1.append(list1[2])
    auc_score.append(list1[3])

plt.plot(eps_es, f1, linestyle='--',  # 折线类型
         linewidth=2,  # 折线宽度
         color='c',  # 折线颜色
         marker='^',  # 点的形状
         markersize=10,  # 点的大小
         markeredgecolor='g',  # 点的边框色
         markerfacecolor='b', label='f1')  # 点的填充色
plt.plot(eps_es, auc_score, linestyle='-.',  # 折线类型
         linewidth=2,  # 折线宽度
         color='g',  # 折线颜色
         marker='^',  # 点的形状
         markersize=10,  # 点的大小
         markeredgecolor='m',  # 点的边框色
         markerfacecolor='b', label='auc_score')  # 点的填充色
plt.yticks(np.linspace(0,0.8,9))
#plt.axis('off')
plt.legend(loc='upper right')
plt.show()
print(auc_score)
print(f1)
