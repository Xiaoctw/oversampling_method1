import seaborn as sns
import matplotlib.pyplot as plt
import dbscan_based
from file4_对多种模型结果进行测试 import *

from datasets import *

file_name = 'adult'
X, Y = load_data(file_name)

dic1 = {'transfusion': (0.15, 3),
        'adult': (1.6, 3),
        'breast-cancer-wisconsin': (0.5, 3),
        'haberman': (0.14, 3)}

outline_radios = np.linspace(0, 1, 11)
# print(eps
# model = KNeighborsClassifier(n_neighbors=3)
# compare_different_oversample_method(model, dbscan_based.DbscanBasedOversample(eps=0.15, min_pts=3,
#                                                                               multiple_k=100,
#                                                                                           fit_outline_radio=False,
#                                                                                             outline_radio=0.9,
#                                                                                           filter_majority=False), X, Y)
f1, auc_score = [], []
# for radio in outline_radios:
#     model = KNeighborsClassifier(n_neighbors=3)
#     list1 = compare_different_oversample_method(model, dbscan_based.DbscanBasedOversample(eps=1.6, min_pts=3,
#                                                                                           multiple_k=50,
#                                                                                           fit_outline_radio=False,
#                                                                                           outline_radio=radio,
#                                                                                           filter_majority=False), X, Y)
#     f1.append(list1[2])
#     auc_score.append(list1[3])

f1_breast = [0.830, 0.847, 0.849, 0.857, 0.872, 0.884, 0.913, 0.938, 0.935, 0.930, 0.926]
auc_score_breast = [0.860, 0.872, 0.876, 0.880, 0.899, 0.907, 0.936, 0.967, 0.968, 0.954, 0.946]
f1_transfusion = [0.620, 0.630, 0.668, 0.688, 0.715, 0.712, 0.704, 0.701, 0.693, 0.682, 0.681]
auc_score_transfusion = [0.780, 0.794, 0.824, 0.832, 0.841, 0.846, 0.843, 0.841, 0.838, 0.831, 0.829]

plt.plot(outline_radios, auc_score_breast, linestyle='--',  # 折线类型
         linewidth=2,  # 折线宽度
         color='c',  # 折线颜色
         marker='^',  # 点的形状
         markersize=10,  # 点的大小
         markeredgecolor='g',  # 点的边框色
         markerfacecolor='b', label='breast-cancer-wisconsin')  # 点的填充色
plt.plot(outline_radios, auc_score_transfusion, linestyle='-.',  # 折线类型
         linewidth=2,  # 折线宽度
         color='g',  # 折线颜色
         marker='*',  # 点的形状
         markersize=10,  # 点的大小
         markeredgecolor='m',  # 点的边框色
         markerfacecolor='b', label='transfusion')  # 点的填充色
plt.yticks(np.linspace(0.5, 1, 9))
plt.ylabel('auc_score')
# plt.axis('off')
plt.legend(loc='upper right')
plt.show()
# print(auc_score)
# print(f1)
