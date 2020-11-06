import seaborn as sns
import matplotlib.pyplot as plt
import dbscan_based
from file6_测试多分类数据集 import *

from datasets import *

file_name = 'automobile'
X, Y = load_data(file_name)

dic1 = {
    'automobile': (1.8, 3),
    'ecoli': (0.12, 3),
    'glass': (0.15, 3),
    'wine': (0.32, 2),
    'yeast': (0.13, 3)
}

outline_radios = np.linspace(0, 1, 11)
f1_auto=[0.660,0.671,0.682,0.706,0.721,0.716,0.717,0.710,0.707,0.704,0.704]
mGM_aut=[0.657,0.669,0.680,0.703,0.725,0.724,0.721,0.717,0.713,0.709,0.708]
f1_ecol=[0.765,0.769,0.772,0.777,0.783,0.791,0.791,0.795,0.800,0.801,0.794]
mGM_eco=[0.759,0.766,0.770,0.774,0.785,0.784,0.792,0.794,0.795,0.803,0.800]
# print(eps
# model = KNeighborsClassifier(n_neighbors=3)
# compare_different_oversample_method(model, dbscan_based.DbscanBasedOversample(eps=0.15, min_pts=3,
#                                                                               multiple_k=100,
#                                                                                           fit_outline_radio=False,
#                                                                                             outline_radio=0.9,
#                                                                                           filter_majority=False), X, Y)
#f1, auc_score = [], []
# for radio in outline_radios:
#     model = KNeighborsClassifier(n_neighbors=3)
#     list1 = compare_different_oversample_method(model, dbscan_based.DbscanBasedOversample(eps=1.6, min_pts=3,
#                                                                                           multiple_k=50,
#                                                                                           fit_outline_radio=False,
#                                                                                           outline_radio=radio,
#                                                                                           filter_majority=False), X, Y)
#     f1.append(list1[2])
#     auc_score.append(list1[3])



plt.plot(outline_radios, mGM_aut, linestyle='--',  # 折线类型
         linewidth=2,  # 折线宽度
         color='c',  # 折线颜色
         marker='^',  # 点的形状
         markersize=10,  # 点的大小
         markeredgecolor='g',  # 点的边框色
         markerfacecolor='b', label='automobile')  # 点的填充色
plt.plot(outline_radios, mGM_eco, linestyle='-.',  # 折线类型
         linewidth=2,  # 折线宽度
         color='g',  # 折线颜色
         marker='*',  # 点的形状
         markersize=10,  # 点的大小
         markeredgecolor='m',  # 点的边框色
         markerfacecolor='b', label='ecoli')  # 点的填充色
plt.yticks(np.linspace(0.5, 1, 9))
plt.ylabel('mGM')
# plt.axis('off')
plt.legend(loc='upper right')
plt.show()
# print(auc_score)
# print(f1)
