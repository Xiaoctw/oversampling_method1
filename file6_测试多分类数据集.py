import numpy as np
from datasets import *
from algorithm_dbscan import *
from find_params import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from algorithm_MCCCR import *
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
import warnings

warnings.filterwarnings('ignore')
'''
时刻记录一下实验的结果
MC-CCR和dbscan_sample和没经过处理的数据集进行比较
在决策树和K近邻上，dbscan_sample都处于优势，而在LR上MC-CCR处于优势

'''


def compare_different_multi_oversample_method(model, sample_method, X, Y):
    n_split = 5
    skf = StratifiedKFold(n_splits=n_split, shuffle=True)
    res_list = np.zeros(5)
    for train_indices, test_indices in skf.split(X, Y):
        # print('正在进行第{}次交叉验证'.format(i))
        train_X, train_Y, test_X, test_Y = X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]
        min_k_kearest = min(Counter(train_Y)) - 1
        if sample_method == 'SMOTE_ENN':
            enn = EditedNearestNeighbours()
            train_X, train_Y = enn.fit_sample(train_X, train_Y)
            smo = SMOTE(k_neighbors=min(3, min_k_kearest))
            if min_k_kearest > 0:
                train_X, train_Y = smo.fit_sample(train_X, train_Y)
        elif sample_method == 'smote':
            smo = SMOTE(k_neighbors=min(3, min_k_kearest))
            if min_k_kearest > 0:
                train_X, train_Y = smo.fit_sample(train_X, train_Y)
        elif sample_method == 'borderline_smote':
            smo = BorderlineSMOTE(kind='borderline-1', k_neighbors=min(3, min_k_kearest))
            if min_k_kearest > 0:
                train_X, train_Y = smo.fit_sample(train_X, train_Y)
        elif sample_method == 'adasyn':
            ada = ADASYN(n_neighbors=min(2, min_k_kearest))
            if min_k_kearest > 0:
                train_X, train_Y = ada.fit_sample(train_X, train_Y)
        elif sample_method:
            train_X, train_Y = sample_method.fit_sample(train_X, train_Y)
        model.fit(train_X, train_Y)
        y_score = model.predict(test_X)
        y_score_prob = model.predict_proba(test_X)
        # res_list1 += cal_multi_class_matrics(test_Y,y_sampled_score,y_sampled_score_prob)
        res_list += cal_multi_class_matrics(test_Y, y_score)
    return res_list / n_split


def save_result(lists, model_name, data_set,i):
    file_name = model_name + data_set + '.csv'
    path = Path(__file__).parent / ('result'+'{}'.format(i)) / file_name
    scores = np.concatenate(lists).reshape(-1, 5)
    scores = np.around(scores, 3)
    methods = np.array(['None', 'SMOTE', 'boarderline-SMOTE', 'ADASYN', 'SMOTE_ENN', 'MC-CCR', 'DbscanOversample'])
    columns = ['method', 'precision', 'recall', 'f1-score', 'mGM', 'CNA']
    data = np.concatenate([methods.reshape(-1, 1), scores], axis=1)
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(path, index=False)


def print_result(model_name, res_list):
    print(
        'sample method:{},precision:{:.3f},recall:{:.3f},f1:{:.3f},mGM:{:.3f},CNA:{:.3f}'.format(model_name,
                                                                                                 *res_list
                                                                                                 ))


dic1 = {
    'automobile': (1.8, 3),
    'ecoli': (0.12, 3),
    'glass': (0.15, 3),
    'wine': (0.32, 2),
    'yeast': (0.13, 3)
}

file_name = 'yeast'
eps, min_pts = dic1[file_name]
if __name__ == '__main__':
    X, Y = load_data(file_name)
    #  plot_data(X, Y)
    # X, Y = method.fit_sample(X, Y)
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import  LogisticRegression
    model = LogisticRegression()
    i=4
    #model=DecisionTreeClassifier(max_depth=5,min_samples_split=3)
    res_list1 = compare_different_multi_oversample_method(model, None, X, Y)
    res_list2 = compare_different_multi_oversample_method(model, 'smote', X, Y)
    res_list3 = compare_different_multi_oversample_method(model, 'borderline_smote', X, Y)
    res_list4 = compare_different_multi_oversample_method(model, 'adasyn', X, Y)
    res_list5 = compare_different_multi_oversample_method(model, 'SMOTE_ENN', X, Y)
    res_list6 = compare_different_multi_oversample_method(model, MultiClassCCR(), X, Y)
    res_list7 = compare_different_multi_oversample_method(model, dbscan_based.MultiDbscanBasedOverSample(eps=eps,
                                                                                                         k=3,
                                                                                                         min_pts=min_pts,
                                                                                                        outline_radio=0.7
                                                                                                        ),
                                                          X, Y)
    print_result('None', res_list1)
    print_result('SMOTE', res_list2)
    print_result('borderline_smote', res_list3)
    print_result('ADASYN', res_list4)
    print_result('SMOTE_ENN', res_list5)
    print_result('MC-CCR', res_list6)
    print_result('dbscan_result', res_list7)
    print(Counter(Y))
    # plot_data(X, Y)
    save_result([res_list1, res_list2, res_list3, res_list4, res_list5, res_list6, res_list7], model_name='LR',
                data_set=file_name,i=i)
