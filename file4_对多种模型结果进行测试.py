import numpy as np
from algorithms_knn import *
from algorithm_dbscan import *
from algorithm_svm import *
import dbscan_based
from sklearn.decomposition import PCA, LatentDirichletAllocation
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import DBSCAN
from algorithm_MCCCR import *
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from datasets import *


def show_result(train_X, train_Y, test_X, test_Y):
    model.fit(train_X, train_Y)
    pdt_Y = model.predict(test_X)
    pdt_prob = model.predict_proba(test_X)[:, 1]
    precision = precision_score(test_Y, pdt_Y)
    recall = recall_score(test_Y, pdt_Y)
    f1 = f1_score(test_Y, pdt_Y)
    auc_score = roc_auc_score(test_Y, pdt_prob)
    print('precision:{:.3f}'.format(precision))
    print('recall:{:.3f}'.format(recall))
    print('f1_score:{:.3f}'.format(f1))
    print('auc:{:.3f}'.format(auc_score))
    return np.array([precision, recall, f1, auc_score])


def plot_data(X, Y):
    if X.shape[1] != 2:
        X = PCA(n_components=2).fit_transform(X)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.show()


def cal_multi_class_matrics(y_label, y_predict, y_predict_prob):
    """
    :param y_label: 实际标签
    :param y_predict: 预测标签，one-hot编码形式
    :param y_predict_prob 每种类别的概率
    :return:
    """
    # 这里计算的是每个类各种评估指标的加权平均值
    conf_mat = confusion_matrix(y_label, y_predict)
    precision = precision_score(y_label, y_predict, )
    recall = recall_score(y_label, y_predict, )
    f1 = f1_score(y_label, y_predict,)
    auc_score = roc_auc_score(y_label, y_predict_prob)
    return np.array([round(precision, 3), round(recall, 3), round(f1, 3),round(auc_score,3)])


def compare_different_oversample_method(model,sample_method, X, Y):
    n_split=5
    skf = StratifiedKFold(n_splits=n_split, shuffle=True)
    res_list = np.zeros(4)
    for train_indices, test_indices in skf.split(X, Y):
       # print('正在进行第{}次交叉验证'.format(i))
        train_X, train_Y, test_X, test_Y = X[train_indices], Y[train_indices], X[test_indices], Y[test_indices]
        min_k_kearest=min(Counter(train_Y))-1
        if sample_method=='SMOTE_ENN':
            enn = EditedNearestNeighbours()
            train_X,train_Y=enn.fit_sample(train_X,train_Y)
            smo = SMOTE(k_neighbors=min(3,min_k_kearest))
            if min_k_kearest>0:
                train_X,train_Y=smo.fit_sample(train_X,train_Y)
        elif sample_method=='smote':
            smo = SMOTE(k_neighbors=min(3,min_k_kearest))
            if min_k_kearest> 0:
                train_X, train_Y = smo.fit_sample(train_X, train_Y)
        elif sample_method=='borderline_smote':
            smo = BorderlineSMOTE(kind='borderline-1', k_neighbors=min(3,min_k_kearest))
            if min_k_kearest > 0:
                train_X, train_Y = smo.fit_sample(train_X, train_Y)
        elif sample_method=='adasyn':
            ada = ADASYN(n_neighbors=min(2,min_k_kearest))
            if min_k_kearest> 0:
                train_X,train_Y=ada.fit_sample(train_X,train_Y)
        elif sample_method:
            train_X, train_Y = sample_method.fit_sample(train_X, train_Y)
        model.fit(train_X, train_Y)
        y_score = model.predict(test_X)
        y_score_prob = model.predict_proba(test_X)[:,1]
        # res_list1 += cal_multi_class_matrics(test_Y,y_sampled_score,y_sampled_score_prob)
        res_list += cal_multi_class_matrics(test_Y, y_score, y_score_prob)
    return res_list / n_split


def print_result(model_name,res_list):
    print(
        'sample method:{},precision:{:.3f},recall:{:.3f},f1:{:.3f},auc:{:.3f}'.format(model_name,
                                                                                                 *res_list
                                                                                                 ))

def save_result(lists,model_name,data_set):
    file_name=model_name+data_set+'.csv'
    path=Path(__file__).parent / 'result' / file_name
    scores=np.concatenate(lists).reshape(-1,4)
    scores=np.around(scores,3)
    methods = np.array(['None', 'SMOTE', 'boarderline-SMOTE', 'ADASYN', 'SMOTE_ENN', 'MC-CCR','DbscanOversample'])
    columns = ['method', 'precision', 'recall', 'f1-score', 'auc_score']
    data=np.concatenate([methods.reshape(-1,1),scores],axis=1)
    df=pd.DataFrame(data,columns=columns)
    df.to_csv(path,index=False)



dic1={'transfusion':(0.15,3),
      'adult':(1.6,3),
      'breast-cancer-wisconsin':(0.5,3),
      'haberman':(0.14,3)}

file_name='transfusion'

eps,min_pts=dic1[file_name]

if __name__ == '__main__':
    X, Y =load_data(file_name)
    #model=KNeighborsClassifier(n_neighbors=3)
    model=DecisionTreeClassifier(max_depth=5,min_samples_split=3)
    model_name='tree'
    res_list1=compare_different_oversample_method(model,None,X,Y)
    res_list2=compare_different_oversample_method(model,dbscan_based.DbscanBasedOversample(eps=eps,min_pts=min_pts,filter_majority=False),X,Y)
    res_list3=compare_different_oversample_method(model,CCR(),X,Y)
    res_list4 = compare_different_oversample_method(model, 'smote', X, Y)
    res_list5 = compare_different_oversample_method(model, 'borderline_smote', X, Y)
    res_list6 = compare_different_oversample_method(model, 'adasyn', X, Y)
    res_list7 = compare_different_oversample_method(model, 'SMOTE_ENN', X, Y)
    print_result('None', res_list1)
    print_result('dbscan_result', res_list2)
    print_result('MC-CCR', res_list3)
    print_result('SMOTE', res_list4)
    print_result('borderline_smote', res_list5)
    print_result('ADASYN', res_list6)
    print_result('SMOTE_ENN', res_list7)
 #   save_result([res_list1,res_list4,res_list5,res_list6,res_list7,res_list3,res_list2],model_name=model_name,data_set=file_name)




