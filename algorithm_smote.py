import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.under_sampling import TomekLinks
from datasets import *

def smote():
    smo=SMOTE(k_neighbors=1)
    return smo

def borderline_smote():
    smo=BorderlineSMOTE(kind='borderline-1',k_neighbors=1)
    return smo

def adasyn():
    ada=ADASYN(n_neighbors=2)
    return ada

def smote_enn():
    # enn=EditedNearestNeighbours()
    # X,Y=enn.fit_sample(X,Y)
    # smo = SMOTE(k_neighbors=2)
    #return smo.fit_sample(X,Y)
    return None




if __name__ == '__main__':
    print(Counter(Y))


