from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.under_sampling import EditedNearestNeighbours

from datasets import *

datasets = ['adult', 'automobile',  'breast-cancer-wisconsin', 'ecoli',
            'glass', 'haberman', 'transfusion', 'transfusion', 'wine', 'yeast']
model_names = ['KNN', 'tree']

if __name__ == '__main__':

    for dataset in datasets:
        for model in model_names:
            file_name = model + dataset + '.csv'
            path0 = Path(__file__).parent / ('result0') / file_name
            df0 = pd.read_csv(path0)
            cols = df0.columns
            data = df0[cols[1:]].values
            for i in range(1, 5):
                path = Path(__file__).parent / ('result' + '{}'.format(i)) / file_name
                df = pd.read_csv(path)
                data += df[cols[1:]].values
            data = np.round((data / 5),3)
            matrix=np.concatenate([df0[cols[0]].values.reshape(-1,1),data],axis=1)
            df=pd.DataFrame(matrix,columns=cols)
            df.to_csv(Path(__file__).parent / ('result') / file_name,index=False)

    # scores = np.concatenate(lists).reshape(-1, 5)
    # scores = np.around(scores, 3)
    # methods = np.array(['None', 'SMOTE', 'boarderline-SMOTE', 'ADASYN', 'SMOTE_ENN', 'MC-CCR', 'DbscanOversample'])
    # columns = ['method', 'precision', 'recall', 'f1-score', 'mGM', 'CNA']
    # data = np.concatenate([methods.reshape(-1, 1), scores], axis=1)
    # df = pd.DataFrame(data, columns=columns)
    # df.to_csv(path, index=False)
