import sys
import random
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.cluster import DBSCAN


def distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


class KNNNormalDistributionOverSample:
    """
    该方法综合了KNN和高斯分布
    针对少数类样本点，首先通过KNN找到最近邻的K个点，
    如果全部为少数类样本点，那么说明这个点位于少数类样本点的内部，不用管它
    如果全部为多数类样本点，说明改点可能为噪声点，先将其加入到删除类表中，
    暂时还没有对其进行处理。
    部分多数类，部分少数类的样本，对根据k近邻中少数类样本点的分布生成新的点，
    k近邻中多数类样本点越多，在该点上生成的数据也就越多
    之后需要对多数类样本点进行平移，平移出k近邻中最大少数类样本点的距离
    """

    def __init__(self, p=2, k=7, alpha=0.05):
        self.p = p
        self.k = k
        self.alpha = alpha

    def fit_sample(self, X, Y, k=-1, min_class=None):
        if k == -1:
            k = self.k
        classes = np.unique(Y)
        sizes = [sum(Y == c) for c in classes]
        if min_class is None:
            min_class = classes[np.argmin(sizes)]
        num_sample, num_feat = X.shape[0], X.shape[1]
        minority_idxes = np.array(range(num_sample))[Y == min_class]
        delete_list = []
        # 保留除少数类样本点以外样本点的位移
        translations = np.zeros(X.shape)
        new_data = []
        # num_new_data = num_sample - 2 * minority_idxes.shape[0]
        k_nearest_minority = {}
        num_over_sample = {}
        total_num_oversample = 0
        for i in minority_idxes:
            dist_arr = (np.sum(np.abs(X[i] - X) ** self.p, 1)) ** (1 / self.p)
            k_nearest_idxs = np.argsort(dist_arr)[:k + 1]
            k_nearest_labels = Y[k_nearest_idxs]
            k_nearest_counts = Counter(k_nearest_labels)
            if k_nearest_counts[min_class] <= 1 or Y[k_nearest_idxs[0]]!=min_class:
                # 噪声点，需要去除
                delete_list.append(i)
            elif k_nearest_counts[min_class] >= k:
                # 大多数为少数类样本点，不需要进行额外的操作
                num_over_sample[i] = self.alpha
                total_num_oversample += num_over_sample[i]
                tem_minority_args = np.array([arg for arg in k_nearest_idxs if Y[arg] == min_class])
                k_nearest_minority[i] = tem_minority_args
                continue
            else:
                tem_minority_args = np.array([arg for arg in k_nearest_idxs if Y[arg] == min_class])
                tem_other_args = np.array([arg for arg in k_nearest_idxs if Y[arg] != min_class])
                k_nearest_minority[i] = tem_minority_args
                num_over_sample[i] = len(tem_other_args)
                total_num_oversample += num_over_sample[i]
                remove_majority_distance = dist_arr[k_nearest_idxs[-1]]
                #   tem_translations = np.zeros((len(tem_other_args), X.shape[1]))
                # if len(tem_other_args)>0:
                # tem_translations[trans_idxs] = (X[trans_idxs]-X[i])*((remove_majority_distance-dist_arr[
                # trans_idxs])/dist_arr[trans_idxs]).reshape(-1,1)
                if len(tem_other_args) > 0:
                    tem_translations = (X[tem_other_args] - X[i]) * (
                            (remove_majority_distance - dist_arr[tem_other_args]) / (
                                dist_arr[tem_other_args] + 1e-6)).reshape(-1, 1)
                    translations[tem_other_args] += tem_translations
        # 开始生成新的数据点
        # deno = np.sum([-np.log(len(k_nearest_minority[i])) for i in k_nearest_minority])
        #translations = np.zeros(X.shape)
        X += translations
        #  print(translations.sum(axis=0))
        num_new_data = num_sample - 2 * minority_idxes.shape[0]
        for i in k_nearest_minority:
            tem_num_new_data = int(num_over_sample[i] / total_num_oversample * num_new_data)
            tem_data = X[k_nearest_minority[i]]
            mean = np.mean(tem_data, axis=0)
            cov = np.cov(tem_data.T)
            new_data.append(np.random.multivariate_normal(mean=mean, cov=cov, size=tem_num_new_data))
        if len(new_data) > 0:
            new_data = np.concatenate(new_data)
        new_label = np.array([min_class] * new_data.shape[0])
        X = np.concatenate([X, new_data])
        Y = np.concatenate([Y, new_label])
        # print('少数类样本点数量:{}'.format(minority_idxes.shape[0]))
        # print('噪声样本点数量:{}'.format(len(delete_list)))
        return X, Y


class MultiKNNOverSample:
    def __init__(self, p=2, k=7):
        self.p = p
        self.k = k

    def fit_sample(self, X, Y):
        classes = np.unique(Y)
        sizes = np.array([sum(Y == c) for c in classes])
        sorted_idxes = np.argsort(sizes)[::-1]
        classes = classes[sorted_idxes]
        observations = {c: X[Y == c] for c in classes}
        n_max = max(sizes)
        for i in range(1, len(classes)):
            tem_class = classes[i]
            n = n_max - len(observations[tem_class])
            used_observations = {}
            unused_observations = {}
            for j in range(i):
                all_indices = list(range(len(observations[classes[j]])))
                used_indices = np.random.choice(all_indices, int(n_max / i), replace=False)
                used_observations[classes[j]] = [
                    observations[classes[j]][idx] for idx in used_indices
                ]
                unused_observations[classes[j]] = [
                    observations[classes[j]][idx] for idx in all_indices if idx not in used_indices
                ]

            used_observations[tem_class] = observations[tem_class]
            unused_observations[tem_class] = []

            for j in range(i + 1, len(classes)):
                used_observations[classes[j]] = []
                unused_observations[classes[j]] = observations[classes[j]]

            unpacked_points, unpacked_labels = self.unpack_observations(used_observations)
            sam_method = KNNNormalDistributionOverSample(p=self.p, k=self.k)
            over_sampled_points, over_sampled_labels = sam_method.fit_sample(unpacked_points, unpacked_labels,
                                                                             min_class=tem_class)
            observations = {}

            for cls in classes:
                class_oversampled_points = over_sampled_points[over_sampled_labels == cls]
                class_unused_points = unused_observations[cls]
                if len(class_unused_points) == 0 and len(class_oversampled_points) == 0:
                    observations[cls] = np.array([])
                elif len(class_oversampled_points) == 0:
                    observations[cls] = class_unused_points
                elif len(class_unused_points) == 0:
                    observations[cls] = class_oversampled_points
                else:
                    observations[cls] = np.concatenate([class_oversampled_points, class_unused_points])

        unpacked_points, unpacked_labels = self.unpack_observations(observations)
        return unpacked_points, unpacked_labels

    def unpack_observations(self, observations):
        unpacked_points = []
        unpacked_labels = []
        for cls in observations:
            if len(observations[cls]) > 0:
                unpacked_points.append(observations[cls])
                unpacked_labels.append(np.array([cls] * len(observations[cls])))

        unpacked_points = np.concatenate(unpacked_points)
        unpacked_labels = np.concatenate(unpacked_labels)
        return unpacked_points, unpacked_labels
