import sys
import random
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.cluster import DBSCAN


def distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


class DbscanBasedOversample:
    """
    一种基于DBSCAN的过采样方法
    该方法首先使用DBSCAN方法聚类，将数据点划分多个块。
    在每个块的边界点上进行重新生成新的数据点
    """

    def __init__(self, eps=0.08, min_pts=8, k=5, p=2, alpha=0.6, radio=15, min_core_number=5):
        self.eps = eps
        self.min_pts = min_pts
        self.alpha = alpha
        self.radio = radio
        self.k = k
        self.p = p
        self.min_core_number = min_core_number

    def fit_sample(self, X, Y, k=-1, min_class=None):
        if k == -1:
            k = self.k
        classes = np.unique(Y)
        class_sizes = [sum(Y == c) for c in classes]
        if min_class is None:
            min_class = classes[np.argmin(class_sizes)]
        num_sample, num_feature = X.shape[0], X.shape[1]
        num_minority = Counter(Y)[min_class]
        num_majority = num_sample - num_minority
        minority_X = X[Y == min_class]
        majority_X = X[Y != min_class]
        minority_Y = Y[Y == min_class]
        majority_Y = Y[Y != min_class]
        if num_majority / num_minority > self.radio:
            # 极度不平衡，生成过多的数据没有什么意义
            num_oversample = 2 * num_minority
        else:
            num_oversample = num_sample - 2 * Counter(Y)[min_class]
        # 把所有少数类样本点放在前面，多数类样本点放在后面，便于处理
        X = np.concatenate([minority_X, majority_X])
        Y = np.concatenate([minority_Y, majority_Y])
        classifier = DBSCAN(eps=self.eps, min_samples=self.min_pts)
        minority_cluster_label = classifier.fit_predict(minority_X)
        # 簇的个数
        num_cluster = max(minority_cluster_label) + 1
        # classifier.core_sample_indices_ 核心点下标
        core_sample_indices = classifier.core_sample_indices_
        noise_sample_indices = np.arange(num_minority)[minority_cluster_label == -1]
        outline_sample_indices = np.ones(minority_cluster_label.shape[0])
        outline_sample_indices[noise_sample_indices] = 0
        outline_sample_indices[core_sample_indices] = 0
        outline_sample_indices = outline_sample_indices != 0
        outline_sample_indices = np.arange(num_minority)[outline_sample_indices]
        import collections
        num_k_nearset_majority = collections.defaultdict(lambda: 1e-3)
        # self.alpha=self.fit_alpha(len(outline_sample_indices)/num_sample)
        num_oversample_outline = num_oversample * self.alpha
        total_k_nearest_majority = 0
        cov_cluster = {}
        cluster_size = {}
        for i in range(num_cluster):
            # 计算出每个cluster所对应的方差大小
            indices = np.tile([False], len(minority_Y))
            indices[core_sample_indices] = True
            indices[minority_cluster_label != i] = False
            if np.sum(indices) >= self.min_core_number:
                cluster_X = minority_X[indices]
            else:
                cluster_X = minority_X[minority_cluster_label == i]
            cov_cluster[i] = np.cov(cluster_X.T) / cluster_X.shape[0]
            cluster_size[i] = len(cluster_X)
        #  print(cluster_size)
        # 多数类样本点的平移
        translations = np.zeros(X.shape)
        for i in outline_sample_indices:
            dist_arr = (np.sum(np.abs(minority_X[i] - X) ** self.p, 1)) ** (1 / self.p)
            k_nearest_idxes = np.argsort(dist_arr)[:k + 1]
            minority_cnt, majority_cnt = Counter(Y[k_nearest_idxes])[min_class], k + 1 - Counter(Y[k_nearest_idxes])[
                min_class]
            if majority_cnt >= k or Y[k_nearest_idxes[0]] != min_class:  # 视为噪声点
                total_k_nearest_majority += num_k_nearset_majority[i]
                continue
            if majority_cnt > 0:
                num_k_nearset_majority[i] += majority_cnt
                max_dist = dist_arr[k_nearest_idxes[-1]]
                majority_idxes = np.array([arg for arg in k_nearest_idxes if Y[arg] != min_class])
                translations[majority_idxes] += (X[majority_idxes] - X[i]) * (
                        (max_dist - dist_arr[majority_idxes]) / (1e-6 + dist_arr[majority_idxes])).reshape(-1, 1)
            total_k_nearest_majority += num_k_nearset_majority[i]
        X += translations
        # print(np.sum(X,axis=0))
        # 生成新的数据
        oversample_outline_data = []
        # 在边界点生成新的数据
        for i in outline_sample_indices:
            cov = cov_cluster[minority_cluster_label[i]]
            oversample_outline_data.append(np.random.multivariate_normal(minority_X[i], cov, int(
                num_oversample_outline * num_k_nearset_majority[i] / (total_k_nearest_majority + 1e-6))))

        if len(oversample_outline_data) > 0:
            oversample_outline_data = np.concatenate(oversample_outline_data).reshape(-1, num_feature)
            new_label = np.array([min_class] * oversample_outline_data.shape[0])
            X = np.concatenate([X, oversample_outline_data])
            Y = np.concatenate([Y, new_label])
        #   print('边界点生成个数{}'.format(len(oversample_outline_data)))
        #  print(total_k_nearest_majority)
        # print(num_k_nearset_majority)
        num_oversample_core = num_oversample - len(oversample_outline_data)
        #   print(num_oversample_core)
        oversample_core_data = []
        for i in range(num_cluster):
            num_oversample_cluster = int(
                num_oversample_core * sum(minority_cluster_label == i) / (sum(minority_cluster_label != -1) + 1e-6))
            #  print('cluster:{},生成数量:{}'.format(i,num_oversample_cluster))
            # 计算出每个cluster所对应的方差大小
            cluster_X = minority_X[minority_cluster_label == i]
            # cov_cluster[i] = np.cov(cluster_X.T) / cluster_X.shape[0]
            oversample_core_data.append(
                np.random.multivariate_normal(np.mean(cluster_X, axis=0), cov_cluster[i] * cluster_size[i],
                                              num_oversample_cluster))
        if len(oversample_core_data) > 0:
            oversample_core_data = np.concatenate(oversample_core_data).reshape(-1, num_feature)
            new_label = np.array([min_class] * oversample_core_data.shape[0])
            X = np.concatenate([X, oversample_core_data])
            Y = np.concatenate([Y, new_label])
        else:  # 在这种情况下，没有生成有效的数据，这说明了通过聚类操作所有的样本点均标记为噪声点，此时可以采用smote采样
            oversample_noise_data = np.random.multivariate_normal(np.mean(minority_X, axis=0), np.cov(minority_X.T),
                                                                  num_oversample)
            new_label = np.array([min_class] * num_oversample)
            X = np.concatenate([X, oversample_noise_data])
            Y = np.concatenate([Y, new_label])
        # 随机重排
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        return X, Y

    def fit_alpha(self, val):
        if val <= 0.1:
            return val * 4
        elif val <= 0.4:
            return (5 / 3) * val + 7 / 30
        return val / 6 + 5 / 6
        # if val<=0.01:
        #     return val*5
        # elif val<=0.05:
        #     return 4*val
        # elif val<=0.3:
        #     return 3*val
        # else:
        #     return 0.92


class MultiDbscanBasedOverSample:
    def __init__(self, p=2, k=7, eps=0.8, min_pts=4):
        self.p = p
        self.k = k
        self.eps = eps
        self.min_pts = min_pts

    def fit_sample(self, X, Y):
        classes = np.unique(Y)
        sizes = np.array([sum(Y == c) for c in classes])
        sorted_idxes = np.argsort(sizes)[::-1]
        classes = classes[sorted_idxes]
        observations = {c: X[Y == c] for c in classes}
        n_max = max(sizes)
        for i in range(1, len(classes)):
            #   self.print_observations(observations)
            tem_class = classes[i]
            n = n_max - len(observations[tem_class])
            used_observations = {}
            unused_observations = {}
            for j in range(i):
                all_indices = list(range(len(observations[classes[j]])))
                # print(len(all_indices))
                # print(int(n_max / i))
                # print('-----------')
                used_indices = np.random.choice(all_indices, min(int(n_max / i), len(all_indices)), replace=False)
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
            sam_method = DbscanBasedOversample(p=self.p, k=self.k, eps=self.eps, min_pts=self.min_pts)
            over_sampled_points, over_sampled_labels = sam_method.fit_sample(unpacked_points, unpacked_labels,
                                                                             min_class=tem_class)
            #    print(Counter(over_sampled_labels))
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

    # def print_observations(self,observations):
    #     dic={}
    #     for c in observations:
    #         dic[c]=len(observations[c])
    #     print(dic)
