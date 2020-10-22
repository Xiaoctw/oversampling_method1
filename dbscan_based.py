import sys
import random
import numpy as np
from collections import Counter, defaultdict
from sklearn import svm
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from imblearn.under_sampling import EditedNearestNeighbours


class DbscanBasedOversample:

    def __init__(self, eps=0.08, min_pts=8, k=5, p=2, outline_radio=0.6, imbalance_radio=15, min_core_number=5,
                 noise_radio=0.3, multiple_k=4, filter_majority=False):
        """
        :param eps: dbscan半径大小
        :param min_pts: 区域内最小点数
        :param k: k近邻
        :param p: 距离度量
        :param outline_radio: 比率，在边界点生成数据点的比率
        :param imbalance_radio: 不平衡率，极度不平衡情况下生成数量较少
        :param min_core_number: 每个簇最少的少数类样本数
        :param multiple_k:单个少数类样本最多生成数量是k的几倍，限制单个样本生成数量，避免生成区域畸形
        """
        self.eps = eps
        self.min_pts = min_pts
        self.outline_radio = outline_radio
        self.imbalance_radio = imbalance_radio
        self.noise_radio = noise_radio
        self.multiple_k = multiple_k
        self.k = k
        self.p = p
        self.min_core_number = min_core_number
        self.filter_majority = filter_majority

    def fit_sample(self, X, Y, k=-1, minority_class=None):
        if k == -1:
            k = self.k
        classes = np.unique(Y)
        classes_size = [sum(Y == c) for c in classes]
        if minority_class is None:
            minority_class = classes[np.argmin(classes_size)]
        if self.filter_majority:
            # 如果需要降采样进行一步降采样
            X, Y = EditedNearestNeighbours().fit_sample(X, Y)
        num_sample, num_feature = X.shape[0], X.shape[1]
        num_minority = Counter(Y)[minority_class]
        num_majority = num_sample - num_minority
        minority_X = X[Y == minority_class]
        majority_X = X[Y != minority_class]
        minority_Y = Y[Y == minority_class]
        majority_Y = Y[Y != minority_class]
        if num_majority / num_minority > self.imbalance_radio:
            # 数据过于不平衡，这种情况下不建议生成过多该样本数据
            num_oversample = (num_majority - num_minority) // 5
        else:
            num_oversample = num_majority - num_minority
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
        if len(noise_sample_indices) / num_minority < self.noise_radio:
            # 说明噪声类样本点数量不是很多，不在上面生成数据
            num_oversample_noise = 0
        else:
            num_oversample_noise = int(self.radio_noise(len(noise_sample_indices) / num_minority) * num_oversample)
        num_oversample -= num_oversample_noise
        self.outline_radio=self.fit_alpha(len(outline_sample_indices)/num_minority)
        num_oversample_outline = num_oversample * self.outline_radio
        total_k_nearest_majority = 0
        cov_cluster, cluster_size = {}, {}
        # 找到每个簇和对应簇的大小
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
        dist_mat = distance_matrix(minority_X, X, p=self.p)
        num_k_nearest_majority = defaultdict(lambda: 1e-3)
        translations = np.zeros(X.shape)
        for i in outline_sample_indices:
            dist_arr = dist_mat[i]
            k_nearest_idxes = np.argsort(dist_arr)[:k + 1]
            minority_cnt, majority_cnt = Counter(Y[k_nearest_idxes])[minority_class], k + 1 - \
                                         Counter(Y[k_nearest_idxes])[
                                             minority_class]
            if majority_cnt >= k or Y[k_nearest_idxes[0]] != minority_class:
                # 这种情况下生成在该点附近生成样本点很少，因为很有可能为噪声点
                total_k_nearest_majority += num_k_nearest_majority[i]
                continue
            if majority_cnt > 0:
                num_k_nearest_majority[i] += majority_cnt
                max_dist = dist_arr[k_nearest_idxes[-1]]
                majority_idxes = np.array([arg for arg in k_nearest_idxes if Y[arg] != minority_class])
                translations[majority_idxes] += (X[majority_idxes] - X[i]) * (
                        (max_dist - dist_arr[majority_idxes]) / (1e-6 + dist_arr[majority_idxes])).reshape(-1, 1)
            total_k_nearest_majority += num_k_nearest_majority[i]
        X += translations  # 对多数类样本点进行位移
        oversample_outline_data = []
        for i in outline_sample_indices:
            cov = cov_cluster[minority_cluster_label[i]]
            num = min(int(k * self.multiple_k),
                      1 + int(num_oversample_outline * num_k_nearest_majority[i] / (total_k_nearest_majority + 1e-6)))
            # print(num)
            oversample_outline_data.append(np.random.multivariate_normal(minority_X[i], cov, num))

        if len(oversample_outline_data) > 0:
            oversample_outline_data = np.concatenate(oversample_outline_data).reshape(-1, num_feature)
            new_label = np.array([minority_class] * oversample_outline_data.shape[0])
            X = np.concatenate([X, oversample_outline_data])
            Y = np.concatenate([Y, new_label])

        num_oversample_core = max(num_oversample - len(oversample_outline_data), 0)
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
            new_label = np.array([minority_class] * oversample_core_data.shape[0])
            X = np.concatenate([X, oversample_core_data])
            Y = np.concatenate([Y, new_label])

        oversample_noise_data = []
        # if len(noise_sample_indices)>=num_oversample_noise:
        for _ in range(num_oversample_noise):
            i = np.random.choice(noise_sample_indices)
            dist_arr = dist_mat[i]
            k_nearest_idxes = np.argsort(dist_arr)[1:k + 1]
            point = X[np.random.choice(k_nearest_idxes)]
            rate = random.random()
            oversample_noise_data.append([point * rate + X[i] * (1 - rate)])
        # else:
        #     for i in noise_sample_indices:
        #         dist_arr=dist_mat[i]
        #         k_nearest_idxes=np.argsort(dist_arr)[1:k+1]
        #         for _ in range(num_oversample_noise//len(noise_sample_indices)):
        #             point=X[np.random.choice(k_nearest_idxes)]
        #             rate=random.random()
        #             oversample_noise_data.append([point*rate+X[i]*(1-rate)])

        if len(oversample_noise_data) > 0:
            oversample_noise_data = np.concatenate(oversample_noise_data).reshape(-1, num_feature)
            new_label = np.array([minority_class] * oversample_noise_data.shape[0])
            X = np.concatenate([X, oversample_noise_data])
            Y = np.concatenate([Y, new_label])

            # 随机重排
        print('核心生成点:{},边界生成点:{},噪声生成点:{}'.format(len(oversample_core_data), len(oversample_outline_data),
                                                  len(noise_sample_indices)))
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        return X, Y

    def fit_alpha(self,val):
        if val<=0.01:
            return val*5
        elif val<=0.05:
            return 4*val
        elif val<=0.3:
            return 3*val
        else:
            return 0.92

    def radio_noise(self, radio):
        a = 0.9 / (1 - self.noise_radio ** 2)
        return a * (radio ** 2) + 1 - a


class MultiDbscanBasedOverSample:
    def __init__(self, p=2, k=7, eps=0.8, min_pts=4,outline_radio=0.6, imbalance_radio=15, min_core_number=5,
                 noise_radio=0.3, multiple_k=4, filter_majority=False):
        self.p = p
        self.k = k
        self.eps = eps
        self.min_pts = min_pts
        self.outline_radio=outline_radio
        self.imbalance_radio=imbalance_radio
        self.min_core_number=min_core_number
        self.noise_radio=noise_radio
        self.multiple_k=multiple_k
        self.filter_majority=filter_majority

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
            sam_method = DbscanBasedOversample(p=self.p, k=self.k, eps=self.eps, min_pts=self.min_pts, outline_radio=self.outline_radio,
                                               imbalance_radio=self.imbalance_radio, min_core_number=self.min_core_number, noise_radio=self.noise_radio, multiple_k=self.multiple_k, filter_majority=self.filter_majority)
            over_sampled_points, over_sampled_labels = sam_method.fit_sample(unpacked_points, unpacked_labels,
                                                                             minority_class=tem_class)
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
