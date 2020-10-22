import sys
import random
import numpy as np
from collections import Counter
from sklearn import svm
from sklearn.cluster import DBSCAN


def distance(x, y, p):
    return np.sum(np.abs(x - y) ** p) ** (1 / p)


class SVMBasedOversample:
    """
    一种基于SVM的过采样方法
    首先是构建SVM分类面，得到决策函数。
    计算出每个样本点的k近邻，根据样本点的k近邻中多数类样本点的数量生成数据，
    多数类样本点越多，生成的数据也就越多。
    在每个样本点上生成数据的过程采用一种迭代爬山的方法，
    寻找几个方向，在这几个方向上行走一小段距离，得到一个新的样本点。
    在新生成的几个样本点中，保留决策函数绝对值最小的点作为新的样本点，迭代进行。
    """

    def __init__(self, kernel='rbf', k=5, p=2, n_steps=20, alpha=3e-3, beta=3e-4):
        '''

        :param kernel: 核函数，默认为高斯核函数
        :param k: k近邻的k值
        :param p: 计算距离的p
        :param n_steps: 迭代的轮次
        :param alpha: 每一步走的距离
        '''
        self.p = p
        self.k = k
        self.kernel = kernel
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta

    def fit_sample(self, X, Y, k=-1, min_class=None):
        if k == -1:
            k = self.k
        classes = np.unique(Y)
        class_sizes = [sum(Y == c) for c in classes]
        if min_class is None:
            min_class = classes[np.argmin(class_sizes)]
        unaltered_Y = Y.copy()
        Y[Y == min_class] = 1
        Y[Y != min_class] = 0
        num_sample, num_feature = X.shape[0], X.shape[1]
        num_oversample = num_sample - 2 * Counter(Y)[1]
        clf = svm.SVC(kernel=self.kernel, probability=False)
        clf.fit(X, Y)
        new_data = []
        k_nearest_majority_samples = {}
        translations = np.zeros(X.shape)
        minority_idxes = np.arange(num_sample)[Y == 1]
        total_majority_k_nearest = 0
        for i in minority_idxes:
            dist_arr = (np.sum(np.abs(X[i] - X) ** self.p, 1)) ** (1 / self.p)
            k_nearest_idxes = np.argsort(dist_arr)[:k + 1]
            k_nearest_labels = Y[k_nearest_idxes]
            k_nearests_cnts = Counter(k_nearest_labels)
            # if k_nearests_cnts[1]<=1:
            #     #少数类样本点附近全部为多数类样本点
            #     continue
            if k_nearests_cnts[1] >= (k):
                # 全部为少数类样本点,或仅有一个多数类样本点，不需要进行任何操作
                continue
            elif k_nearests_cnts[0] >= k or Y[k_nearest_idxes[0]]!=min_class:
                # 全部为多数类样本点，说明该样本点大概率为噪声点，不在该点附近生成新的样本
                continue
            else:
                # 在这个点附近生成数据点，方向向着决策函数绝对值减小的方向
                # 记录下少数类样本个数
                k_nearest_majority_samples[i] = k_nearests_cnts[0]
                total_majority_k_nearest += k_nearest_majority_samples[i]
                # 接下来对多数类样本点进行移出,移出的距离为k近邻中最大的距离
                max_dist = dist_arr[k_nearest_idxes[-1]]
                majority_idxes = np.array([arg for arg in k_nearest_idxes if Y[arg] == 0])
                translations[majority_idxes] += (X[majority_idxes] - X[i]) * (
                        (max_dist - dist_arr[majority_idxes]) / (dist_arr[majority_idxes]+1e-6)).reshape(-1, 1)
        # 对多数类样本点进行平移
        X += translations
        # 针对少数类样本点生成新的数据点
        for i in k_nearest_majority_samples:
            num_new_sample = int(k_nearest_majority_samples[i] / total_majority_k_nearest * num_oversample)
            sample = X[i].copy()
            for _ in range(num_new_sample):
                '''
                可采取一种其他方法，迭代n次，每次在每个方向上随机选-1或者1
                '''
                translation = [0 for ___ in range(num_feature)]
                decision_function_value = sys.maxsize
                for __ in range(self.n_steps):
                    random_directions = self.generate_random_directions(num_feature)
                    sample_directions1 = sample + self.alpha * random_directions
                    sample_directions2 = sample - self.alpha * random_directions
                    if self.decision_function(clf, sample_directions1) < decision_function_value:
                        decision_function_value = self.decision_function(clf, sample_directions1)
                        translation = self.alpha * random_directions
                    if self.decision_function(clf, sample_directions2) < decision_function_value:
                        decision_function_value = self.decision_function(clf, sample_directions2)
                        translation = -self.alpha * random_directions
                new_data.append(sample + translation + self.beta * self.generate_random_directions(num_feature))
                sample = sample + translation
                # translation = [0 for _ in range(num_feature)]
                # decision_function_value = self.decision_function(clf, X[i])
                # possible_directions = self.generate_possible_directions(num_feature)
                # for __ in range(self.n_steps):
                #     if len(possible_directions) == 0:
                #         break
                #     dimension, sign = possible_directions.pop()
                #     modified_translations = translation.copy()
                #     modified_translations[dimension] += sign * self.alpha
                #     modified_decision_function_val = self.decision_function(clf, X[i] + modified_translations)
                #
                #     if modified_decision_function_val < decision_function_value:
                #         translation = modified_translations
                #         decision_function_value = modified_decision_function_val
                #         possible_directions = self.generate_possible_directions(num_feature, (dimension, -sign))
                # new_data.append(X[i] + translation)

        if len(new_data) > 0:
            new_data = np.concatenate(new_data).reshape(-1, num_feature)
            new_label = np.array([min_class] * new_data.shape[0])
            X = np.concatenate([X, new_data])
            Y = np.concatenate([unaltered_Y, new_label])
        return X, Y

    def decision_function(self, clf, sample):
        '''
        返回某个数据项对应的决策函数的绝对值
        :param clf: svm分类器
        :param sample: 样本数据点
        :return: 绝对值大小
        '''
        return abs(clf.decision_function(sample.reshape(1, -1))[0])

    def generate_possible_directions(self, n_dimensions, excluded_direction=None):
        possible_directions = []
        for dimension in range(n_dimensions):
            for sign in [-1, 1]:
                if excluded_direction is None or (excluded_direction[0] != dimension or excluded_direction != sign):
                    possible_directions.append((dimension, sign))
        np.random.shuffle(possible_directions)
        return possible_directions

    def generate_random_directions(self, n_dimensions):
        random_directions = [random.choice([-1, 1]) for _ in range(n_dimensions)]
        return np.array(random_directions)


class MultiSVMBasedOverSample:
    def __init__(self, p=2, k=7, kernel='rbf', n_steps=20, alpha=3e-3, beta=3e-4):
        self.p = p
        self.k = k
        self.kernel = kernel
        self.n_steps = n_steps
        self.alpha = alpha
        self.beta = beta

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
            sam_method = SVMBasedOversample(p=self.p, k=self.k, kernel=self.kernel, n_steps=self.n_steps,
                                            alpha=self.alpha,
                                            beta=self.beta)
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
