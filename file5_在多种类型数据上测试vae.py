import numpy as np
from sklearn import datasets
from algorithm_vae import generate_data_vae
import matplotlib.pyplot as plt


def generate_density_based_data():
    X1, y1 = datasets.make_circles(n_samples=5000, factor=.6,
                                   noise=.05)
    X2, y2 = datasets.make_blobs(n_samples=1000, n_features=2, centers=[[1.2, 1.2]], cluster_std=[[.1]],
                                 random_state=9)
    X = np.concatenate((X1, X2))
    return X2


if __name__ == '__main__':
    X = generate_density_based_data()
    generate_data_vae(X, 500)
