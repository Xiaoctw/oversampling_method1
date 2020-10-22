import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import norm
import keras.backend as K
from keras.layers import Input, Dense, Lambda
from keras.models import Model, Sequential
from datasets import *

# 生成数据
mean2 = [2, 2]
cov2 = [[-3, -2], [-2, -3]]


# X = np.random.multivariate_normal(mean2, cov2, num_data)
# Y = np.tile([1], num_data)


def generate_data_vae(X, num_new_data):
    original_dim = X.shape[1]
    intermediate_dim = 8
    latent_dim = 2
    epochs = 100
    batch_size = 32

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=K.shape(z_mean))
        return z_mean + K.exp(z_log_var / 2) * epsilon

    # 这部分相当于是编码器部分
    x = Input(shape=(original_dim,))
    encoder_sequential = Sequential(
        [Dense(intermediate_dim, activation='relu'),
         Dense(intermediate_dim, activation='relu')]
    )
    h = encoder_sequential(x)
    z_mean = Dense(latent_dim)(h)
    z_log_var = Dense(latent_dim)(h)

    # 连接解码器和编码器的z
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # 解码器部分
    decoder_sequential = Sequential(
        [Dense(intermediate_dim, activation='relu'),
         Dense(intermediate_dim, activation='relu')]
    )
    # 获得输出的层
    decoder_mean = Dense(original_dim)
    h_decoded = decoder_sequential(z)
    x_decoded_mean = decoder_mean(h_decoded)
    # 建立模型
    vae = Model(x, x_decoded_mean)
    # xent_loss是重构loss，kl_loss是KL loss
    # xent_loss = K.sum((K.sqrt(K.square(x-x_decoded_mean))), axis=-1)
    # xent_loss=K.sum((x-x_decoded_mean)**2)
    xent_loss = K.sum(((K.square(x - x_decoded_mean))), axis=-1)
    #  print('xent_loss:{}'.format(xent_loss.shape))
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    # kl_loss=- 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var))
    # print('kl_loss:{}'.format(kl_loss.shape))
    vae_loss = K.mean(xent_loss + kl_loss)
    #  print(vae_loss.shape)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    history = vae.fit(X, shuffle=True, epochs=epochs, batch_size=batch_size)
    # plt.plot(history.history['loss'])
    # plt.show()

    # 构建生成器
    generator_input = Input(shape=(latent_dim,))
    h_generator = decoder_sequential(generator_input)
    x_generator_mean = decoder_mean(h_generator)
    generator = Model(generator_input, x_generator_mean)

    grid_x = np.random.normal(0, 1, num_new_data)
    grid_y = np.random.normal(0, 1, num_new_data)

    appended = []
    for i in range(num_new_data):
        xi, yi = grid_x[i], grid_y[i]
        z_sample = np.array([[xi, yi]])
        x_generate = generator.predict(z_sample)
        appended.append(x_generate)

    appended = np.concatenate(appended)
    # X = np.concatenate([X, appended])
    # Y = np.concatenate([Y, np.tile([2], len(appended))])
    # plt.scatter(X[:, 0], X[:, 1], c=Y)
    # plt.show()
    return appended


class VaeOversample:
    def fit_sample(self, X, Y):
        classes = np.unique(Y)
        sizes = np.array([sum(Y == c) for c in classes])
        sorted_idxes = np.argsort(sizes)[::-1]
        classes = classes[sorted_idxes]
        sizes = sizes[sorted_idxes]
        generate_X = []
        generate_Y = []
        max_size = max(sizes)
        for i in range(1, len(classes)):
            num_generate = max_size - sizes[i]
            minority_X = X[Y == classes[i]]
            generate_X.append(generate_data_vae(minority_X, num_generate))
            generate_Y.append([classes[i]] * num_generate)
        if len(generate_Y) > 0:
            generate_X = np.concatenate(generate_X)
            generate_Y = np.concatenate(generate_Y)
        X = np.concatenate([X, generate_X])
        Y = np.concatenate([Y, generate_Y])
        return X, Y


if __name__ == '__main__':
    # X, Y = pre_ecoli()
    # X,Y=VaeOversample().fit_sample(X,Y)
    mean1 = [5, 5]
    cov = [[5, 0], [0, 5]]
    mean2 = [-5, -5]
    num_data = 1000
    num_generate_data = 150
    X1 = np.random.multivariate_normal(mean1, cov, num_data // 2)
    X2 = np.random.multivariate_normal(mean2, cov, num_data // 2)
    Y = np.tile([1], num_data)
    X3 = generate_data_vae(np.concatenate([X1,X2]), num_generate_data)
    Y1 = np.tile([2], num_generate_data)
    X = np.concatenate([X1, X2, X3])
    Y = np.concatenate([Y, Y1])
    plt.rcParams['figure.figsize'] = (6.0, 7.0)
    plt.scatter(X[:, 0], X[:, 1], c=Y)
    plt.colorbar()
    plt.show()
    print(Counter(Y))
