import numpy as np

from awgn import awgn
from conv import conv
from isi import isi
from sigexpand import sigexpand

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

eps = np.finfo(float).eps


def xishu(SNR, M):
    '''

    :param SNR:
    :param M:
    :return:
    '''
    Ts_2 = 1
    N_sample_2 = 128
    eye_num_2 = 5
    alpha_2 = 1
    N_data_2 = 100
    dt_2 = Ts_2 / N_sample_2
    t_2 = np.arange(np.dot(-3, Ts_2), np.dot(3, Ts_2) + 1, dt_2)
    # 产生双极性数字信号
    d_2 = np.sign(np.random.randn(1, N_data_2))
    dd_2 = sigexpand(d_2, N_sample_2)
    # 基带系统冲击响应
    ht_2 = np.multiply(np.sinc(t_2 / Ts_2), (np.cos(np.dot(np.dot(alpha_2, np.pi), t_2) / Ts_2))) / (
            1 - np.dot(np.dot(4, alpha_2 ** 2), t_2 ** 2) / Ts_2 ** 2 + eps).reshape(1, -1)

    # print(dd_2.shape)
    # print(dd_2)
    # print(ht_2.shape)
    # print(ht_2)
    st_2 = conv(dd_2, ht_2)
    # print(st_2)
    # print(st_2.shape)
    tt_2 = np.arange(np.dot(-3, Ts_2), np.dot(np.dot((N_data_2 + 3), N_sample_2), dt_2) - dt_2 + 1, dt_2)
    H_2 = isi(st_2)
    # print(H_2)
    t_H_2 = np.arange(-3, 10003.9921875 + 1, dt_2)
    # 加入噪声得信号 st_y
    st_y_2 = awgn(H_2, SNR)
    u = 0.99
    m = 0.009
    w = np.zeros(M)
    # print(w1)
    p = np.dot(1 / m, np.eye(M))
    NN = len(st_2)
    e = [None] * (NN + 2)
    # print(st_2)
    for n in range(M, NN + 1):
        # print(np.arange(n, n - M, -1))
        z1 = st_y_2[np.arange(n, n - M, -1)]
        # print(n, M, (NN + 2), len(st_2))
        e[n] = st_2[n - 1] - np.dot(w.T, z1)
        k = np.dot(np.multiply(u ** (-1), p), z1) / (1 + np.dot(np.dot(np.multiply(u ** (-1), z1.T), p), z1))
        p = np.multiply(u ** (-1), p) - np.dot(np.dot(np.multiply(u ** (-1), k), z1.T), p)
        # print(w1)
        # print(np.multiply(k, np.conj(e[n])))
        w = w + np.multiply(k, np.conj(e[n]))
        # print(w1)
        # time.sleep((100))

    return w

# print(xishu1(10, 31).shape)
# print(xishu1(10, 31))
