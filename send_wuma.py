import numpy as np

from awgn import awgn
from conv import conv
from isi import isi
from sigexpand import sigexpand

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

eps = np.finfo(float).eps


def send_wuma(SNR, w, N_data_2, M):
    """

    :param SNR:
    :param w:
    :param M:
    :return:
    """

    global d
    global st
    global t
    global tt
    global sum
    global N_data
    global N_sample
    global sum_2
    global sum_3
    global st_2
    # global N_data_2

    Ts_2 = 1
    N_sample_2 = 128
    eye_num_2 = 5
    alpha_2 = 1
    # N_data_2 = 10000
    dt_2 = Ts_2 / N_sample_2
    t_2 = np.arange(np.dot(-3, Ts_2), np.dot(3, Ts_2) + 1, dt_2)
    # 产生双极性数字信号
    d_2 = np.sign(np.random.randn(N_data_2))
    dd_2 = sigexpand(d_2, N_sample_2)
    # 基带系统冲击响应
    ht_2 = np.multiply(np.sinc(t_2 / Ts_2), (np.cos(np.dot(np.dot(alpha_2, np.pi), t_2) / Ts_2))) / (
            1 - np.dot(np.dot(4, alpha_2 ** 2), t_2 ** 2) / Ts_2 ** 2 + eps)
    st_2 = conv(dd_2, ht_2)
    tt_2 = np.arange(np.dot(-3, Ts_2), np.dot(np.dot((N_data_2 + 3), N_sample_2), dt_2) - dt_2 + 1, dt_2)
    H_2 = isi(st_2)
    t_H_2 = np.arange(-3, 10003.9921875 + 1, dt_2)
    # 加入噪声得信号st_y
    st_y_2 = awgn(H_2, SNR)
    # 时域均衡
    # M = 31
    NN = len(st_2)
    out_2 = [None] * (NN + 2)
    # print(M, NN + 1)
    for n in range(M, NN + 1):
        # num = ((n - M) / (NN + 1 - M)) * 100
        # print("\r{:^3.0f}% ".format(num), end=' ')
        z1 = st_y_2[np.arange(n, n - M, -1)]
        # print(w.shape)
        # print(z1.shape)
        out_2[n] = np.dot(w.T, z1)

    # 时域均衡前进行抽样判决，并求误码率
    receive_2 = np.zeros(N_data_2)
    for i in range(0, N_data_2):
        receive_2[i] = st_y_2[np.dot(((i + 1) + 2), N_sample_2)]
        if receive_2[i] > 0:
            receive_2[i] = 1
        else:
            receive_2[i] = -1

    sum_2 = 0
    for i in range(0, N_data_2):
        if receive_2[i] != d_2[i]:
            sum_2 = sum_2 + 1

    # 时域均衡后进行抽样判决，并求误码率
    receive_3 = np.zeros(N_data_2)
    sum_3 = 0
    for i in range(0, N_data_2):
        receive_3[i] = out_2[np.dot(((i + 1) + 2), N_sample_2)]
        if receive_3[i] > 0:
            receive_3[i] = 1
        else:
            receive_3[i] = -1
        if receive_3[i] != d_2[i]:
            sum_3 = sum_3 + 1
