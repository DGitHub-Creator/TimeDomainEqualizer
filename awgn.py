import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def awgn(x, snr, seed=5):
    '''

    :param x: 原始信号
    :param snr: 信噪比
    :param seed: 随机种子
    :return: 加入噪声后的信号
    '''
    # print(x)
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise
