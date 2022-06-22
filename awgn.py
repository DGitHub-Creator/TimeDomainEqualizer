import numpy as np

np.set_printoptions(precision=5)  # 设置Numpy输出精度(小数精度)为5位
np.set_printoptions(suppress=True)  # 取消科学计数法，压制Numpy输出精度


def awgn(x, snr, seed=5):
    """
    将白色高斯噪声添加到信号中
    参考链接：https://blog.csdn.net/sinat_41657218/article/details/106171768
    :param x: 原始信号
    :param snr: 信噪比
    :param seed: 随机种子
    :return: 加入噪声后的信号
    """
    # print(x)
    if x.shape[0] == 1:
        x = x[0]
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    noise = np.random.randn(len(x)) * np.sqrt(npower)
    return x + noise
