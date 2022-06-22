import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def conv(a, b):
    """
    卷积/多项式乘法
    参考链接：https://blog.csdn.net/TomorrowAndTuture/article/details/106249179 多项式计算
    :param a:多项式a
    :param b:多项式b
    :return:多项式相乘的结果
    """
    if a.shape[0] == 1:
        a = a[0]
    if b.shape[0] == 1:
        b = b[0]
    return np.array(np.poly1d(a) * np.poly1d(b))
