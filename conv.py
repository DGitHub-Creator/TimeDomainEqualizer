import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def conv(a, b):
    '''
    卷积/多项式乘法
    :param a:
    :param b:
    :return:
    '''
    if a.shape[0] == 1:
        a = a[0]
    if b.shape[0] == 1:
        b = b[0]
    return np.array(np.poly1d(a) * np.poly1d(b))
