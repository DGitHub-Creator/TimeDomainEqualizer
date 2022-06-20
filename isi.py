import numpy as np

from conv import conv

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def isi(x):
    '''

    :param x:
    :return:
    '''
    # isi_xishu = [-0.88, -0.0005, 0.13, -0.088, 0.3, -0.2047, -0.25, 0, -0.00266, -0.038, -0.0005]
    isi_xishu = np.array([-0.0005, 0.63, -0.88, 2.6, 0.5, -0.9047, -0.25, 0, -0.00266, -0.038, -0.88])
    H = conv(isi_xishu, x)
    return H

# a = isi(10)
# print(a)
# print(a.shape)
# print(a.size)
# print(type(a))
