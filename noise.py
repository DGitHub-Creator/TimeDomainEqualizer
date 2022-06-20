import numpy as np

from isi import isi
from awgn import awgn

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def noise(H, SNR):
    '''

    :param H:
    :param SNR:
    :return:
    '''
    st_y = awgn(H, SNR)
    return st_y

# a = isi(10)
# print(a)
# b = noise(H=a, SNR=-16)
# print(b)
