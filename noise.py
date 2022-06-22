import numpy as np

from awgn import awgn

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def noise(H, SNR):
    """

    :param H:
    :param SNR:
    :return:
    """
    st_y = awgn(H, SNR)
    return st_y

