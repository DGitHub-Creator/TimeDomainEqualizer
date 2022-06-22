import numpy as np

from conv import conv

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

eps = np.finfo(float).eps


def send(dd, dt, Ts, alpha, N_data, N_sample):
    """

    :param dd:
    :param dt:
    :param Ts:
    :param alpha:
    :param N_data:
    :param N_sample:
    :return:
    """
    t = np.arange(np.dot(-3, Ts), np.dot(3, Ts) + 1, dt)
    ht = np.multiply(np.sinc(t / Ts), (np.cos(np.dot(np.dot(alpha, np.pi), t) / Ts))) / (
            1 - np.dot(np.dot(4, alpha ** 2), t ** 2) / Ts ** 2 + eps)
    tt = np.arange(np.dot(-3, Ts), np.dot(np.dot((N_data + 3), N_sample), dt) - dt + 1, dt)
    st = conv(dd, ht)
    return st
