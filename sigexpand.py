import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def sigexpand(d, M):
    """

    :param d:
    :param M:
    :return:
    """
    N = d.size
    out = np.zeros((M, N))
    out[0, :] = d
    out = np.reshape(out, (1, np.dot(M, N)), order='F')
    return out
