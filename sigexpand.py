import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def sigexpand(d, M):
    '''

    :param d:
    :param M:
    :return:
    '''
    N = d.size
    # print(M, N)
    out = np.zeros((M, N))
    # print(out)
    out[0, :] = d
    # print(out)
    out = np.reshape(out, (1, np.dot(M, N)), order='F')
    return out

# print(sigexpand(np.array([10, 20]), 10))
