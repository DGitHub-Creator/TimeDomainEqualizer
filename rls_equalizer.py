import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def rls_equalizer(z, mo, u, m):
    """

    :param z:
    :param mo:
    :param u:
    :param m:
    :return:
    """
    global w
    # RLS 算法自适应滤波器实现
    M = 31
    w = np.zeros(M)
    p = np.dot(1 / m, np.eye(M))
    NN = len(mo)
    e = [None] * (NN + 2)
    h = [None] * (NN + 2)
    for n in range(M, NN + 1):
        z1 = z[np.arange(n, n - M + 1, - 1)]
        e[n] = m[n] - np.dot(w.T, z1)
        k = np.dot(np.multiply(u ** (- 1), p), z1) / (1 + np.dot(np.dot(np.multiply(u ** (- 1), z1.T), p), z1))
        p = np.multiply(u ** (- 1), p) - np.dot(np.dot(np.multiply(u ** (- 1), k), z1.T), p)
        w = w + np.multiply(k, np.conj(e[n]))
        h[n] = np.dot(w.T, z1)
    return h
