import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def rls(x, d, N=4, lmbd=0.999, delta=0.0002):
    L = min(len(x), len(d))
    lmbd_inv = 1 / lmbd
    h = np.zeros((N, 1))
    P = np.eye(N) / delta
    e = np.zeros(L - N)
    for n in range(L - N):
        x_n = np.array(x[n:n + N][::-1]).reshape(N, 1)
        d_n = d[n]
        y_n = np.dot(x_n.T, h)
        e_n = d_n - y_n
        g = np.dot(P, x_n)
        g = g / (lmbd + np.dot(x_n.T, g))
        h = h + e_n * g
        P = lmbd_inv * (P - np.dot(g, np.dot(x_n.T, P)))
        e[n] = e_n
    return e


def rls(z, mo, u, m, c):
    # Init
    M = 31
    sig_len = len(z) - M + 1
    p = np.eye(M)
    w = np.zeros(sig_len)
    errn = np.zeros(len(z))
    wn = np.zeros((sig_len, M))
    # Loop
    for i in range(0, sig_len):
        if i == 0:
            p_old = I / c
            w_old = np.zeros(M)
        d = dk[i]
        x = np.flipud(z[i:i + M])
        xt = np.reshape(x, (-1, 1)).T  # x的转置
        k = np.dot(p_old, x) / (lambd + np.dot(np.dot(xt, p_old), x))
        y = np.dot(xt, w_old)  # FIR滤波器输出
        err = d - y  # 估计误差
        ket = k * err
        w = w_old + ket  # 计算滤波器系数矢量
        p = np.dot((I - np.dot(np.reshape(k, (-1, 1)), xt)), p_old) / lambd
        # 更新
        p_old = p
        w_old = w
        # 滤波结果存储
        yn[i] = y
        errn[i] = err
        wn[i] = np.reshape(w, (-1, 1)).T
    return yn, errn, wn
