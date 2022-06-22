import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def eye_image(st, eye_num, dt, N_sample):
    """
    绘制眼图
    :param st:
    :param eye_num:
    :param dt:
    :param N_sample:
    :return:
    """

    ttt = np.arange(0, np.dot(np.dot(eye_num, N_sample), dt) - dt, dt)
    for k in range(3, 30 + 1):
        s1 = st[np.dot(k, N_sample) + 1: np.dot((k + eye_num), N_sample)]
        plt.subplot(212)
        plt.plot(ttt, s1)
