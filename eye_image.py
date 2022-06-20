import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def eye_image(st, eye_num, dt, N_data, N_sample):
    '''

    :param st:
    :param eye_num:
    :param dt:
    :param N_data:
    :param N_sample:
    :return:
    '''

    s1 = np.zeros(np.dot(eye_num, N_sample))
    ttt = np.arange(0, np.dot(np.dot(eye_num, N_sample), dt) - dt, dt)
    for k in range(3, 30 + 1):
        s1 = st[np.dot(k, N_sample) + 1: np.dot((k + eye_num), N_sample)]
        plt.subplot(212)
        plt.plot(ttt, s1)

#
# ttt=[1,2,3,4,5,6,7,8,9]
# s1 = [11,22,33,44,55,66,77,88,99]
# for k in range(3, 30 + 1):
#     subplot(212)
#     plot(ttt, s1)
# show()
