import numpy as np

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def receive(st_y):
    '''

    :param st_y:
    :return:
    '''
    global d
    global st
    global t
    global tt
    global sum
    global N_data
    global N_sample
    global sum_2
    global sum_3
    global st_2
    global N_data_2

    receive = np.zeros(N_data)
    for i in range(1, 100 + 1):
        receive[i] = st_y[np.dot((i + 3), N_sample)]
        if receive[i] > 0:
            receive[i] = 1
        else:
            receive[i] = -1
    sum = 0
    for i in range(1, 100 + 1):
        if receive[i] != d[i]:
            sum = sum + 1
