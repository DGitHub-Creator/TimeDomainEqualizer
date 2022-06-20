import numpy as np

from sigexpand import sigexpand

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)


def concat(mxs):
    '''

    :param mxs:
    :return:
    '''

    mx = np.array(mxs[0])
    # print(type(mx))
    # print(mx.shape)
    # print(mx.size)
    for i in range(1, len(mxs)):
        # print(mx)
        mx = np.concatenate((mx, np.array(mxs[i])), axis=1)
    return mx


def source(N_data, N_sample):
    '''

    :param N_data:
    :param N_sample:
    :return: d(多行),dd(一行多列),se(一行多列)
    '''
    d = np.sign(np.random.randn(N_data))
    gt = np.ones(N_sample).reshape(1, -1)
    se = np.array([])
    for i in range(0, N_data):
        if d[i] == 1:
            if se.size <= 0:
                se = gt
            else:
                se = concat([se, gt])
        else:
            if se.size <= 0:
                se = np.dot(-1, gt)
            else:
                se = concat([se, np.dot(-1, gt)])
    dd = sigexpand(d, N_sample)
    if dd.shape[0] == 1:
        dd = dd[0]
    if se.shape[0] == 1:
        se = se[0]
    return d, dd, se

# a, b, c = source(12, 5)
# print(a)
# print(a.shape)
# print(b)
# print(b.shape)
# print(c)
# print(c.shape)
