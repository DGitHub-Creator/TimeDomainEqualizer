from matplotlib.pyplot import *
from numpy import *
from numpy.random import randn

set_printoptions(suppress=True)
# from smop.libsmop import *

# def conv(lsa, lsb):
#     return array(poly1d(lsa) * poly1d(lsb)).reshape(1, -1)

# def concat(mxs):
#     # print(mxs)
#     mx = np.array(mxs[0])
#     print(type(mx))
#     print(mx.shape)
#     print(mx.size)
#     for i in range(1, len(mxs)):
#         print(mx)
#         mx = concatenate((mx, np.array(mxs[i])), axis=1)
#     return mx

global d
global st
global tt
global sum
global N_data
global N_sample
global sum_2
global sum_3
global st_2
global N_data_2

eps = np.finfo(float).eps


def awgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def conv(lsa, lsb):
    return array(poly1d(lsa) * poly1d(lsb)).reshape(1, -1)


def concat(mxs):
    # print(mxs)
    mx = mxs[0]
    for i in range(1, len(mxs)):
        mx = concatenate((mx, mxs[i]), axis=1)
    return mx


def sigexpand(d=None, M=None):
    N = len(d)
    out = zeros((M, N))
    out[1, :] = d
    out = out.reshape(1, dot(M, N))
    return out


def source(N_data=None, N_sample=None):
    d = sign(randn(1, N_data))
    gt = np.ones(N_sample).reshape(1, -1)
    se = array([])
    for i in range(0, N_data):
        if d[i] == 1:
            if se.size <= 0:
                se = gt
            else:
                se = concat([se, gt])
        else:
            if se.size <= 0:
                se = dot(-1, gt)
            else:
                se = concat([se, dot(-1, gt)])
    dd = sigexpand(d, N_sample)
    return d, dd, se


def send(dd=None, tt=None, t=None, dt=None, Ts=None, alpha=None, N_data=None, N_sample=None):
    t = arange(dot(-3, Ts), dot(3, Ts), dt)
    ht = multiply(sinc(t / Ts), (cos(dot(dot(alpha, pi), t) / Ts))) / (
            1 - dot(dot(4, alpha ** 2), t ** 2) / Ts ** 2 + eps)
    tt = arange(dot(-3, Ts), dot(dot((N_data + 3), N_sample), dt) - dt, dt)
    st = conv(dd, ht)
    return st


def send_wuma(SNR=None, w2=None):
    Ts_2 = 1
    N_sample_2 = 128
    eye_num_2 = 5
    alpha_2 = 1
    N_data_2 = 10000
    dt_2 = Ts_2 / N_sample_2
    t_2 = arange(dot(-3, Ts_2), dot(3, Ts_2), dt_2)
    # 产生双极性数字信号
    d_2 = sign(randn(1, N_data_2))
    dd_2 = sigexpand(d_2, N_sample_2)
    # 基带系统冲击响应
    ht_2 = multiply(sinc(t_2 / Ts_2), (cos(dot(dot(alpha_2, pi), t_2) / Ts_2))) / (
            1 - dot(dot(4, alpha_2 ** 2), t_2 ** 2) / Ts_2 ** 2 + eps)
    st_2 = conv(dd_2, ht_2)
    tt_2 = arange(dot(-3, Ts_2), dot(dot((N_data_2 + 3), N_sample_2), dt_2) - dt_2, dt_2)
    H_2 = isi(st_2)
    t_H_2 = arange(-3, 10003.9921875, dt_2)
    # 加入噪声得信号st_y
    st_y_2 = awgn(H_2, SNR)
    # 时域均衡
    M = 31
    NN = len(st_2)
    out_2 = []
    for n in range(M, NN + 1):
        z1 = concat([st_y_2(arange(n, n - M + 1, -1))]).T
        out_2.append(dot(w2.T, z1))

    # 时域均衡前进行抽样判决，并求误码率
    receive_2 = zeros(N_data_2)
    for i in range(1, N_data_2 + 1):
        receive_2[i] = st_y_2(dot((i + 2), N_sample_2))
        if receive_2[i] > 0:
            receive_2[i] = 1
        else:
            receive_2[i] = -1

    sum_2 = 0
    for i in range(1, N_data_2 + 1):
        if receive_2[i] != d_2(i):
            sum_2 = sum_2 + 1

    # 时域均衡后进行抽样判决，并求误码率
    receive_3 = zeros(N_data_2)
    sum_3 = 0
    for i in range(1, N_data_2 + 1):
        receive_3[0][i] = out_2[dot((i + 2), N_sample_2)]
        if receive_3[0][i] > 0:
            receive_3[0][i] = 1
        else:
            receive_3[0][i] = -1
        if receive_3[0][i] != d_2(i):
            sum_3 = sum_3 + 1


def rls_equalizer(z=None, mo=None, u=None, m=None):
    h = []
    global w
    # RLS 算法自适应滤波器实现
    M = 31
    w = zeros((1, M)).T
    p = dot(1 / m, eye(M))
    NN = len(mo)
    for n in range(M, NN):
        z1 = concat([z(arange(n, n - M + 1, -1))]).T
        e[n] = mo(n) - dot(w.T, z1)
        k = dot(multiply(u ** (-1), p), z1) / (1 + dot(dot(multiply(u ** (-1), z1.T), p), z1))
        p = multiply(u ** (-1), p) - dot(dot(multiply(u ** (-1), k), z1.T), p)
        w = w + multiply(k, conj(e(n)))
        h[n] = dot(w.T, z1)
    return h


def receive(st_y=None):
    receive = zeros(N_data)
    for i in range(1, 100):
        receive[0][i] = st_y(dot((i + 3), N_sample))
        if receive[0][i] > 0:
            receive[0][i] = 1
        else:
            receive[0][i] = -1
    sum = 0
    for i in range(1, 100):
        if receive[0][i] != d(i):
            sum = sum + 1


def noise(H=None, SNR=None):
    st_y = awgn(H, SNR)
    return st_y


def isi(x=None):
    isi_xishu = concat([-0.88, -0.0005, 0.13, -0.088, 0.3, -0.2047, -0.25, 0, -0.00266, -0.038, -0.0005])
    H = conv(isi_xishu, x)
    return H


def eye_image(st=None, eye_num=None, dt=None, N_data=None, N_sample=None):
    s1 = zeros(dot(eye_num, N_sample))
    ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
    for k in arange(3, 30).reshape(-1):
        s1 = st[dot(k, N_sample) + 1: dot((k + eye_num), N_sample)]
        # # drawnow
        subplot(212)
        plot(ttt, s1)
        show()  # hold('on')


def xishu1(SNR=None, M=None):
    Ts_2 = 1
    N_sample_2 = 128
    eye_num_2 = 5
    alpha_2 = 1
    N_data_2 = 100
    dt_2 = Ts_2 / N_sample_2
    t_2 = arange(dot(-3, Ts_2), dot(3, Ts_2), dt_2)
    # 产生双极性数字信号
    d_2 = sign(randn(1, N_data_2))
    dd_2 = sigexpand(d_2, N_sample_2)
    # 基带系统冲击响应
    ht_2 = multiply(sinc(t_2 / Ts_2), (cos(dot(dot(alpha_2, pi), t_2) / Ts_2))) / (
            1 - dot(dot(4, alpha_2 ** 2), t_2 ** 2) / Ts_2 ** 2 + eps)
    st_2 = conv(dd_2, ht_2)
    tt_2 = arange(dot(-3, Ts_2), dot(dot((N_data_2 + 3), N_sample_2), dt_2) - dt_2, dt_2)
    H_2 = isi(st_2)
    t_H_2 = arange(-3, 10003.9921875, dt_2)
    # 加入噪声得信号 st_y
    st_y_2 = awgn(H_2, SNR)
    u = 0.99
    m = 0.009
    w1 = zeros((1, M)).T
    p = dot(1 / m, eye(M))
    NN = len(st_2)
    for n in arange(M, NN).reshape(-1):
        z1 = concat([st_y_2(arange(n, n - M + 1, -1))]).T
        e[n] = st_2[n] - dot(w1.T, z1)
        k = dot(multiply(u ** (-1), p), z1) / (1 + dot(dot(multiply(u ** (-1), z1.T), p), z1))
        p = multiply(u ** (-1), p) - dot(dot(multiply(u ** (-1), k), z1.T), p)
        w1 = w1 + multiply(k, conj(e(n)))


if __name__ == '__main__':

    Ts = 1
    N_sample = 128
    eye_num = 2
    alpha = 1
    N_data = 100
    dt = Ts / N_sample
    # **********信源信号************
    d, dd, se = source(N_data, N_sample)
    figure(1)
    subplot(211)
    plot(se)
    axis(concat([0, 14000, -1.5, 1.5]))
    title('传输码')
    show()  # hold('on')
    subplot(212)
    stem(dd, '.')
    axis(concat([0, 14000, -1.5, 1.5]))
    title('发送滤波器的输入符号序列')
    # **********发送滤波器************
    t = arange(dot(-3, Ts), dot(3, Ts), dt)
    st = send(dd, tt, t, dt, Ts, alpha, N_data, N_sample)
    figure(2)
    subplot(211)
    plot(st)
    title('经过发送滤波器波形图')
    # ************眼图**************
    eye_image(st, eye_num, dt, N_data, N_sample)
    title('经过发送滤波器后的眼图')
    # **********码间干扰************
    H = isi(st)
    t_H = arange(-3, 103.0703125, dt)
    # **********叠加噪声************
    SNR = 6
    st_y = noise(H, SNR)
    figure(3)
    plot(st_y)
    title('叠加干扰后的波形图')
    axis(concat([0, 200, -4, 4]))
    # ************眼图**************
    s2 = zeros((1, dot(eye_num, N_sample)))
    ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
    for k in arange(3, 30).reshape(-1):
        s2 = st_y[dot(k, N_sample) + 1:dot((k + eye_num), N_sample)]
        # drawnow
        figure(4)
        subplot(211)
        plot(ttt, s2)
        show()  # hold('on')
        title('叠加码间干扰眼图')
    # ****************时域均衡 M 为 7*******************
    M = 7
    w1 = xishu1(SNR, M)
    NN = len(st)
    out_m11 = []
    for n in arange(M, NN).reshape(-1):
        z1 = concat([st_y(arange(n, n - M + 1, -1))]).T
        out_m11.append(dot(w1.T, z1))
    s3 = zeros((1, dot(eye_num, N_sample)))
    ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
    for k in arange(3, 30).reshape(-1):
        s3 = out_m11[dot(k, N_sample) + 1: dot((k + eye_num), N_sample)]
        # drawnow
        figure(4)
        subplot(212)
        plot(ttt, s3)
        show()  # hold('on')
        title('7 抽头系数时域均衡后眼图')
    # ****************时域均衡 M 为 31*******************
    M = 31
    w2 = xishu1(SNR, M)
    NN = len(st)
    out_m31 = []
    for n in arange(M, NN).reshape(-1):
        z1 = concat([st_y(arange(n, n - M + 1, -1))]).T
        out_m31.append(dot(w2.T, z1))
    s4 = zeros((1, dot(eye_num, N_sample)))
    ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
    for k in arange(3, 30).reshape(-1):
        s4 = out_m31[dot(k, N_sample) + 1: dot((k + eye_num), N_sample)]
        # drawnow
        figure(5)
        subplot(211)
        plot(ttt, s4)
        show()  # hold('on')
        title('31 抽头系数时域均衡后眼图')
    # ****************时域均衡 M 为 99*******************
    M = 99
    w3 = xishu1(SNR, M)
    NN = len(st)
    out_m99 = []
    for n in arange(M, NN).reshape(-1):
        z1 = concat([st_y(arange(n, n - M + 1, -1))]).T
        out_m99.append(dot(w3.T, z1))
    figure(6)
    plot(out_m99)
    title('时域均衡后波形图')
    s5 = zeros((1, dot(eye_num, N_sample)))
    ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
    for k in arange(3, 30).reshape(-1):
        s5 = out_m99[dot(k, N_sample) + 1:dot((k + eye_num), N_sample)]
        # drawnow
        figure(5)
        subplot(212)
        plot(ttt, s5)
        show()  # hold('on')
        title('99 抽头系数时域均衡后眼图')
    # ************抽样判决 and 误码率**************
    receive = zeros(N_data)
    for i in arange(1, 100).reshape(-1):
        receive[i] = st_y(dot((i + 1), N_sample))
        if receive[i] > 0:
            receive[i] = 1
        else:
            receive[i] = -1
    figure(7)
    subplot(211)
    stem(receive, '.')
    title('有码间串扰的波形采样')
    axis(concat([0, 100, -1.5, 1.5]))
    sum = 0
    for i in arange(1, 100).reshape(-1):
        if receive[i] != d[i]:
            sum = sum + 1
    receive99 = zeros(N_data)
    for i in arange(1, 100).reshape(-1):
        receive99[i] = out_m99[dot((i + 2), N_sample)]
        if receive99[i] > 0:
            receive99[i] = 1
        else:
            receive99[i] = -1
    figure(7)
    subplot(212)
    stem(receive99, '.')
    title('均衡后波形采样')
    axis(concat([0, 100, -1.5, 1.5]))
    # ************时域均衡前抽样判决\误码率曲线图,M 为 7**************
    SNR_error = zeros((1, 37))
    error_rate = zeros((1, 37))
    SNR_error_rls_7 = zeros((1, 37))
    error_rate_rls_7 = zeros((1, 37))
    for SNR in arange(-15, 18, 1).reshape(-1):
        sum_1 = 0
        sum_rls = 0
        N_data_2 = 100
        w4 = copy(w1)
        M = 7
        for n in arange(1, 10).reshape(-1):
            send_wuma(SNR, w4, N_data_2, M)
            sum_1 = sum_1 + sum_2
            sum_rls = sum_rls + sum_3
        error_sum = sum_1 / 10
        error_sum_rls = sum_rls / 10
        SNR_error[SNR + 16] = SNR
        error_rate[SNR + 16] = error_sum / N_data_2
        SNR_error_rls_7[SNR + 16] = SNR
        error_rate_rls_7[SNR + 16] = error_sum_rls / N_data_2
    figure(8)
    semilogy(SNR_error, error_rate, SNR_error_rls_7, error_rate_rls_7, 'r')
    show()  # hold('on')
    title('有无均衡器时误码率随信噪比变化曲线')
    xlabel('r/dB')
    xlabel('Pe')
    show()  # hold('on')
    # ************时域均衡前抽样判决\误码率曲线图,M 为 31**************
    SNR_error_rls_31 = zeros((1, 37))
    error_rate_rls_31 = zeros((1, 37))
    for SNR in arange(-15, 18, 1).reshape(-1):
        sum_1 = 0
        sum_rls = 0
        N_data_2 = 100
        w4 = copy(w2)
        M = 31
        for n in arange(1, 10).reshape(-1):
            send_wuma(SNR, w4, N_data_2, M)
            sum_1 = sum_1 + sum_2
            sum_rls = sum_rls + sum_3
        error_sum = sum_1 / 10
        error_sum_rls = sum_rls / 10
        SNR_error_rls_31[SNR + 16] = SNR
        error_rate_rls_31[SNR + 16] = error_sum_rls / N_data_2
    figure(8)
    semilogy(SNR_error_rls_31, error_rate_rls_31, 'g')
    show()  # hold('on')
    title('有无均衡器时误码率随信噪比变化曲线')
    xlabel('r/dB')
    ylabel('Pe')
    show()  # hold('on')
    SNR_error_rls_99 = zeros((1, 37))
    error_rate_rls_99 = zeros((1, 37))
    for SNR in arange(-15, 18, 1).reshape(-1):
        sum_1 = 0
        sum_rls = 0
        N_data_2 = 100
        w4 = copy(w3)
        M = 99
        for n in arange(1, 10).reshape(-1):
            send_wuma(SNR, w4, N_data_2, M)
            sum_1 = sum_1 + sum_2
            sum_rls = sum_rls + sum_3
        error_sum = sum_1 / 10
        error_sum_rls = sum_rls / 10
        SNR_error_rls_31[SNR + 16] = SNR
        error_rate_rls_31[SNR + 16] = error_sum / N_data_2
        SNR_error_rls_99[SNR + 16] = SNR
        error_rate_rls_99[SNR + 16] = error_sum_rls / N_data_2
    figure(8)
    semilogy(SNR_error_rls_99, error_rate_rls_99, 'm')
    show()
    show()  # hold('on')
    figure(9)
    stem(w4, '.')
    show()  # hold('on')
    # ************时域均衡前抽样判决\误码率曲线图,M 为 7**************
    SNR_error = zeros((1, 37))
    error_rate = zeros((1, 37))
    SNR_error_rls_7 = zeros((1, 37))
    error_rate_rls_7 = zeros((1, 37))
    for SNR in arange(-15, 18, 1).reshape(-1):
        sum_1 = 0
        sum_rls = 0
        N_data_2 = 10000
        w4 = copy(w1)
        M = 7
        for n in arange(1, 100).reshape(-1):
            send_wuma(SNR, w4, N_data_2, M)
            sum_1 = sum_1 + sum_2
            sum_rls = sum_rls + sum_3
        error_sum = sum_1 / 100
        error_sum_rls = sum_rls / 100
        SNR_error[SNR + 16] = SNR
        error_rate[SNR + 16] = error_sum / N_data_2
        SNR_error_rls_7[SNR + 16] = SNR
        error_rate_rls_7[SNR + 16] = error_sum_rls / N_data_2
    figure(10)
    semilogy(SNR_error, error_rate, SNR_error_rls_7, error_rate_rls_7, 'r')
    show()  # hold('on')
    title('有无均衡器时误码率随信噪比变化曲线')
    xlabel('r/dB')
    xlabel('Pe')
    show()  # hold('on')
    # ************时域均衡前抽样判决\误码率曲线图,M 为 31**************
    SNR_error_rls_31 = zeros((1, 37))
    error_rate_rls_31 = zeros((1, 37))
    for SNR in arange(-15, 18, 1).reshape(-1):
        sum_1 = 0
        sum_rls = 0
        N_data_2 = 10000
        w4 = copy(w2)
        M = 31
        for n in arange(1, 100).reshape(-1):
            send_wuma(SNR, w4, N_data_2, M)
            sum_1 = sum_1 + sum_2
            sum_rls = sum_rls + sum_3
        error_sum = sum_1 / 100
        error_sum_rls = sum_rls / 100
        SNR_error_rls_31[SNR + 16] = SNR
        error_rate_rls_31[SNR + 16] = error_sum_rls / N_data_2
    figure(10)
    semilogy(SNR_error_rls_31, error_rate_rls_31, 'g')
    show()  # hold('on')
    title('有无均衡器时误码率随信噪比变化曲线')
    xlabel('r/dB')
    ylabel('Pe')
    show()  # hold('on')
    SNR_error_rls_99 = zeros((1, 37))
    error_rate_rls_99 = zeros((1, 37))
    for SNR in arange(-15, 18, 1).reshape(-1):
        sum_1 = 0
        sum_rls = 0
        N_data_2 = 10000
        w4 = copy(w3)
        M = 99
        for n in arange(1, 100).reshape(-1):
            send_wuma(SNR, w4, N_data_2, M)
            sum_1 = sum_1 + sum_2
            sum_rls = sum_rls + sum_3
        error_sum = sum_1 / 100
        error_sum_rls = sum_rls / 100
        SNR_error_rls_31[SNR + 16] = SNR
        error_rate_rls_31[SNR + 16] = error_sum / N_data_2
        SNR_error_rls_99[SNR + 16] = SNR
        error_rate_rls_99[SNR + 16] = error_sum_rls / N_data_2
    figure(10)
    semilogy(SNR_error_rls_99, error_rate_rls_99, 'm')
    show()  # hold('on')
    pass
