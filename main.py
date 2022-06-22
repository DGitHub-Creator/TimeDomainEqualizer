import time
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
from numpy import *

from awgn import awgn
from eye_image import eye_image
from isi import isi
from noise import noise
from send import send
from xishu import xishu
from source import source
from conv import conv
from sigexpand import sigexpand

warnings.filterwarnings("ignore")

mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

set_printoptions(precision=5)
set_printoptions(suppress=True)

eps = finfo(float).eps


def set_tick():
    """
    对刻度字体进行设置，让上标的符号显示正常
    :return: None
    """
    ax = plt.gca()  # 获取当前图像的坐标轴
    # 更改坐标轴字体，避免出现指数为负的情况
    tick_font = font_manager.FontProperties(family='DejaVu Sans')
    for x_label in ax.get_xticklabels():
        x_label.set_fontproperties(tick_font)  # 设置 x轴刻度字体
    for y_label in ax.get_yticklabels():
        y_label.set_fontproperties(tick_font)  # 设置 y轴刻度字体
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))  # x轴刻度设置为整数
    plt.tight_layout()


def time_convert(seconds):
    """
    将秒数转化为：小时:分钟:秒的形式
    :param seconds: 秒数
    :return: 小时:分钟:秒
    """
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def send_wuma(SNR, w, N_data_2, M):
    """

    :param SNR:
    :param w:
    :param N_data_2:
    :param M:
    :return:
    """

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
    # global N_data_2

    Ts_2 = 1
    N_sample_2 = 128
    eye_num_2 = 5
    alpha_2 = 1
    # N_data_2 = 100
    dt_2 = Ts_2 / N_sample_2
    t_2 = arange(dot(-3, Ts_2), dot(3, Ts_2) + 1, dt_2)
    # 产生双极性数字信号
    d_2 = sign(random.randn(N_data_2))
    dd_2 = sigexpand(d_2, N_sample_2)
    # 基带系统冲击响应
    ht_2 = multiply(sinc(t_2 / Ts_2), (cos(dot(dot(alpha_2, pi), t_2) / Ts_2))) / (
            1 - dot(dot(4, alpha_2 ** 2), t_2 ** 2) / Ts_2 ** 2 + eps)
    st_2 = conv(dd_2, ht_2)
    tt_2 = arange(dot(-3, Ts_2), dot(dot((N_data_2 + 3), N_sample_2), dt_2) - dt_2 + 1, dt_2)
    H_2 = isi(st_2)
    t_H_2 = arange(-3, 10003.9921875 + 1, dt_2)
    # 加入噪声得信号st_y
    st_y_2 = awgn(H_2, SNR)
    # 时域均衡
    NN = len(st_2)
    out_2 = [None] * (NN + 2)
    for n in range(M, NN + 1):
        z1 = st_y_2[arange(n, n - M, -1)]
        out_2[n] = dot(w.T, z1)

    # 时域均衡前进行抽样判决，并求误码率
    receive_2 = zeros(N_data_2)
    for i in range(0, N_data_2):
        receive_2[i] = st_y_2[dot(((i + 1) + 2), N_sample_2)]
        if receive_2[i] > 0:
            receive_2[i] = 1
        else:
            receive_2[i] = -1

    sum_2 = 0
    for i in range(0, N_data_2):
        if receive_2[i] != d_2[i]:
            sum_2 = sum_2 + 1

    # 时域均衡后进行抽样判决，并求误码率
    receive_3 = zeros(N_data_2)
    sum_3 = 0
    for i in range(0, N_data_2):
        receive_3[i] = out_2[dot(((i + 1) + 2), N_sample_2)]
        if receive_3[i] > 0:
            receive_3[i] = 1
        else:
            receive_3[i] = -1
        if receive_3[i] != d_2[i]:
            sum_3 = sum_3 + 1


def receive(st_y):
    """
    !!!此函数没有在主函数中使用
    :param st_y:
    :return:
    """
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

    receive = zeros(N_data)
    for i in range(1, 100 + 1):
        receive[i] = st_y[dot((i + 3), N_sample)]
        if receive[i] > 0:
            receive[i] = 1
        else:
            receive[i] = -1
    sum = 0
    for i in range(1, 100 + 1):
        if receive[i] != d[i]:
            sum = sum + 1


Ts = 1
N_sample = 128
eye_num = 2
alpha = 1
N_data = 100
dt = Ts / N_sample

# **********信源信号************
d, dd, se = source(N_data, N_sample)
plt.figure(num=1, figsize=(12, 8), dpi=200)
plt.subplot(211)
plt.plot(se)
plt.axis([0, se.size, -1.5, 1.5])
plt.title('传输码')
plt.subplot(212)
plt.stem(range(0, dd.size), dd, linefmt=".", )
plt.axis([0, dd.size, -1.5, 1.5])
plt.title('发送滤波器的输入符号序列')
plt.savefig("image/传输码与发送滤波器的输入符号序列图.jpg")
plt.show()

# **********发送滤波器************
t = arange(dot(-3, Ts), dot(3, Ts) + 1, dt)
st = send(dd, dt, Ts, alpha, N_data, N_sample)
plt.figure(num=2, figsize=(12, 8), dpi=200)
plt.subplot(211)
plt.plot(st)
plt.title('经过发送滤波器波形图')
# ************眼图**************
eye_image(st, eye_num, dt, N_sample)
plt.title('经过发送滤波器后的眼图')
plt.savefig("image/经过发送滤波器波形图与眼图.jpg")
plt.show()

# **********码间干扰************
H = isi(st)
t_H = arange(-3, 103.0703125 + 1, dt)
# **********叠加噪声************
SNR = 6
st_y = noise(H, SNR)
plt.figure(num=3, figsize=(12, 8), dpi=200)
plt.plot(st_y)
plt.title('叠加干扰后的波形图')
plt.axis([0, 200, -2, 2])
plt.savefig("image/叠加干扰后的波形图.jpg")
plt.show()

# ************眼图**************
s2 = zeros((1, dot(eye_num, N_sample)))
ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
plt.figure(num=4, figsize=(12, 8), dpi=200)
for k in range(3, 30 + 1):
    s2 = st_y[dot(k, N_sample) + 1:dot((k + eye_num), N_sample)]
    plt.subplot(211)
    plt.plot(ttt, s2)
    plt.title('叠加码间干扰眼图')
# ****************时域均衡 M 为 7*******************
M = 7
w1 = xishu(SNR, M)
NN = len(st)
out_m11 = []
for n in range(M, NN + 1):
    z1 = st_y[arange(n, n - M, -1)]
    out_m11.append(dot(w1.T, z1))
s3 = zeros((1, dot(eye_num, N_sample)))
ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
for k in range(3, 30 + 1):
    s3 = out_m11[dot(k, N_sample) + 1: dot((k + eye_num), N_sample)]
    plt.subplot(212)
    plt.plot(ttt, s3)
    plt.title('7 抽头系数时域均衡后眼图')
plt.savefig("image/叠加码间干扰眼图与7抽头系数时域均衡后眼图.jpg")
plt.show()

# ****************时域均衡 M 为 31*******************
M = 31
w2 = xishu(SNR, M)
NN = len(st)
out_m31 = []
for n in range(M, NN + 1):
    z1 = st_y[arange(n, n - M, -1)]
    out_m31.append(dot(w2.T, z1))
s4 = zeros((1, dot(eye_num, N_sample)))
ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
for k in range(3, 30 + 1):
    s4 = out_m31[dot(k, N_sample) + 1: dot((k + eye_num), N_sample)]
    plt.figure(num=5, figsize=(12, 8), dpi=200)
    plt.subplot(211)
    plt.plot(ttt, s4)
    plt.title('31 抽头系数时域均衡后眼图')
# ****************时域均衡 M 为 99*******************
M = 99
w3 = xishu(SNR, M)
NN = len(st)
out_m99 = []
for n in range(M, NN + 1):
    z1 = st_y[arange(n, n - M, -1)]
    out_m99.append(dot(w3.T, z1))
plt.figure(num=6, figsize=(12, 8), dpi=200)
plt.plot(out_m99)
plt.title('时域均衡后波形图')
plt.savefig("image/时域均衡后波形图.jpg")

s5 = zeros((1, dot(eye_num, N_sample)))
ttt = arange(0, dot(dot(eye_num, N_sample), dt) - dt, dt)
for k in range(3, 30 + 1):
    s5 = out_m99[dot(k, N_sample) + 1:dot((k + eye_num), N_sample)]
    plt.figure(num=5, figsize=(12, 8), dpi=200)
    plt.subplot(212)
    plt.plot(ttt, s5)
    plt.title('99 抽头系数时域均衡后眼图')
plt.savefig("image/33抽头系数与99抽头系数时域均衡后眼图.jpg")
plt.show()

# ************抽样判决 and 误码率**************
receive = zeros(N_data)
for i in range(0, 100):
    receive[i] = st_y[dot(((i + 1) + 1), N_sample)]
    if receive[i] > 0:
        receive[i] = 1
    else:
        receive[i] = -1
plt.figure(num=7, figsize=(12, 8), dpi=200)
plt.subplot(211)
plt.stem(range(0, receive.size), receive, linefmt=".")
plt.title('有码间串扰的波形采样')
plt.axis([0, 100, -1.5, 1.5])
sum = 0
for i in range(0, 100):
    if receive[i] != d[i]:
        sum = sum + 1
receive99 = zeros(N_data)
for i in range(0, 100):
    receive99[i] = out_m99[dot(((i + 1) + 2), N_sample)]
    if receive99[i] > 0:
        receive99[i] = 1
    else:
        receive99[i] = -1
plt.figure(num=7, figsize=(12, 8), dpi=200)
plt.subplot(212)
plt.stem(range(0, receive99.size), receive99, linefmt=".")
plt.title('均衡后波形采样')
plt.axis([0, 100, -1.5, 1.5])
plt.savefig("image/有码间串扰的波形采样图与均衡后波形采样图.jpg")
plt.show()

N_data_2 = 100
sum_2 = 0
sum_3 = 0

# ************时域均衡前抽样判决\误码率曲线图,M 为 7**************
print("时域均衡前抽样判决\\误码率曲线图,M 为 7")
sstart_time = time.time()
SNR_error = zeros(37)
error_rate = zeros(37)
SNR_error_rls_7 = zeros(37)
error_rate_rls_7 = zeros(37)
for SNR in range(-15, 18 + 1, 1):
    start_time = time.time()
    print("{} :{}/{} :".format(SNR, SNR + 15 + 1, 18 + 15 + 1))
    sum_1 = 0
    sum_rls = 0
    # N_data_2 = 100
    w4 = copy(w1)
    M = 7
    for n in range(0, 10):
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("\r\t{}/{} : {}  ".format(n + 1, 10, now_time), end=' ')
        send_wuma(SNR, w4, N_data_2, M)
        sum_1 = sum_1 + sum_2
        sum_rls = sum_rls + sum_3
    print()
    error_sum = sum_1 / 10
    error_sum_rls = sum_rls / 10
    SNR_error[SNR + 16] = SNR
    error_rate[SNR + 16] = error_sum / N_data_2
    SNR_error_rls_7[SNR + 16] = SNR
    error_rate_rls_7[SNR + 16] = error_sum_rls / N_data_2
    end_time = time.time()
    print("运行时间为" + str(time_convert(end_time - start_time)) + "s")
send_time = time.time()
print("M=7 总运行时间为" + str(time_convert(send_time - sstart_time)) + "s")

savetxt("data/SNR_error_" + str(N_data_2) + ".txt", SNR_error)
savetxt("data/error_rate_" + str(N_data_2) + ".txt", error_rate)
savetxt("data/SNR_error_rls_7_" + str(N_data_2) + ".txt", SNR_error_rls_7)
savetxt("data/error_rate_rls_7_" + str(N_data_2) + ".txt", error_rate_rls_7)

# ************时域均衡前抽样判决\误码率曲线图,M 为 31**************
print("时域均衡前抽样判决\\误码率曲线图,M 为 31")
sstart_time = time.time()
SNR_error_rls_31 = zeros(37)
error_rate_rls_31 = zeros(37)
for SNR in range(-15, 18 + 1, 1):
    start_time = time.time()
    print("{} :{}/{} :".format(SNR, SNR + 15 + 1, 18 + 15 + 1))
    sum_1 = 0
    sum_rls = 0
    # N_data_2 = 100
    w4 = copy(w2)
    M = 31
    for n in range(0, 10):
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("\r\t{}/{} : {}  ".format(n + 1, 10, now_time), end=' ')
        send_wuma(SNR, w4, N_data_2, M)
        sum_1 = sum_1 + sum_2
        sum_rls = sum_rls + sum_3
    print()
    error_sum = sum_1 / 10
    error_sum_rls = sum_rls / 10
    SNR_error_rls_31[SNR + 16] = SNR
    error_rate_rls_31[SNR + 16] = error_sum_rls / N_data_2
    end_time = time.time()
    print("运行时间为" + str(time_convert(end_time - start_time)) + "s")
send_time = time.time()
print("M=33 总运行时间为" + str(time_convert(send_time - sstart_time)) + "s")

savetxt("data/SNR_error_rls_31_" + str(N_data_2) + ".txt", SNR_error_rls_31)
savetxt("data/error_rate_rls_31_" + str(N_data_2) + ".txt", error_rate_rls_31)

# ************时域均衡前抽样判决\误码率曲线图,M 为 99**************
print("时域均衡前抽样判决\\误码率曲线图,M 为 99")
sstart_time = time.time()
SNR_error_rls_99 = zeros(37)
error_rate_rls_99 = zeros(37)
for SNR in range(-15, 18 + 1, 1):
    start_time = time.time()
    print("{} :{}/{} :".format(SNR, SNR + 15 + 1, 18 + 15 + 1))
    sum_1 = 0
    sum_rls = 0
    # N_data_2 = 100
    w4 = copy(w3)
    M = 99
    for n in range(0, 10):
        now_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        print("\r\t{}/{} : {}  ".format(n + 1, 10, now_time), end=' ')
        send_wuma(SNR, w4, N_data_2, M)
        sum_1 = sum_1 + sum_2
        sum_rls = sum_rls + sum_3
    print()
    error_sum = sum_1 / 10
    error_sum_rls = sum_rls / 10
    SNR_error_rls_99[SNR + 16] = SNR
    error_rate_rls_99[SNR + 16] = error_sum_rls / N_data_2
    end_time = time.time()
    print("运行时间为" + str(time_convert(end_time - start_time)) + "s")
send_time = time.time()
print("M=99 总运行时间为" + str(time_convert(send_time - sstart_time)) + "s")

savetxt("data/SNR_error_rls_99_" + str(N_data_2) + ".txt", SNR_error_rls_99)
savetxt("data/error_rate_rls_99_" + str(N_data_2) + ".txt", error_rate_rls_99)

# 读取数据
SNR_error = loadtxt("data/SNR_error_" + str(N_data_2) + ".txt")
error_rate = loadtxt("data/error_rate_" + str(N_data_2) + ".txt")
SNR_error_rls_7 = loadtxt("data/SNR_error_rls_7_" + str(N_data_2) + ".txt")
error_rate_rls_7 = loadtxt("data/error_rate_rls_7_" + str(N_data_2) + ".txt")
SNR_error_rls_31 = loadtxt("data/SNR_error_rls_31_" + str(N_data_2) + ".txt")
error_rate_rls_31 = loadtxt("data/error_rate_rls_31_" + str(N_data_2) + ".txt")
SNR_error_rls_99 = loadtxt("data/SNR_error_rls_99_" + str(N_data_2) + ".txt")
error_rate_rls_99 = loadtxt("data/error_rate_rls_99_" + str(N_data_2) + ".txt")

plt.figure(num=8, figsize=(12, 8), dpi=200)
plt.semilogy(SNR_error, error_rate)
plt.semilogy(SNR_error_rls_7, error_rate_rls_7)
plt.semilogy(SNR_error_rls_31, error_rate_rls_31)
plt.semilogy(SNR_error_rls_99, error_rate_rls_99)
plt.legend(['SNR_error_' + str(N_data_2), 'SNR_error_rls_7_' + str(N_data_2), 'SNR_error_rls_31_' + str(N_data_2),
            'SNR_error_rls_99_' + str(N_data_2)])
plt.xlim(-15, 15)
font = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'size': 12,
}
plt.xlabel('r/dB', font)
plt.ylabel('Pe', font)
plt.title("有无均衡器时误码率随信噪比变化曲线")
set_tick()
plt.savefig("image/有无均衡器时误码率随信噪比变化曲线_" + str(N_data_2) + ".jpg")
plt.show()

plt.figure(num=9, figsize=(12, 8), dpi=200)
plt.stem(range(0, w4.size), w4, linefmt=".")
plt.title("99 抽头的抽头系数")
plt.savefig("image/99 抽头的抽头系数.jpg")
plt.show()
