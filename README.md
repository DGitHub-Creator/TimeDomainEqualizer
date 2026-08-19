# TimeDomainEqualizer

通信原理课程设计：数字基带传输系统中的**时域均衡仿真**。通过仿真发送滤波器、信道、码间干扰（ISI）与噪声环境，验证横向滤波器（时域均衡器）对码间串扰的抑制效果，并对比均衡前后的眼图与误码率性能。

## 功能特性

- 随机符号序列生成与发送滤波成形（`source.py`、`send.py`）
- 信道码间干扰（ISI）建模（`isi.py`）与高斯白噪声叠加（`awgn.py`、`noise.py`）
- 横向滤波器时域均衡，抽头系数求解（`xishu.py`）与卷积实现（`conv.py`）
- RLS 自适应均衡器（`rls_equalizer.py`），支持不同抽头数（7 / 31 / 99）对比
- 眼图绘制（`eye_image.py`）、误码率统计与 SNR-误码率曲线（`main.py`）
- 仿真结果数据保存于 `data/`，对比图保存于 `image/`

## 目录结构

```
.
├── main.py            # 主程序（完整仿真流程 + 绘图）
├── main.ipynb         # Jupyter 交互版本
├── rls_equalizer.py   # RLS 自适应均衡器
├── send.py / send_wuma.py  # 发送端（有/无误码场景）
├── source.py          # 符号序列源
├── isi.py             # 码间干扰信道
├── awgn.py / noise.py # 加性高斯白噪声
├── xishu.py           # 均衡器抽头系数求解
├── conv.py            # 卷积运算
├── sigexpand.py       # 信号扩展
├── eye_image.py       # 眼图绘制
├── receive.py         # 接收端
├── data/              # SNR-误码率仿真结果数据
└── image/             # 仿真结果图（眼图、波形、误码率曲线）
```

## 运行方式

```bash
python main.py
```

依赖：`numpy`、`matplotlib`（绘图使用中文字体 SimHei）。
