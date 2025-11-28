---
layout: default
title: MPQ-DM
permalink: /docs/MPQ-DM
---

* Table of Contents
{:toc}

![MPQ-DM](../images/MPQ-DM.png)

# 问题背景

对大模型的量化 (quantization) 主流为两种策略，**训练后量化** (Post-training quantization, PTQ) 和**量化感知训练** (Quantization-aware training, QAT). PTQ 不需要额外的训练，只需要在预训练模型上直接量化或利用 calibration data 做特定粒度的量化操作. QAT 则是 fine-tune 量化处理后的预训练模型，训练开销较大. 

在极低量化精度 (bit-width) 下，PTQ-based 的模型表现效果会严重下降，而 QAT-based 的模型表现下降较低. 考虑到训练成本较大，QAT 虽然效果较好但是训练压力太大，因而优化 PTQ-based 的方法在极低精度下的表现成为量化的一个热门研究方向.

## 理论分析

对 diffusion models 进行及其低 bit 量化时 (2-4 bit)，涉及到对 activation 张量值的**离散化**，如下所示

$$
    \bm{x}_q = \mathrm{clip}\left( \lfloor \frac{\bm{x}_f}{s} \rceil + z, 0, 2^N - 1 \right)
$$

$\bm{x}_f$ 为量化前的浮点数张量，在代码实现中类型一般为 `torch.float32`.  $\lfloor \cdot \rceil$ 表示四舍五入取整，$s = \frac{x\_{\max} - x\_{\min}}{2^N-1}$ 为量化缩放比例 (scale)，$z = - \lfloor \frac{x\_{\min}}{s} \rceil$ 为偏置项调整缩放零点. $N$ 为量化 bit 数，$\mathrm{clip}$ 表示截断操作，将超出范围的数值截断在 $[0, 2^N -1]$ 中. 

**量化操作将数据近似线性地从 $[\mathrm{min\_{float}}, \mathrm{max\_{float}}]$ 映射到量化后的范围 $[0, 2^N - 1]$**. 可以预测到当两个数据误差在一定范围内时，由于四舍五入的精度问题，这个映射会将二者映射到同一值. 

```python
import torch

def quantize_tensor(x, n_bits=4):
    x_max = x.amax(dim=-1, keepdim=True)
    x_min = x.amin(dim=-1, keepdim=True)

    q_max = 2**n_bits - 1
    q_min = 0
    scales = (x_max-x_min).clamp(min=1e-5) / (q_max - q_min)
    base = torch.round(-x_min/scales).clamp_(min=q_min, max=q_max)
    x_q = (torch.clamp(torch.round(x / scales) + base, q_min, q_max) - base) * scales 
    # 这里并入量化后数据参与计算前的复原操作，方便比较原始张量与量化后复原张量的差异.
    return x_q

x_f1 = torch.tensor([0.1, -0.4, 0.3, 0.8, -0.2], dtype=torch.float32)
x_f2 = torch.tensor([0.1, -0.4, 0.3, 0.8, -8], dtype=torch.float32)
x_q1 = quantize_tensor(x_f1)
x_q2 = quantize_tensor(x_f2)
print(f'x_q1: {x_q1}\nx_q2: {x_q2}')
# x_q1: tensor([ 0.0800, -0.4000,  0.3200,  0.8000, -0.1600])
# x_q2: tensor([ 0.0000, -0.5867,  0.5867,  0.5867, -8.2133])
```

低 bit 下离散化的精度极其低，基于 channels 的量化如果部分 channels 中存在**异常值 outlier**，那么这些 channels 的量化效果就会大打折扣. 并且离散化的特征也不便于 diffusion model 在不同时间步下稳定学习图像特征.

## 相关工作

针对量化过程出现的 outliers，一个可行的方向是 **Smooth 策略**. 由于在 forward 过程中 activation 的 outlier 显著多于 weight 的 outlier，于是可以按一定比例缩小 activation，再同比放大 weight 的策略削弱 outlier 影响. [**\[ICML 2023 poster\]SmoothQuant**](https://arxiv.org/abs/2211.10438) 是其中的代表性工作，在 transformer 架构中可以取得较好的效果.

![smoothquant](../images/smoothexp.png)

另一个可行的方向则是**混合精度量化**，对 outlier 影响较大的张量做较高 bit 量化，另外张量做低 bit 量化. 但由于 outlier 总是分布在 activation **特定 channel** 中(见 SmoothQuant 原理图)，均匀精度量化以及 layer-wise 的混精度量化策略无法解决 outliers in target weight channel 导致的精度问题，量化后 error 较大. diffusion model 在多步扩散过程中会逐步累积 errors，使得量化后模型表现效果大幅度下降. MPQ-DM 即在这条赛道上开展的工作.

# 论文方法创新
MPQ-DM 提出一种 **channel-wise** 的混合精度量化方法，主要分为两个部分，**基于异常值的混合精度量化** (Outlier Driven Mixed Quantization, OMD) 和 **时间平滑相关性蒸馏** (Time Smoothed Relation Distillation, TRD). 

![Method](../images/MPQ-DM-method.png)

