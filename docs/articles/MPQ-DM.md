---
layout: default
title: MPQ-DM
permalink: /docs/MPQ-DM
---

* Table of Contents
{:toc}

![MPQ-DM](../images/MPQ-DM.png)

# 问题背景

## 理论分析
对 diffusion models 进行及其低 bit 量化时 (2-4 bit)，涉及到对 activation 张量值的**离散化**. 然而由于低 bit 下离散化的精度极其低，基于 channels 的量化如果部分 channels 中存在**异常值 outlier**，那么这些 channels 的量化效果就会大打折扣. 并且离散化的特征也不便于 diffusion model 在不同时间步下稳定学习图像特征.

## 相关工作

针对量化过程出现的 outliers，一个可行的方向是 **Smooth 策略**. 由于在 forward 过程中 activation 的 outlier 显著多于 weight 的 outlier，于是可以按一定比例缩小 activation，再同比放大 weight 的策略削弱 outlier 影响. [**\[ICML 2023 poster\]SmoothQuant**](https://arxiv.org/abs/2211.10438) 是其中的代表性工作，在 transformer 架构中可以取得较好的效果.

![smoothquant](../images/smoothexp.png)

另一个可行的方向则是**混合精度量化**，对 outlier 影响较大的张量做较高 bit 量化，另外张量做低 bit 量化. 但由于 outlier 总是分布在 activation **特定 channel** 中(见 SmoothQuant 原理图)，均匀精度量化以及 layer-wise 的混精度量化策略无法解决 outliers in target weight channel 导致的精度问题，量化后 error 较大. diffusion model 在多步扩散过程中会逐步累积 errors，使得量化后模型表现效果大幅度下降.

# 论文方法创新
MPQ-DM 提出一种 **channel-wise** 的混合精度量化方法，

![Method](../images/MPQ-DM-method.png)