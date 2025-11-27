---
layout: default
title: MPQ-DM
permalink: /docs/MPQ-DM
---

* Table of Contents
{:toc}

![MPQ-DM](../images/MPQ-DM.png)

# 问题背景

对 diffusion models 进行及其低 bit 量化时 (2-4 bit)，涉及到对 activation 张量值的离散化. 然而由于低 bit 下离散化的精度极其低，基于 channels 的量化如果部分 channels 中存在异常值 outlier，那么这些 channels 的量化效果就会大打折扣. 并且离散化的特征也不便于 diffusion model 在不同时间步下稳定学习图像特征.

先前工作中均匀精度量化以及 layer-wise 的混精度量化无法解决 outliers in target weight channel 导致的精度问题，而 diffusion model 在多步扩散过程中会逐步累积 errors 使得模型表现效果大幅度下降.

# 方法创新
MPQ-DM 提出一种混合精度量化，

![Method](../images/MPQ-DM-method.png)