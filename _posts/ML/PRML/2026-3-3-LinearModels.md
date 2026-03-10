---
layout: post
title: Week 2 Linear Models 线性模型
date: 2026-3-3
permalink: /ML/PRML/LinearModels
toc: true
---

# Introduction

在 [统计机器学习 Week 2 - 统计决策理论](/ML/ProML/Decision) 中我们提到，监督学习的主要构成为**模型**、**训练数据**、**目标函数** (损失函数)，同时我们还需要设计一些学习或是**训练算法**以从**假设空间** $\mathcal{H}$ (Hypothesis Space) 中得到最优解。从数值分析中拟合一些函数遇到的问题可以得知，假设空间的合理性是模型训练成功很重要的一个点，而假设空间又与模型架构强相关，于是我们就需要尽可能合理化模型架构让假设空间中的函数尽可能保持以下好的性质：连续性、光滑性、简单且好处理。线性模型则是让假设空间保持以上好性质的最简单模型。

# 线性回归 Linear Regression
线性回归指代拟合单个自变量 $\bm{x} \in \mathbb{R}^d$ 与因变量 $\bm{y} \in \mathbb{R}$ 间线性关系的模型，常见的拟合式为

$$
    \hat{y}_i = \bm{w_0} + \bm{w_1^T x_i}, i = 1, 2, \cdots, N
$$

其中 $\bm{w\_0}$ 称作偏置 (bias)，$\bm{w\_1}$ 称作权重 (weight)。这也可以用齐次坐标将 bias 纳入 weight, 写为简单的矩阵乘法方式：

$$
    \hat{y}_i = \bm{w^T x_i}, i = 1, 2, \cdots, N
$$

选择 L2 Loss Function $l(f(x\_i), y\_i) = \frac{1}{N} \sum\_{j=1}^N (f(x\_i) - y\_i) ^ 2$，这可以带入上述预测值写为向量形式

$$
    l(f(x_i), y_i) = \frac{1}{N} \| \bm{Xw - y} \|^2 = \frac{1}{N} (bm{Xw - y})^T(\bm{Xw - y}) \\
    where \bm{X} = [\bm{x_i}^T], \bm{y} = [y_i]
$$

那么只需要对 $w^*$ 求偏导，并令其等于 0 即可得到最优解 $\bm{w}^*$ 满足的式子：

$$
    \bm{X}^T\bm{Xw}^* = \bm{X}^T\bm{y}
$$

然而 $\bm{X}^T\bm{X}$ 是半正定的，不一定具有逆，因而在工程中为保证其可逆性会让最优解变为以下形式

$$
    w^* = \left(\bm{X}^T\bm{X} + \lambda I)^{-1} \bm{X}^T \bm{y}
$$

其中 $\lambda$ 是一个合理的实数，这种操作被称为 **ridge**，这个结果被称为**岭回归** (Ridge Regression)。将这步操作反推会发现此时的损失函数变为

$$
    l^*(f(\bm{x}_i), y_i) = \underset{\min}{\bm{w}} \frac{1}{N}(\bm{Xw - y})^T(\bm{Xw - y}) + \frac{\lambda}{N}\bm{w}^T\bm{w}
$$

后一项 $\frac{\lambda}{N} \bm{w}^T \bm{w}$ 被称为正则项，而 ridge 的做法也被称为**正则化** (regularization)。因为机器学习最终目标是泛化预测，而不是训练数据拟合，因而正则化的操作是 reasonable 的。

# 线性判别分析 Linear Discriminant Analysis
线性判别分析是分类问题 (Classification)，对于不同类的数据，一个直觉是寻找一个投影方向，最大化不同类数据的分离度，最小化同类型数据的分离度，那么就可以实现不同类数据的分类。

先考虑二分类问题，这是最简单的分类问题。记投影操作为 $\hat{y}_i = \bm{w}^T\bm{x}_i$，为定量两类数据的分离度，直觉上最简单的做法是求两类数据投影的分离度：

$$
    \hat{m}_i = \frac{1}{N_i} \sum_{\bm{x}_j \in \mathcal{X}_i} \bm{w}^T \bm{x}_j , i = 1, 2
    \hat{S}_b(\bm{w}) = |\hat{m}_1 - \hat{m}_2|^2
$$

这可以最大化不同类数据的分离度，但还需要解决最小化同类型数据分离度的问题。Fisher 定义每一类数据的**散度** (scatter)

$$
    \hat{S}_i^2 = \sum_{y_j \in \mathcal{y}_i} (y_j - \hat{m}_i)^2, i = 1, 2
$$

投影方式 $\bm{w}$ 的总散度被定义为 $\hat{S}\_w = \sum\_{i} \hat{S}_i^2$. Fisher 将总目标函数定义为分离度和散度的比：

$$
    J_{F}(\bm{w}) = \frac{\hat{S}_b}{\hat{S}_w} = \frac{|\hat{m}_1 - \hat{m}_2 |^2}{\hat{S}_1^2 + \hat{S}_2^2}
$$

若定义 $m\_1, m\_2$ 为输入数据的中心点，那么分离度和散度都可以写为：

$$
    \hat{S}_b = \bm{w}^T (m_1 - m_2)(m_1 - m_2)^T w = \bm{w}^T \bm{S}_b \bm{w} \\
    \hat{S}_w = \bm{w}^T \left( \sum_{x_j \in \mathcal{X}_i} (x_j - m_i)(x_j - m_i)^T \right) w = \bm{w}^T \bm{S}_w \bm{w}
$$

因而目标函数就可以写为

$$
    J_F(\bm{w}) = \frac{\bm{w}^T \bm{S}_b \bm{w}}{\bm{w}^T \bm{S}_w \bm{w}}
$$

分子分母同时存在 $\bm{w}$ 不好优化，在处理这种优化问题时我们会加上约束，如令分母项为常数

$$
    \max \bm{w}^T \bm{S}_b \bm{w} \\
    \text{s.t. } \bm{w}^T \bm{S}_w \bm{w} = c \neq 0
$$

后续就可以用拉格朗日乗子法解出极值点的位置，拉格朗日函数为

$$
    L(\bm{w}, \lambda) = \bm{w}^T \bm{S}_b \bm{w} - \lambda (\bm{w}^T \bm{S}_w \bm{w} - c)
$$

对 $L(\bm{w}, \lambda)$ 求 $\bm{w}$ 偏导，可得到

$$
    \bm{S}_w^{-1}\bm{S}_b \bm{w}^* = \lambda \bm{w}^*
$$

注意到 $\bm{S} = (\bm{m}\_1 - \bm{m}\_2)(\bm{m}\_1 - \bm{m}\_2)^T$，那么最优解就可以写作

$$
    \lambda \bm{w}^* = \bm{S}_w^{-1}(\bm{m}_1 - \bm{m}_2)(\bm{m}_1 - \bm{m}_2)^T \bm{w}^*
$$

RHS 中 $(\bm{m}\_1 - \bm{m}\_2)^T\bm{w}^*$ 是标量，那么就可以令 $\lambda$ 与这个标量相等，就有

$$
    \bm{w}^* = \bm{S}_w^{-1}(\bm{m}_1 - \bm{m}_2)
$$

这就是最优解。得到投影方式的最优解后还需要确定的则是阈值 (thresholds)，即如何区分两类数据，这个阈值一般是取两类数据中心投影的中点，当然也可以做些工程适配，这个选取仍在研究中。

多类别问题也可同样定义不同类别的散度 (分离度) 和相同类别的散度，不过在目标函数选取上则变为迹 (Trace)。

# Logistic 回归 Logistic Regression
线性回归对于分类问题有几个重要的问题：
- 线性回归输出无界，在很多问题上较大范围的输出是无意义的
- 对异常值 (Outliers) 敏感

Logistic 回归则是将输入值 $x$ 统一映射至 $[0, 1]$ 内，一个常见的表达式为 Sigmoid Function:

$$
    P(y | x) = \sigmoid(s) = \frac{e^{w_0 + w_1x}}{1 + e^{w_0 + w_1x}}
$$

这种映射将输出空间变为概率空间，就可以用概率相关内容进行分析，或是分析概率相关问题。进一步可以定义几率 (odds) 为 $\frac{P}{1 - P} = e^{w\_0 + w\_1x}$，logits 为 $\text{logits} = \log (\text{odds}) = w\_0 + w\_1 x$。

[统计机器学习 Week 2 - 统计决策理论](/ML/ProML/Decision) 中较多分析如最大似然估计等均是建立在 logistic regression 映射得到的概率空间中，故机器学习与概率论、统计推断之间有密不可分的联系。当来到 K-分类问题时 logistic regression 的函数就变为 softmax function，

$$
    P(y = k | \bm{x}; \bm{W}) = \frac{e^{\bm{w}_k^T \bm{x}}}{\sum_{i = 1}^K e^{\bm{w}_i^T \bm{x}}}, \bm{x} \in \mathbb{R}^{d + 1}
$$

同样 K-分类也可以用最大似然方法求得最优解。

# 多层感知机 Perceptron
Perceptron 更贴近现代大模型的结构，其通过线性结构来完成各项任务。一个 Perceptron 的伪代码如下：

```python
init:
    all-zeroes weight vector w=0
    t = 1

Given x, check
    if w.t @ x > 0:
    else:
```

然而 Perceptron 最原始版本并不支持非线性分类，面对非线性问题的表现会急剧下滑，同时不能保证解的泛化性，因而后续便提出支持向量机 SVM，以解决解的泛化性问题。