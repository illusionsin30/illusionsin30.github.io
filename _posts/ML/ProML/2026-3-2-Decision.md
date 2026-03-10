---
layout: post
title: Week 2 Statistical Decision Theory 统计决策理论
date: 2026-3-2
permalink: /ML/ProML/Decision
toc: true
---

# 监督学习概述

在统计机器学习中，所谓的“学习”通常分为以下三类

|机器学习方式|介绍|
|:---:|:---:|
|监督学习 Supervised Learning|给定**特征** (features) 和**标签** (label) 对 $(x, y)$，让机器学习其中对应规律|
|无监督学习 Unsupervised Learning|仅给定**特征** (features)，不对机器做标签化指导|
|强化学习 Reinforcement Learning|构建学习系统，让机器agent基于当前**状态** (state) 和环境 (environment) 给出**行为** (action)，赋予其奖励或惩罚 (reward)|

监督学习目前普遍应用在模型预训练 (Pretrain) 层面，是机器学习最表层也是其次重要的部分。监督学习可以用以下数学模型进行建模：

$$
    \text{Given data} \mathcal{D} = {(X_i, Y_i)}_{i=1}^N, \text{learn a mapping} f: \mathcal{X} \to \mathcal{Y} \text{from} X \in \mathcal{X}, Y \in \mathcal{Y}
$$

其中 $X, Y$ 分别被称作**特征** (features) 和**标签** (label)，$\mathcal{D}$ 为**训练数据** (training data)，$N$ 为**样本数量** (sample size)，$(X_i, Y_i)$ 也被称作**样本** (sample)。从这个建模中可以看出 $f$ 就是监督学习过程中机器的作用：用训练数据拟合得到数据分布 $\mathcal{X}, \mathcal{Y}$ 之间的映射关系。

类似于数理统计中的情况，样本数据会分为很多种类型，因而机器学习对不同数据的拟合有不同的称呼。对于数值型的数据，学习过程被称为**回归** (Regression)，而对于类别型的数据，学习过程被称为**分类** (Classification)。下面尝试从数学层面建立相应的模型。

# 回归 Regression
## M-estimator 和估计一致性
回顾数理统计中对样本分布的拟合，对于给定样本 $\mathcal{D} = \\{(X_i, Y_i)\\}\_{i=1}^N$，设 $Y\_i \approx f(X\_i;\theta)$，$X\_i$ i.i.d.，有以残差平方为最小化损失函数的最小二乘估计 (OLS-estimator)：

$$
    \hat{\theta}_n = \underset{\theta}{\argmin} \sum_{i=1}^N (Y_i - f(X_i;\theta)) \overset{P}{\to} \underset{\theta}{\argmin} \sum_{i=1}^N (Y - f(X;\theta))
$$

类似于 OLS-estimator，对于回归问题可以提出更一般的渐进统计估计方式，估计量称作 **M-estimator**：

> **Definition** (M-estimator): 给定函数 $m(X\_i; \theta)$，估计量 $\hat{\theta}\_n$ 定义为让以下表达式取最大值时的 $\theta$：
>
> $$
>   M_n(\theta) = \frac{1}{n} \sum_{i=1}^n m(X_i; \theta)
> $$
>
> 若存在解，估计量可以简单写作
>
> $$
>   \hat{\theta}_n = \underset{\theta \in \Theta}{\argmax} M_n(\theta)
> $$

在估计中大样本下**估计的一致性**是一个很重要的考量点，若对于任何真实值 $\theta\_0 \in \Theta$，估计量 $\hat{\theta}\_n \overset{P}{\to} \theta\_0$，则称 $\hat{\theta}\_n$ 具有一致性。van der Vaart 对于 M-estimator 和估计的一致性给出以下定理：

> **Theorem**: 给定确定函数 $M$ (fixed function, 不随样本选取改变) 和随机函数 $M\_n$ (random function, 意为随样本选取随机性发生变化的函数)，若 $\forall \epsilon > 0$，有
>
> $$
>   \underset{\theta \in \Theta}{\sup} |M_n(\theta) - M(\theta)| \overset{P}{\to} 0, \hspace{2ex} \underset{\{\theta | d(\theta, \theta_0) \geq \epsilon\}} M(\theta) < M(\theta_0)
> $$
>
> 则对任意满足 $M\_n(\hat{\theta}\_n) \geq M\_n(\theta\_0) - o_p(1)$ 的估计量 $\hat{\theta}\_n$ 依概率收敛到 $\theta\_0$。

van der Vaart 定理前一条件要求与样本有关的函数 $M\_n$ 依概率一致收敛于确定函数 $M$，后一条件要求确定函数 $M$ 仅会在 $M(\theta\_0)$ 处取到最大值，有这两个条件就可以忽略估计量的形式，只需要保证我们优化的这个目标函数 $M$ 性质足够好，我们就可以找到一致的最优解。

## $L_2$ 损失函数
依据 van der Vaart 定理，我们只需要关心如何定义特征与标签之间的**损失函数** $L(Y, f(X))$，即可得到对 Y 最好的估计 $f$. 在机器学习中，上述 OLS 给出的损失函数

$$
    L(Y, f(X)) = (Y - f(X))^2
$$

被称为 **$L\_2$ Loss Function**，其对应的最优解为 $f\_0(X) = \underset{f}{\argmin} E[(Y - f(X))^2]$。若已知 $(X_i, Y_i)$ 的分布函数，可将最优解写为 $X, Y$ 的形式，先处理损失函数的期望表达式：

$$
    E[(Y - f(X))^2] = E\left\{ E \left[ (Y - f(X))^2 | X \right] \right\}
$$

从期望表达式容易得到，只要让 $E[(Y - f(X))^2 \| X]$ 在每个给定的 $X = x$ 处最小即可得到最小期望，那么最优解就变为

$$
\begin{align*}
    f\_0(x) &= \underset{f}{\argmin} E[(Y - f(X))^2 | X = x] \\
    &= \underset{f}{\argmin} f(x)^2 - 2f(x)E(Y | X = x) + E(Y^2 | X = x) \\
    &= \underset{f}{\argmin} [f(x) - E(Y | X = x)]^2 
\end{align*}
$$

于是在 $L\_2$ loss 下最优解为 $f\_0(X) = E(Y \| X)$，因而若在学习过程中选定 $L\_2$ 损失函数，那么学习过程就是依训练数据 $\mathcal{D}$ 拟合 $f\_0(x) = E[Y \| X = x]$ 的过程。

## $L_1$ 损失函数
在机器学习中定义 $L_1$ Loss Function 为以下形式：

$$
    L(Y, f(X)) = |Y - f(X)|
$$

类似 $L_2$ 寻找最优解的方式，我们也可以类似地找出其最优解，同样先对损失函数期望做处理：

$$
    E[|Y - f(X)|] = E \left\{ E \left[ |Y - f(X)| | X \right] \right\}
$$

再令 $E[\|Y - f(X)\| \| X]$ 在每一点 $X = x$ 处最小，得到最优解

$$
\begin{align*}
    f\_0(X) &= \underset{f}{\argmin} E[|Y - f(X)| | X = x] \\
    &= \underset{f}{\argmin} \int_{-\infty}^{f(x)} (f(x) - y)p_{Y|X}(y | x) dy + \int_{f(x)}^{\infty} (y - f(x))p_{Y|X}(y | x) dy
\end{align*}
$$

积分式对 $f(x)$ 求偏导,

$$
\begin{align*}
    &\frac{\partial}{\partial (f(x))} \left( \int_{-\infty}^{f(x)} (f(x) - y)p_{Y|X}(y | x) dy + \int_{f(x)}^{\infty} (y - f(x))p_{Y|X}(y | x) dy \right) \\
    =& 2 \int_{-\infty}^{f(X)} p_{Y | X}(y | x) dy - 1 \\
    =& 0
\end{align*}
$$

得到 $P(Y < f(X)) = \frac{1}{2}$，即 $f\_0(X) = \text{Median}(Y)$，$L_1$ loss 的最优解是 $Y$ 的中位数。

## 偏差 & 方差分解 Bias-Variance Decomposition
实际采样中认为 $Y$ 总是存在噪声误差 $\varepsilon$，这被称为无法消除的误差 (Irreducible Error)，那么实际上 $Y = f(X) + \varepsilon$。因而监督学习过程中拟合 的 $\hat{f}\_{\mathcal{D}}$ 存在两大随机性来源：噪声误差与采样误差。用数学语言表述则是

$$
\begin{align*}
    &E[(Y - \hat{f}_{\mathcal{D}}(X))^2] \\
    =& E[(f(X) + \varepsilon - E[\hat{f}_\mathcal{D}(X)] + E[\hat{f}_\mathcal{D}(X)] - \hat{f}_{\mathcal{D}}(X))^2] \\
    &\text{(代入模型的期望预测，一加一减)}\\
    =& E(\varepsilon^2) + E\left[(f - E(f_{\mathcal{D}}))^2\right] + E\left[(E(\hat{f}_{\mathcal{D}}) - f_{\mathcal{D}})^2\right] \\
    &(\text{噪声与 } f(X), \hat{f}_{\mathcal{D}}(X) \text{ 独立，直接分离, 后一项交叉项为 0}) \\
    =& Var(\varepsilon) + Bias(\hat{f}_\mathcal{D}(X))^2 + Var(\hat{f}_{\mathcal{D}}(X))
\end{align*}
$$

因而监督学习过程中模型的误差总可以表示为

$$
    \mathrm{Error} = Var(\varepsilon) + Bias(\hat{f}_{\mathcal{D}}(X))^2 + Var(\hat{f}_{\mathcal{D}}(X))
$$

模型复杂度越高，其对训练数据的拟合程度越高，bias 相对会低一些，但是拟合结果的方差会变大，整体呈现训练集 error 较低，但是测试集 error 较高的情况，这种学习情况称为**过拟合** (Overfitting)。而模型复杂度较低时其对训练数据的拟合程度不足，bias 相对较高，方差相对较低，呈现出训练集和测试集 error 均较高的情况，这种学习情况称为**欠拟合** (Underfitting)。

# 分类 Classification
分类任务一般是设置好类别数量 $K$，让模型直接预测类别 $f(x)$，或是预测类别概率 $f(y \| x)$。预测类别则是基于 0-1 Loss Function 给出 $f(x) = \underset{k}{\argmax} P(Y=k \| X=x)$，这和回归问题类似，这种分类被称为 Bayes Classifier，期望损失被称为 Bayes rate。而预测概率在统计学中则是用极大似然估计的方法预测，这推导出的结果与回归结果极其类似，最佳映射是条件概率。对于对数最大似然函数

$$
    l = \sum_{k=1}^K p_k \log q_k
$$

其中 $p\_k = P(Y=k \| X=x)$ 表示物品为 $k$ 类别的条件概率，$q\_k$ 为模型预测 $k$ 类别的概率。依据 Lagrange 乗子法，可以定义

$$
    L(q, \lambda) = \sum_{k=1}^K p_k \log q_k + \lambda(\sum_{k=1}^K q_k - 1)
$$

容易得到 $q\_k = p\_k$ 即是 $l$ 取最小值时 $q\_k$ 的分布。而机器学习 (或是信息论) 中有一个概念为**交叉熵损失** (Cross Entropy Loss)，其与最大似然在本质上一致。先定义熵 (Entropy):

$$
    H(p) = -\sum_y p(y) \log_{2} p(y) = E(-\log_2 p(y))
$$

熵可以被解释为不确定性的衡量，也可以被解释为二进制下编码事件信息的最小期望编码。交叉熵的定义则从熵出发得到：

$$
    H(p, q) = - \sum_{y} p(y) \log_2 q(y) = E_{p(Y)}[-\log_2 q(y)]
$$

这则是说在分布 $P$ 下用分布 $Q$ 的最优事件编码方式来编码的期望长度。可以从数学上证明，极大似然等价于最小化交叉熵，因而机器学习中常常用交叉熵损失函数来优化模型表现。