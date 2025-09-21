# Bridging Supervised Learning and Reinforcement Learning in Math Reasoning

## 摘要
在这项工作中，我们挑战了 "自我改进专属 RL" 的普遍观点，提出了负样本感知微调（NFT）—— 一种监督学习方法，使 LLM 能够在没有外部教师的情况下反思失败并自主改进。
在在线训练中，NFT 不会丢弃模型生成的错误答案，而是构建一个隐式负策略来对其进行建模。这个隐式策略与我们用于优化正样本的 LLM 共享参数，从而能够对模型的所有生成结果进行直接策略优化。
我们在数学推理任务上对 7B 和 32B 规模的模型进行了实验。结果一致表明，通过额外利用负反馈，NFT 显著优于拒绝采样微调等 SL 基线方法，并能匹配甚至超越 GRPO 和 DAPO 等领先 RL 算法。
此外，我们证明了尽管源自完全不同的理论基础，NFT 和 GRPO 在严格的 on-policy 训练中实际上是等价的。

“on-policy”（同策略）是强化学习（Reinforcement Learning, RL）中的核心概念之一，核心定义是策略学习与经验收集依赖于同一份当前策略，即 “用谁学，就用谁采”。

## 介绍
强化学习 (RL) 似乎天然适合此类基于验证的训练。PPO [40] 和 GRPO [41] 等算法专为最大化奖励信号设计，可方便地采用二元验证结果作为奖励。
相比之下，监督学习 (SL) 很少被视为实现自我改进的途径。普遍观点认为 SL 本质上是通过记忆正向训练数据来模仿外部教师，不适合从负面错误中进行自我反思学习 [11]。
在这项工作中，我们挑战了 "自我改进专属 RL" 这一主流观点，并展示了它同样可以在监督学习范式内实现。
我们从一个简单的 SL 基线开始：拒绝采样微调 (RFT)[62, 15]。在每次迭代中，LLM 生成问题答案，验证器剔除所有错误答案，剩余正确答案被汇编成数据集用于监督微调模型本身。
RFT 已被多种研究证明有效 [3, 32, 59, 43, 46, 53]。然而，它完全无法从负面反馈中学习，鼓励模型强化已有优势，而非反思错误 —— 我们认为这种能力对于实现通用智能至关重要。
为克服这一局限，我们提出负样本感知微调 (NFT)，一种能让 LLM 从负面生成中学习的在线学习算法。与 RFT 类似，NFT 通过监督在正确答案上微调正向 LLM。关键在于，NFT 不丢弃错误答案，而是构建一个隐式负策略来对其建模。
这个隐式策略与我们用于优化正向数据的 LLM 共享参数，使得能够对模型所有生成进行直接策略优化 (图 2)。NFT 内存开销极小，训练过程中仅需维护单个模型。

关键发现
1. 仅靠监督学习即可显著增强 LLM 的数学推理能力，无需外部教师。NFT 匹配甚至超越了 GRPO [41] 和 DAPO [58] 等先进 RL 算法
2. 在线训练中 SL 与 RL 的性能差距，主要源于 SL 过去无法利用负面反馈，而非 RL 本身具有内在优势

## 背景

### 2.1 最大似然与策略梯度

监督学习（SL）和强化学习（RL）是两种不同的机器学习范式，它们在目标和优化方法上存在根本差异。

#### 监督学习 (Supervised Learning)
- **目标**：学习一个模型 `π_θ(a | q)` 来拟合数据分布 `π(a | q)`
- **方法**：最大化对数似然函数：
  $
  \max_{\theta} \mathbb{E}_{a \sim \pi(a|q)} \log \pi_{\theta}(a | q)
  $
  等价于最小化KL散度 `D_KL[π(a|q) || π_θ(a|q)]`
- **数据需求**：需要带有正确答案的数据集 `D = {q, a ~ π(a|q)}`
- **LLM应用**：q 是问题提示，a 是答案

#### 强化学习 (Reinforcement Learning)
- **目标**：最大化预定义的奖励 `r(q, a)`
- **方法**：最大化奖励期望：
  $
  \max_{\theta} J(θ) := \mathbb{E}_{a \sim \pi_θ(a|q)} r(q, a)
  $
- **优化挑战**：直接对J(θ)求导困难，需使用策略梯度估计：
  $
  \nabla_{\theta} J(θ) = \mathbb{E}_{a \sim \pi_θ(a|q)} \nabla_{\theta} [ r(q,a) \log \pi_θ(a|q) ]
  $
- **LLM应用**：在语言推理中，a 可视为每步的token决策，r(q,a)可用优势函数A(q,a)替代

### 2.2 数学推理中的强化学习：从策略梯度到GRPO

在数学推理等序列决策问题中，基础的策略梯度方法存在局限性，研究者们提出了多种改进算法。

#### 从On-Policy到Off-Policy
- **On-Policy限制**：策略梯度要求数据必须由当前策略生成，更新后数据即失效
- **重要性采样**：通过引入旧策略 `π_old` 打破这一限制：

  $ \nabla_{\theta} J(θ) = \mathbb{E}_{a \sim \pi_{old}(a|q)} [ r(q,a) \nabla_{\theta} R_θ(q,a) ] $

  其中 `R_θ(q,a)` 是新旧策略的似然比

#### GRPO算法
GRPO是PPO的改进版，通过梯度裁剪进一步约束策略更新：
- **损失函数**：
  $
  \mathcal{L}^{GRPO}(\theta) = -\sum_{q,a \sim \pi_{old}} \sum_{t} \min\left[ R_{\theta}^{t}(q,a) \hat{A}_{q,a}, \text{clip}(R_{\theta}^{t}(q,a), 1-\epsilon', 1+\epsilon') \hat{A}_{q,a} \right]
  $
- **优势估计**：
  $
  \hat{A}_{q,a} := \frac{r(q,a) - \text{mean}\{r^{1:K}\}}{\text{std}\{r^{1:K}\}}
  $
  研究表明，有时可以去掉标准差归一化，仅保留均值归一化

## 3 方法
### 3.1 问题设定
#### 数据集
给定一组包含N个数学问题的集合$\{q^{1:N}\}$、一个预训练大语言模型$\pi(a | q)$（其中$a$表示答案，$q$表示问题），以及一个用于判断答案正确性的验证器。在每次迭代中，我们生成一个服从策略$\pi$分布的数据集$D=\{q, a^{1:K} \sim \pi, r^{1:K}\}^{1:N}$，其中$r \in \{0,1\}$为正确性标签（1表示正确，0表示错误），$K$为每个问题对应的生成答案数量。服从策略$\pi$分布的数据集$D$可划分为两个子集：$D^{+}$和$D^{-}$，其中$D^{+}$包含所有正确答案，$D^{-}$包含其余错误答案。我们将$D^{+}$的潜在答案分布记为$\pi^{+}(\cdot | q)$。

#### 学习目标
我们希望将原始策略$\pi$优化为新的正策略$\pi_{\theta}^{+} \approx \pi^{+}$。目标正策略$\pi^{+}(a | q)$可利用贝叶斯规则形式化表示为：
$
\pi^{+}(a | q):=\pi(a | q, r=1)=\frac{\pi(a | q) p(r=1 | q, a)}{\sum_{A} \pi(a | q) p(r=1 | q, a)},
$
其中$A$代表答案$a$所属的所有可能语言空间。

#### 讨论
学习$\pi^{+}$的一个直观方案是仅利用正确答案子集$D^{+}$进行微调，并完全丢弃错误答案子集$D^{-}$（即拒绝采样微调（RFT）[15, 53]）。然而，这种方法会导致模型无法从负反馈（即$D^{-}$）中学习。我们认为，从自身错误中进行反思的能力不仅是理想的特性，更是实现通用智能的核心，它标志着模型学习模式从单纯模仿向自我反思学习的转变。尽管传统观点认为这种能力是强化学习（RL）的独特优势[11, 63]，但我们提出疑问：在监督学习（SL）范式下，是否同样能实现这种自我反思式的改进？

### 3.2 利用负答案直接优化语言模型
在本节中，我们将探讨如何利用负样本集$D^{-}$直接优化目标正策略$\pi_{\theta}^{+}$。尽管初看似乎不可能，但我们发现目标正策略$\pi^{+}$与负策略$\pi^{-}$存在紧密关联，这使得直接利用$D^{-}$训练$\pi_{\theta}^{+}$成为可能。

首先，我们参照式（5）（指原文中的公式编号，下同）对负策略$\pi^{-}$进行形式化定义：
$
\pi^{-}(a | q):=\pi(a | q, r=0)=\frac{\pi(a | q)[1-p(r=1 | q, a)]}{\sum_{A} \pi(a | q)[1-p(r=1 | q, a)]}.
$

结合式（5）与式（6），我们得到一个关键发现：
$
r_{q} \pi^{+}(a | q)+\left[1-r_{q}\right] \pi^{-}(a | q)=\pi(a | q), \quad (7)
$
其中$r_{q}:=\sum_{A} \pi(a | q) p(r=1 | q, a)=p(r=1 | q)$表示大语言模型$\pi$针对问题$q$的正确率。在实际场景中，$r_{q}$可通过数据集$D$中$K$个蒙特卡洛奖励的均值（即$r_{q} \approx \text{mean}\{r^{1:K}\}$）进行估算。

#### 隐式负策略
式（7）揭示了$\pi^{+}$与$\pi^{-}$之间的紧密关联（见图3）。鉴于我们已拥有预训练大语言模型$\pi$，且$r_{q}$可估算，理论上，从负样本中学习$\pi^{-}$应能以类似于从正样本中进行监督学习的方式，对目标正策略$\pi_{\theta}^{+}$产生影响。

为实现这一思路，我们利用式（7）中的关系，通过目标正策略$\pi_{\theta}^{+}$对隐式负策略（记为$\pi_{\theta}^{-}$）进行参数化，具体形式如下：
$
\pi_{\theta}^{-}(a | q):=\frac{\pi(a | q)-r_{q} \pi_{\theta}^{+}(a | q)}{1-r_{q}}.
$
因此，在负答案上训练$\pi_{\theta}^{-}$，将直接实现对底层正策略$\pi_{\theta}^{+}$的优化（见图2）。这一过程具有以下理论保障：

#### 定理3.1（基于负答案的策略优化）
考虑针对隐式负策略$\pi_{\theta}^{-}$训练的最大似然目标函数：
$
\max_{\theta} \mathbb{E}_{p(q) \pi^{-}(a | q)}\left[ \log \pi_{\theta}^{-}(a | q)\right] \Leftrightarrow \min_{\theta}\left[-\mathbb{E}_{(q, a) \sim \mathcal{D}^{-}} \log \frac {\pi(a | q)-r_{q} \pi_{\theta}^{+}(a | q)}{1-r_{q}}\right] \quad (8)
$
在数据量无限且模型容量足够的前提下，求解式（8）的最优解满足：
$
\forall q,a: \pi _{\theta }^{+}(a|q)=\pi ^{+}(a|q)
$
（证明过程见附录A）。定理3.1证明了仅利用负样本进行策略优化的可行性。为进一步利用正样本，我们还在$D^{+}$上进行监督训练，最终得到NFT的损失函数：
$
\mathcal{L}_{(a, q, r) \sim \mathcal{D}}^{NFT}(\theta)=r\left[-\log \frac{\pi_{\theta}^{+}(a | q)}{\pi(a | q)}\right]+(1-r)\left[-\log \frac{1-r_{q} \frac{\pi_{\theta}^{+}(a | q)}{\pi(a | q)}}{1-r_{q}}\right]
$

需要说明的是，我们在损失函数中减去了基准项$-\log \pi(a | q)$。由于该基准项与参数$\theta$无关，因此不会对损失梯度及最优解产生影响。其中$\pi(a | q)$表示优化器更新前的旧策略似然。在训练初始阶段，$\pi_{\theta}^{+}=\pi$，此时$\mathcal{L}_{(a, q, r)}^{\theta}=0$。

我们将该方法命名为“负样本感知微调（Negative-aware Fine-Tuning, NFT）”，原因在于与RFT相比，它能额外利用负样本进行策略优化。

NFT具有较高的内存效率：在实际训练过程中，我们只需在内存中维护一个模型副本。旧策略似然$\pi(a | q)$可在数据生成阶段预先计算。

### 3.3 实用算法
我们对式（9）进行了多项改进，提出了NFT的实用目标函数：
$
\mathcal{L}_{\mathcal{D}}^{NFT}(\theta)=-\sum_{q, a, r} \omega(q) \sum_{t}\left[r \log R_{\theta}^{t}(q, a)+(1-r) \log \max\_v\left(\frac{1-\hat{r}_{q} R_{\theta}^{t}(q, a)}{1-\hat{r}_{q}}, \epsilon\right)\right] \quad (10)
$
其中$R_{\theta}^{t}(q, a)=\frac{\pi_{\theta}^{+}\left(a_{t} | q, a_{<t}\right)}{\pi\left(a_{t} | q, a_{<t}\right)}$，$\hat{r}_{q}=\frac{1}{K} \sum_{a | q} r(q, a)$。

NFT的伪代码如算法1所示。下文将对其中的关键设计选择进行说明：

#### Token级损失
式（9）本质上处理的是序列数据，而答案似然$\pi(a | q)=\prod_{t} \pi(a_{t} | q, a_{<t})$与答案长度高度相关，这会导致梯度估算的方差增大，并在训练过程中引发数值不稳定问题。借鉴现有研究[40, 31, 58]的思路，我们将每个token的决策视为独立单元，并在式（10）中对所有token的损失进行求和。

#### 负似然比裁剪
式（10）中负损失的计算涉及对数运算，其参数必须为正，这就要求$\frac{1-\hat{r}_{q} R_{\theta}^{t}}{1-\hat{r}_{q}}>0$。在$R_{\theta}^{t}$未优化时，该条件可能无法满足，进而导致训练崩溃。因此，我们为负似然比设置了一个最小值$\epsilon>0$。为在裁剪后仍能保留梯度流，我们进一步采用了直通梯度估算（straight-through gradient estimation）[4, 47]，具体实现细节见算法1。

#### Prompt加权
为使训练聚焦于更具信息价值的样本，我们为每个问题prompt $q$分配了权重$\omega(q)$，并对正确率$r_{q}$较低的“难题”赋予更高的重要性（相关消融实验见5.4节）。这一设计还有助于使NFT与GRPO等强化学习算法保持一致性，具体细节将在第4节中讨论。


#### 算法1 负样本感知微调（NFT）
1. **输入**：语言模型$\pi$、prompt集合$q^{1:N}$、验证器$r(\cdot)$。
2. **定义函数**：`max_v(x, ϵ)`：
   3. **返回**：`stop_gradient[max(x, ϵ) − x] + x`  // 保留梯度的裁剪操作（直通最大值算子）
4. **迭代训练**：
   5.  // 数据采集阶段
   6.  对每个采样的prompt $q$：
      - 生成$K$个答案$a^{1:K}$，并验证其正确性$r^{1:K}$
      - 计算正确率$\hat{r}_q = \text{mean}\{r^{1:K}\}$，以及token级似然$\{\pi(a_t|q, a_{<t})\}_{1:|a|}^{1:K}$
      - 若$0 < \hat{r}_q < 1$（prompt筛选），则将$\{q, \hat{r}_q, a^{1:K}, r^{1:K}, \pi_{1:K}^t\}$加入数据集$D$
   7.  结束对prompt $q$的处理
   8.  初始化$\pi_{\theta}^{+} \leftarrow \pi$
   9.  // 最大似然训练阶段
   10. 对数据集中的每个mini-batch $\{q, a, r, \hat{r}_q, \pi_t\}$：
        - 计算正似然比：$R_{\theta}^t(q, a) = \frac{\pi_{\theta}^{+}(a_t|q,a_{<t})}{\pi(a_t|q,a_{<t})}, \forall t$
        - 若$r = 0$（负样本）：
            * 计算隐式负似然比：$R_{\theta}^t(q, a) = \frac{1 - \hat{r}_q R_{\theta}^t(q, a)}{1 - \hat{r}_q}$
            * 裁剪负似然比：$R_{\theta}^t(q, a) = \text{max}_v[R_{\theta}^t(q, a), \epsilon]$
        - 参数更新：$\theta \leftarrow \theta + \lambda \nabla_{\theta} \sum_t \log R_{\theta}^t(q, a)$（式10）
   11. 结束对mini-batch的处理
12. 更新模型：$\pi \leftarrow \pi_{\theta}^{+}$，进入下一轮迭代
13. **结束迭代**

## 4 理解NFT与GRPO之间的差异
尽管NFT（负样本感知微调）与GRPO（梯度裁剪策略优化）源自完全不同的理论框架，但二者存在显著的相似性。值得注意的是，我们发现**在在线策略（on-policy）训练场景下，NFT与GRPO完全等价**。为厘清这一关联，我们将通过计算与对比二者的损失梯度展开分析：


### 4.1 命题4.1（算法梯度比较）
假设对于某一问题$q$，存在$\hat{r}_{q} K$个正答案与$(1-\hat{r}_{q}) K$个负答案（其中$\hat{r}_{q}$为该问题的估算正确率，$K$为每个问题的生成答案总数）：

#### （a）GRPO的梯度
考虑式（3）中仅包含{0,1}二进制奖励的情况，GRPO的损失梯度满足：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{GRPO}(\theta)=-\sum\left\{r A_{q}^{+} \cdot \mathcal{I}\left[R_{\theta}^{t}(q, a)<1+\epsilon'\right]+(1-r) A_{q}^{-} \cdot \mathcal{I}\left[R_{\theta}^{t}(q, a)>1-\epsilon'\right]\right\} \nabla_{\theta} R_{\theta}^{t}(q, a)
$
其中，$A_{q}^{+}=\sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}}$与$A_{q}^{-}=-\sqrt{\frac{\hat{r}_{q}}{1-\hat{r}_{q}}}$分别为正、负答案的归一化优势值（normalized advantages），$\mathcal{I}[\cdot]$为指示函数（满足括号内条件时取值1，否则取值0），$R_{\theta}^{t}(q, a)$为$t$时刻的似然比（likelihood ratio），$\epsilon'$为GRPO的梯度裁剪参数。

#### （b）NFT的梯度
若设置prompt权重$\omega(q)=\sqrt{(1-\hat{r}_{q}) / \hat{r}_{q}}$，则NFT的损失梯度满足：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{NFT}(\theta)=-\sum \bigg\{r A_{q}^{+} \cdot \frac {1}{R_{\theta }^{t}(q, a)}+(1-r)A_{q}^{-}\cdot max \bigg [\frac {1-\hat{r}_{q} R_{\theta }^{t}(q, a)}{1-\hat{r}_{q}}, \epsilon \bigg ]^{-1}\bigg \} \nabla_{\theta }R_{\theta }^{t}(q, a)
$
其中，$\epsilon$为NFT中负似然比的裁剪阈值，$max[\cdot, \epsilon]$表示对括号内结果进行下限裁剪（确保不低于$\epsilon$），其余符号含义与GRPO梯度公式一致。

所有证明过程详见附录A。对比式（11）与式（12）可知，NFT与GRPO的唯一差异体现在**训练数据为离线策略（off-policy）时的梯度裁剪策略**（见图4）：GRPO在学习策略$\pi_\theta$与原始策略偏离过大时，会直接将梯度置零；而NFT则采用“更柔和”的衰减机制（通过似然比的倒数或裁剪后的倒数实现梯度权重衰减）。


### 4.2 命题4.2（在线策略下的梯度等价性）
基于命题4.1的设定，且当$\epsilon \leq 1$时，**在线策略训练场景下NFT与GRPO的损失梯度完全等价**，即：
$
R_{\theta}^{t}(q, a)=1 \Rightarrow \nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{NFT}(\theta)=\nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{GRPO}(\theta)
$
这一发现令人意外——尽管NFT（监督学习框架）与GRPO（强化学习框架）的推导路径截然不同，但在策略更新与数据采集完全同步的在线策略场景下，二者对模型参数的优化方向与强度完全一致。


### 4.3 隐式组归一化（Implicit Group Normalization）
命题4.1还表明，GRPO中“归一化优势”（normalized advantage）项已**隐式包含在NFT的损失函数中**。这一发现为GRPO中“组归一化（Group Normalization）”的设计选择提供了理论依据——该设计最初仅作为一种经验性优化手段被提出[41]，而NFT的梯度结构证明了其合理性。

在附录A中，我们进一步证明：通过调整prompt权重$\omega(q)=1-\hat{r}_{q}$，NFT可与Dr. GRPO[31]（GRPO的变体，移除了优势计算中的标准差归一化步骤）实现梯度对齐。这些发现共同表明，在二进制奖励场景下，强化学习（RL）与监督学习（SL）框架之间存在深刻的内在关联。

## 5 实验
我们通过实验旨在解答以下问题：
1. NFT与GRPO等现有RL算法相比，性能表现如何？（见5.2节）
2. 负样本对NFT的性能提升有何贡献？（见5.3节）
3. 哪些经验性设计选择对NFT的有效性至关重要？（见5.4节）


### 5.1 实验设置
#### 训练配置
我们在Qwen2.5-Math-7B[57]和Qwen2.5-32B[56]模型上进行在线微调，以提升其数学能力，且整个过程不依赖外部“教师模型”。训练数据集采用公开可用的DAPO-Math-17k[58]，该数据集仅包含数学问题及对应的整数形式标准答案。所有模型均训练约5000个梯度步，批大小（batch size）设为512，生成温度（generation temperature）固定为1.0。

#### 评估配置
我们在6个验证基准上评估模型性能，并报告其平均准确率，这些基准包括：AIME 2024、AIME 2025、AMC 2023[27]、MATH500[20]、OlympiadBench[19]和Minerva Math[26]。验证时采用的top-p值为0.7；7B模型的验证温度设为1.0，32B模型则设为0.6。训练验证阶段使用math-verify[24]作为验证器，最终评估阶段则采用simpleRL验证器[64]。

#### 基准方法
我们将NFT与一系列在线微调算法进行对比，包括迭代式DPO（Iterative DPO）[38,52,18]、GRPO[41]、Dr. GRPO[31]、DAPO[58]以及RFT[15,62]。下文重点介绍DAPO和RFT，其他算法的详细信息见附录B。
- **DAPO**：GRPO的一种变体，在32B模型上实现了当前最先进的AIME性能。我们的NFT实现基于DAPO的官方代码库，依托VeRL框架[42]开发，继承了DAPO的大部分超参数与设计选择，包括动态数据采样、token级损失归一化以及无KL正则化。
- **RFT**：一种简单有效的SL基准方法，仅使用正样本对LLM进行微调，完全丢弃负样本。在我们的实现中，RFT与NFT的主要区别在于：RFT会将负样本的损失置零，且训练过程中使用固定的提示词权重ω(q)=1。


### 5.2 NFT性能评估
#### 模型对比
我们将NFT应用于Qwen2.5-Math-7B模型，得到开源模型NFT-7B-Zero（见图5）。与其他7B零样本风格数学模型[13,31,53,52]相比，NFT-7B-Zero在所有基准测试中均实现了具有竞争力的性能。这一结果为NFT算法的有效性提供了有力的实证支持，同时表明仅通过SL框架即可实现LLM在数学任务中的有效自改进。

#### 算法对比
为单独评估算法本身的贡献，我们在相同的训练数据、基础设施和通用超参数条件下，对各类在线算法进行了基准测试（见表1）。结果显示，NFT的性能与DAPO等当前最先进方法相当，甚至有所超越。图6和图11展示了多轮训练的曲线，可见NFT在收敛速度和最终性能上均与DAPO持平，进一步证明了其稳定性。


### 5.3 负样本的作用
#### 负反馈提升性能与探索能力
表1显示，NFT在所有场景下均以明显优势优于RFT，这凸显了训练过程中融入负反馈的价值。值得注意的是，我们观察到RFT与NFT的训练动态存在显著差异：在7B和32B模型的训练过程中，RFT的熵（entropy）随时间推移不断降低，而NFT与DAPO等RL方法则会促进熵的增加（见图8）。这种差异表明，NFT能鼓励模型进行更主动的探索[58]，这可能是其与RFT产生性能差距的核心原因。

#### 负反馈对大模型的重要性递增
在32B模型的实验中，RFT与NFT的性能差距随训练进程不断扩大（见图11），而这一趋势在7B模型中并不明显。类似地，DeepSeek-R1的研究报告[14]也指出，在更大规模的模型中，RL相比监督微调（SFT）能带来更显著的收益。这一现象的潜在解释是：随着模型规模增大，负样本的重要性会逐步提升。

此外，RFT仍是一个性能强劲的基准方法：尽管被多种算法超越，但其极致的简洁性仍值得关注。在32B模型的实验中（见图11），仅通过正样本学习（RFT）就能实现最佳模型80%的性能提升，而负样本的贡献仅占剩余的20%。这一发现与近期的相关研究[63,53,30,66,50]结论一致，即RL在大模型中主要是放大已有的能力，而非培养全新技能。如何更高效地利用负反馈，仍是一个具有巨大潜力的开放性问题。


### 5.4 NFT有效性的关键设计
我们重点讨论对NFT实现优异性能至关重要的两项经验性设计选择：

#### 优先关注难题
我们发现，为答案正确率（$\hat{r}_q$）较低的难题分配更高权重，能有效提升模型性能。在式（10）的提示词权重ω(q)选择上，我们测试了三种方案：
1. ω(q)=1（均匀权重）；
2. ω(q)=1−$\hat{r}_q$（该设置使NFT在在线策略训练中与Dr. GRPO对齐，见4节）；
3. ω(q)=√[(1−$\hat{r}_q$)/$\hat{r}_q$]（该设置使NFT与GRPO对齐）。

图9展示了不同ω(q)的效果，结果显示方案（2）和（3）的性能相近，且均优于方案（1）的均匀权重设置。

#### 避免过度惩罚错误
NFT中式（10）的裁剪值ε，为负答案的似然比（$R_{\theta}^t$）增大时的惩罚权重设定了上限。当ε取值较小时（如接近0），算法会对错误答案似然比的上升施加高强度惩罚（见图10）。然而，我们的实验表明，若采用ε→0的过度激进惩罚策略，会导致模型整体性能下降。因此，我们将ε的默认值设为1.0。

## 6 相关工作
### 6.1 带可验证奖励的强化学习（RLVR）
带可验证奖励的强化学习（RLVR）推动了大语言模型（LLMs）推理能力的前沿发展[14, 36, 45, 9]。与以往依赖强大奖励模型[49, 60, 65]来模拟人类反馈[37, 10, 13]的强化学习实践不同，RLVR采用真值验证器（ground truth verifier）提供可靠的二进制监督信号[25, 41]。此外，与直接偏好优化（DPO）[38, 5, 2, 16, 48, 6, 21, 54]等基于偏好的学习算法不同，RLVR无需成对的偏好数据，因此具备更高的灵活性和内存效率。

尽管强化学习算法在基于验证器的训练中已被证明有效[28, 1, 22, 58, 12, 61, 55]，但近期研究表明，监督学习（SL）或许同样能实现LLMs的自改进[15, 53]。本文提出的方法进一步解决了监督学习无法融入负反馈的问题[23]，不仅弥合了这两个领域（SL与RL）在理论和性能上的差距，还可轻松适配掩码语言模型（masked LMs）[17, 68, 33]等其他语言范式。

### 6.2 隐式策略建模的相关设计
NFT的核心设计之一是通过隐式策略建模实现直接策略优化。这种“通过隐式定义的模型实现直接优化”的设计思路，与部分现有方法存在概念上的相似性。在基于偏好的训练中，DPO[38]引入了由策略网络参数化的隐式奖励模型，从而实现对策略的直接优化。在近期的视觉建模研究中，也有工作利用由生成网络参数化的隐式条件模型或残差模型，以避免引导采样（guided sampling）[8, 7]或提升生成质量[67]。


## 7 结论
在本文中，我们提出了**负样本感知微调（NFT）**——一种监督学习方法，能够让大语言模型（LLMs）从自身生成的负样本中学习。在在线训练过程中，NFT通过额外利用负反馈，显著优于传统监督学习基准（如拒绝采样微调RFT），且性能可与GRPO等主流强化学习（RL）算法持平。

值得注意的是，尽管NFT与GRPO源自完全不同的理论框架，但我们在严格的在线策略（on-policy）训练条件下，揭示了二者的理论等价性。这些发现不仅证明了监督学习在基于验证器的自改进任务中具备强大能力，还大幅弥合了二进制反馈学习系统中监督学习（SL）与强化学习（RL）范式在概念和实践上的差距。

## A 定理证明
### A.1 定理A.1（基于负答案的策略优化）
考虑针对隐式负策略$\pi_{\theta}^{-}$训练的最大似然目标函数：
$
\max _{\theta} \mathbb{E}_{p(q) \pi^{-}(a | q)}\left[\log \pi_{\theta}^{-}(a | q)\right] \Leftrightarrow \min _{\theta}\left[-\mathbb{E}_{(q, a) \sim \mathcal{D}^{-}} \log \frac{\pi(a | q)-r_{q} \pi_{\theta}^{+}(a | q)}{1-r_{q}}\right]
$
在数据量无限且模型容量足够的前提下，求解式（8）的最优解满足：
$
\forall q,a: \pi _{\theta }^{+}(a|q)=\pi ^{+}(a|q)
$

#### 证明过程
该证明过程十分直观。首先，我们证明最大似然训练会收敛至最优解$\pi_{\theta^{*}}^{-}(a | q)=\pi^{-}(a | q)$：
$
\begin{array} {rl}
&{ \max_{\theta }\mathbb {E}_{p(q)\pi ^{-}(a | q)}\left[ \log \pi _{\theta }^{-}(a | q)\right] }\\ 
{\Leftrightarrow }&{\max_{\theta }\mathbb {E}_{p(q)\pi ^{-}(a | q)}\left[ \log \pi _{\theta }^{-}(a | q)-\log \pi ^{-}(a | q)\right] }\\ 
&{\Leftrightarrow \min_{\theta }\mathbb {E}_{p(q)}D_{KL}\left[ \pi ^{-}(a | q) \| \pi _{\theta }^{-}(a | q)\right] }
\end{array}
$
由于KL散度$D_{KL}[\pi^{-}(a | q) \| \pi_{\theta}^{-}(a | q)] \geq 0$，当且仅当对所有问题$q$均满足$\pi_{\theta}^{-}=\pi^{-}$时，等号成立。因此可得：
$
\begin{array} {rlr}
{\forall q,a:}&{\pi _{\theta ^{*}}^{-}(a|q)=\pi ^{-}(a|q).}&{}&{ (13)}
\end{array}
$

接下来，我们证明$\pi_{\theta^{*}}^{+}=\pi^{+}$：

需注意，在训练过程中，$\pi_{\theta}^{-}$仅是通过以下关系由$\pi_{\theta}^{+}$定义的隐式策略：
$
\pi _{\theta }^{-}(a|q):=\frac {\pi (a|q)-r_{q}\pi _{\theta }^{+}(a|q)}{1-r_{q}}.
$
另一方面，根据式（7），负样本分布$\pi^{-}$与正策略$\pi^{+}$同样满足上述关系：
$
\pi ^{-}(a|q):=\frac {\pi (a|q)-r_{q}\pi ^{+}(a|q)}{1-r_{q}}.
$
我们在训练中已确保$0<r_{q}<1$，结合上述观察结果与式（13），可推出：
$
\forall q,a: \pi _{\theta *}^{+}(a | q)=\pi ^{+}(a | q)
$
口


### A.2 命题A.2（算法梯度比较）
假设对于某一问题$q$，存在$\hat{r}_{q} K$个正答案与$(1-\hat{r}_{q}) K$个负答案（其中$\hat{r}_{q}$为该问题的估算正确率，$K$为每个问题的生成答案总数）：

#### （a）GRPO梯度
考虑式（3）中仅包含{0,1}二进制奖励的情况，GRPO的损失梯度满足：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{GRPO}(\theta)=-\sum\left\{r A_{q}^{+} \cdot \mathcal{I}\left[R_{\theta}^{t}(q, a)<1+\epsilon'\right]+(1-r) A_{q}^{-} \cdot \mathcal{I}\left[R_{\theta}^{t}(q, a)>1-\epsilon'\right]\right\} \nabla_{\theta} R_{\theta}^{t}(q, a)
$
其中，$A_{q}^{+}=\sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}}$与$A_{q}^{-}=-\sqrt{\frac{\hat{r}_{q}}{1-\hat{r}_{q}}}$分别为正、负答案的归一化优势值。

#### （b）NFT梯度
若设置提示词权重$\omega(q)=\sqrt{(1-\hat{r}_{q}) / \hat{r}_{q}}$，则NFT的损失梯度满足：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{NFT}(\theta)=-\sum\left\{r A_{q}^{+} \cdot \frac{1}{R_{\theta}^{t}(q, a)}+(1-r) A_{q}^{-} \cdot \max \left[\frac{1-\hat{r}_{q} R_{\theta}^{t}(q, a)}{1-\hat{r}_{q}}, \epsilon\right]^{-1}\right\} \nabla_{\theta} R_{\theta}^{t}(q, a) .
$

#### 证明过程
（a）GRPO梯度：首先回顾式（3）中GRPO损失函数的定义：
$
\mathcal{L}_{\mathcal{D}}^{GRPO}(\theta)=-\sum_{q, a, t} \min \left[R_{\theta}^{t}(q, a) \hat{A}_{q, a}, \text{clip}\left(R_{\theta}^{t}(q, a), 1-\epsilon', 1+\epsilon'\right) \hat{A}_{q, a}\right] .
$
归一化优势值可通过以下方式估算：
$
\hat{A}_{q, a}:=\left[r(q, a)-\text{mean}\left\{r^{1: K}\right\}\right] / \text{std}\left\{r^{1: K}\right\}
$
其中，$\text{mean}\left\{r^{1: K}\right\}=\frac{1}{K}\left[\hat{r}_{q} K \times 1+(1-\hat{r}_{q}) K \times 0\right]=\hat{r}_{q}$，而标准差$\text{std}\left\{r^{1: K}\right\}$的计算如下：
$
\text{std}\left\{r^{1: K}\right\}=\sqrt{\frac{1}{K}\left[\hat{r}_{q} K \times\left(1-\hat{r}_{q}\right)^{2}+(1-\hat{r}_{q}) K \times\left(0-\hat{r}_{q}\right)^{2}\right]}=\sqrt{\hat{r}_{q}\left(1-\hat{r}_{q}\right)}.
$

当$a$为正答案时，$r(q, a)=1$，此时$A_{q}^{+}=\sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}}>0$，GRPO在正样本集上的损失及梯度为：
$
\mathcal {L}_{\mathcal {D}^{+}}^{GRPO}(\theta )=-\sum _{q,a^{+},t}\min\left[ R_{\theta }^{t}(q,a^{+}),1+\epsilon ^{\prime }\right] \hat {A}_{q,a^{+}}
$
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{+}}^{GRPO}(\theta)=-\sum_{q, a^{+}, t} A_{q}^{+} \cdot \mathcal{I}\left[R_{\theta}^{t}\left(q, a^{+}\right)<1+\epsilon'\right] .
$

当$a$为负答案时，$r(q, a)=0$，此时$A_{q}^{-}=-\sqrt{\frac{\hat{r}_{q}}{1-\hat{r}_{q}}}<0$，GRPO在负样本集上的损失及梯度为：
$
\mathcal {L}_{\mathcal {D}^{-}}^{GRPO}(\theta )=-\sum _{q,a^{-},t}\max\left[ R_{\theta }^{t}(q,a^{-}),1-\epsilon ^{\prime }\right] \hat {A}_{q,a^{-}}
$
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{-}}^{GRPO}(\theta)=-\sum_{q, a^{-}, t} A_{q}^{-} \cdot \mathcal{I}\left[R_{\theta}^{t}\left(q, a^{-}\right)>1-\epsilon'\right] .
$

结合上述两式（式14与式15），可得GRPO的整体损失梯度：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}}^{GRPO}(\theta)=-\sum\left\{r A_{q}^{+} \cdot \mathcal{I}\left[R_{\theta}^{t}(q, a)<1+\epsilon'\right]+(1-r) A_{q}^{-} \cdot \mathcal{I}\left[R_{\theta}^{t}(q, a)>1-\epsilon'\right]\right\} \nabla_{\theta} R_{\theta}^{t}(q, a).
$

（b）NFT梯度：首先回顾式（10）中NFT损失函数的定义：
$
\mathcal{L}_{\mathcal{D}}^{NFT}(\theta)=-\sum_{q, a, t} \omega(q)\left[r \log R_{\theta}^{t}(q, a)+(1-r) \log \max\_v\left(\frac{1-\hat{r}_{q} R_{\theta}^{t}(q, a)}{1-\hat{r}_{q}}, \epsilon\right)\right]
$

当$a$为正答案时，$r(q, a)=1$，NFT在正样本集上的损失及梯度为：
$
\begin{aligned} 
\mathcal{L}_{\mathcal{D}^{+}}^{NFT}(\theta) & =-\sum_{q, a^{+}, t} \omega(q) \log R_{\theta}^{t}(q, a) \\ 
& =-\sum_{q, a^{+}, t} \sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}} \log R_{\theta}^{t}(q, a) \\ 
& =-\sum_{q, a^{+}, t} A_{q}^{+} \log R_{\theta}^{t}\left(q, a^{+}\right) 
\end{aligned}
$
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{+}}^{NFT}=-\sum_{q, a^{+}, t} A_{q}^{+} \frac{1}{R_{\theta}^{t}\left(q, a^{+}\right)} \nabla_{\theta} R_{\theta}^{t}\left(q, a^{+}\right)
$

当$a$为负答案时，$r(q, a)=0$，NFT在负样本集上的损失及梯度为：
$
\begin{aligned} 
\mathcal{L}_{\mathcal{D}^{-}}^{NFT}(\theta) & =-\sum_{q, a^{-}, t} \omega(q) \log \left[\max\_v\left(\frac{1-\hat{r}_{q} R_{\theta}^{t}\left(q, a^{-}\right)}{1-\hat{r}_{q}}, \epsilon\right)\right] \\ 
& =-\sum_{q, a^{-}, t} \sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}} \log \left[\max\_v\left(\frac{1-\hat{r}_{q} R_{\theta}^{t}\left(q, a^{-}\right)}{1-\hat{r}_{q}}, \epsilon\right)\right] 
\end{aligned}
$
$
\begin{aligned} 
\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{-}}^{NFT} & =-\sum_{q, a^{-}, t} \sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}}\left[\max \left(\frac{1-\hat{r}_{q} R_{\theta}^{t}\left(q, a^{-}\right)}{1-\hat{r}_{q}}, \epsilon\right)^{-1} \cdot \frac{-\hat{r}_{q}}{1-\hat{r}_{q}} \cdot \nabla_{\theta} R_{\theta}^{t}\left(q, a^{-}\right)\right] \\ 
& =-\sum_{q, a^{-}, t}-\sqrt{\frac{\hat{r}_{q}}{1-\hat{r}_{q}}}\left[\max \left(\frac{1-\hat{r}_{q} R_{\theta}^{t}\left(q, a^{-}\right)}{1-\hat{r}_{q}}, \epsilon\right)^{-1} \cdot \nabla_{\theta} R_{\theta}^{t}\left(q, a^{-}\right)\right] \\ 
& =-\sum_{q, a^{-}, t} A_{q}^{-}\left[\max \left(\frac{1-\hat{r}_{q} R_{\theta}^{t}\left(q, a^{-}\right)}{1-\hat{r}_{q}}, \epsilon\right)^{-1} \cdot \nabla_{\theta} R_{\theta}^{t}\left(q, a^{-}\right)\right] 
\end{aligned}
$

结合上述两式（式16与式17），可得NFT的整体损失梯度：
$
\nabla _{\theta }\mathcal {L}_{\mathcal {D}}^{NFT}(\theta )=-\sum \bigg \{ rA_{q}^{+}\cdot \frac {1}{R_{\theta }^{t}(q,a)}+(1-r)A_{q}^{-}\cdot \max\bigg [\frac {1-\hat {r}_{q}R_{\theta }^{t}(q,a)}{1-\hat {r}_{q}},\epsilon \bigg ]^{-1}\bigg \} \nabla _{\theta }R_{\theta }^{t}(q,a).
$
口


### A.3 备注A.3（关于Dr. GRPO）
Dr. GRPO[31]与GRPO的主要实际差异在于：Dr. GRPO在估算组归一化优势值（group-normalized advantages）时，移除了标准差归一化项。根据命题4.1，只需将提示词权重设置为$\omega(q)=1-\hat{r}_{q}$（而非$\omega(q)=\sqrt{\frac{1-\hat{r}_{q}}{\hat{r}_{q}}}$），即可使NFT的损失函数与Dr. GRPO对齐。


### A.4 命题A.4（在线策略下的梯度等价性）
基于命题4.1的设定，且当$\epsilon \leq 1$时，在线策略（on-policy）训练场景下，GRPO与NFT的损失梯度完全等价：
$
R_{\theta }^{t}(q,a)=1 \Rightarrow \nabla _{\theta }\mathcal {L}_{\mathcal {D}}^{NFT}(\theta )=\nabla _{\theta }\mathcal {L}_{\mathcal {D}}^{GRPO}(\theta )
$

#### 证明过程
该证明过程较为简洁。当处于在线策略训练场景时，$R_{\theta}^{t}(q, a)=1$：

对于正答案$a^{+}$，GRPO梯度（式14）与NFT梯度（式16）可简化为同一形式：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{+}}^{GRPO}(\theta)=\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{+}}^{NFT}(\theta)=A_{q}^{+} \nabla_{\theta} R_{\theta}^{t}\left(q, a^{+}\right) 
$

对于负答案$a^{-}$，GRPO梯度（式15）与NFT梯度（式17）同样可简化为同一形式：
$
\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{-}}^{GRPO}(\theta)=\nabla_{\theta} \mathcal{L}_{\mathcal{D}^{-}}^{NFT}(\theta)=A_{q}^{-} \nabla_{\theta} R_{\theta}^{t}\left(q, a^{-}\right) .
$

## B 实验细节
### B.1 通用训练设置
所有算法均基于VeRL框架[42]下的DAPO官方代码库实现。所有实验均采用1e-6的学习率，并搭配线性预热调度。在每个滚动步（rollout step）中，我们为512个采样问题各生成16个答案，随后将数据划分为16个迷你批（mini-batch），并对策略网络进行16个梯度步的训练。模型共训练320个滚动步，累计完成超过5000个梯度步。除非另有说明，否则我们均遵循DAPO的默认设计选择，包括动态数据采样、token级损失归一化以及不使用KL正则化。

对于7B模型的训练，我们将上下文长度限制为4K，并使用64块NVIDIA H100 GPU；对于32B模型的训练，我们将DAPO的上下文长度限制为32K，其他算法则限制为16K，并使用128-256块NVIDIA H100 GPU。


### B.2 各算法具体设置
#### B.2.1 DAPO
我们严格遵循DAPO官方代码库的实现方式，未修改任何超参数。

#### B.2.2 NFT
与DAPO相比，NFT在32B模型训练时将上下文长度调整为16K。实验表明，这一调整对性能无显著影响，但能显著缩短数据采集时间。另一处差异是，我们移除了DAPO中过长答案的奖励塑造（reward shaping）技术，以确保奖励结果为二进制（0或1）。在我们的设置中，被截断的答案会被判定为负样本，这一方式足以抑制模型生成过长答案。NFT的负似然比裁剪参数设为$\epsilon=1.0$，提示词权重定义为$\omega(q)=1-r_{q}$。

#### B.2.3 RFT
在NFT的实现基础上，RFT将负样本的损失置零，并在训练过程中使用固定的提示词权重$\omega(q)=1$。

#### B.2.4 GRPO
GRPO未采用DAPO提出的动态采样技术，而是直接使用所有数据进行训练——尽管算法本身会自动将全为正样本或全为负样本的问题对应的梯度置零[58]。其他与DAPO相关的技术均被保留，例如将正样本裁剪参数设为更高的$\epsilon_{+}'=0.28$。由于GRPO的数据采样耗时更短，我们将GRPO模型的训练滚动步设置为580步以上（而非320步），使其训练总时长与DAPO实验大致相当。

#### B.2.5 Dr. GRPO
Dr. GRPO基于我们的GRPO实现修改而来，唯一差异在于计算组归一化优势值（group-normalized advantages）时，移除了标准差归一化项。

#### B.2.6 迭代式DPO（Iterative DPO）
由于DPO需要成对数据来计算训练目标，而我们为每个问题采样了16个非成对答案，因此难以直接将DPO与其他RL算法进行一对一对比。为解决这一问题，我们采用了InfoNCA[6]的实现方案——InfoNCA是DPO算法的一种变体，可处理每个问题对应$K>2$个答案的场景，其损失函数如下：
$
\mathcal{L}_{\left(q, a^{1: K}, r^{1: K}\right) \sim \mathcal{D}}^{InfoNCA }(\theta)=-\sum_{k=1}^{K}\left[\frac{r^{(k)}}{\sum_{i=1}^{K} r^{(i)}} \log \frac{e^{\beta R_{\theta}\left(q, a^{k}\right)}}{\sum_{i=1}^{K} e^{\beta R_{\theta}\left(q, a^{i}\right)}}\right]
$
当$K=2$时，InfoNCA可确保退化为标准DPO算法。我们对$\beta \in \{0.1, 0.01, 0.02\}$进行了消融实验，并报告验证结果的最佳平均值。实验发现InfoNCA训练过程不稳定，因此我们在原始损失函数中加入了监督微调（SFT）正则化项，以稳定训练过程。


### B.3 验证细节
验证过程采用0.7的top-p值；7B模型的验证温度设为1.0，32B模型设为0.6，且上下文长度与训练配置保持一致。训练验证阶段使用math-verify[24]作为验证器，最终评估阶段则采用simpleRL验证器[64]。

DAPO-17k基准数据集仅包含训练问题，其标准答案均为整数，且每个问题均包含前缀（prefix）和后缀（lastfix）。因此，在AIME和AMC问题的验证中，我们调整了提示词格式以匹配训练模式；对于其他包含非整数答案的基准数据集，问题提示词则保持不变。