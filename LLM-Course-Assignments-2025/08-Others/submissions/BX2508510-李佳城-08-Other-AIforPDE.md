## 用于求解参数化偏微分方程的上下文生成式预训练方法

### 1.背景

偏微分方程（Partial Differential Equations, PDEs）可以用于描述物理世界中的绝大多数系统，是刻画连续介质物理行为和自然现象的重要数学工具 [7]。例如，纳维-斯托克斯方程用于描述流体流动，热方程描述描述热传导过程，玻尔兹曼方程用于气体动力学的刻画。不同于少数具有良好数学性质的经典方程，大多数实际应用中的 PDEs 并不具备显式解析解，因此其求解高度依赖于数值方法。

数值方法在过去的百年时间内被科研人员广泛研究，发展出了一系列稳定且具有理论保正的工具，例如有限元方法等。然而，这类方法通常具有较高的计算代价。以定义在二维区域 $\Omega \subset \mathbb{R}^2$ 上的偏微分方程为例，数值求解通常需要先将计算区域离散为一组空间网格点 $\{(x_i, y_i)\}_{i=1}^N$，再通过求解由原方程诱导得到的离散方程组，获得各个网格点处解的数值近似 $\{u_i\}_{i=1}^N$。为了提高数值解的精度，网格规模 $N$ 往往较大，从而在计算时间和内存消耗方面带来严峻挑战，尤其是在高维或多次参数扫描的应用场景中更为突出。

近年来，深度学习，特别是神经网络方法，在许多领域取得了突破性进展，例如文本，视觉及多模态等。这促使研究人员开始探索其在偏微分方程求解中的潜力。神经网络强大的非线性表示能力，使其有望捕获 PDE 解中复杂的多尺度结构和高度非线性特征。当前，基于深度学习的 PDE 求解方法大致可以分为以下几类：

1.  物理驱动 [2]：将方程本身直接作为嵌入神经网络的损失函数，无需数据训练，代表性方法为物理信息神经网络（Physics-Informed Neural Networks, PINNs）；
2.  神经算子 [1, 4, 8]：学习一族PDEs，而非仅考虑单个固定参数问题，代表性方法为傅里叶神经算子（Fourier Neural Operator, FNO）。
3. 大模型相关：（a) 基于PINN 的Agent框架 [5]，(b) 生成式方法，如扩散模型 [6] 及自回归模型 [3] 等。

然而，不同于视觉数据，PDEs数据通常蕴含更加复杂的底层物理规律，解对方程参数和初始条件往往高度敏感，微小扰动便可能引发显著的解行为震荡。这一特性使得模型在跨参数、跨分辨率或跨物理场景下的泛化能力面临严峻挑战。近年来，一个逐渐受到关注的研究方向是上下文学习（In-Context Learning），即通过向模型提供少量的示例，引导其在新的参数或动力学条件下进行有效推理。与现有方法相比，==上下文学习在灵活性和适应性方面展现出显著优势==：模型能够利用不同类型和规模的上下文信息，仅凭有限示例即可快速适应新的问题设置，从而为复杂 PDE 系统的高效求解提供了一种具有前景的研究思路。在本报告中，我们重点关注该方向中的一种代表性实现——Zebra [3]，并对其方法框架及改进思路进行分析与讨论。



### 2.问题设置

**问题描述**：本报告关注的问题是参数化、时间依赖偏微分方程的求解。与早先工作不同，允许物理参数、边界条件和外力项发生变化，从而形成一族具有不同动力学行为的方程实例。形式上，将一个 PDE 实例定义为一个“环境” $\mathcal{F}_\xi$，其中参数集合
$$
\xi := \{\mu, B, \delta\}
$$
分别表示物理系数、边界条件和外力项。对应的 PDE 解 $u(x,t)$ 满足
$$
\frac{\partial u}{\partial t} = F\left(\delta, \mu, t, x, u, \frac{\partial u}{\partial x}, \frac{\partial^2 u}{\partial x^2}, \ldots \right), \\ 
B(u)(x,t) = 0,\quad x\in\partial\Omega, \\ 
u(0,x) = u_0,
$$
其中初始条件 $u_0$从某一分布中采样。该问题的核心难点在于：参数的微小变化可能引发解结构的显著变化，从而导致分布偏移。现有神经 PDE 求解器往往需要通过梯度更新或微调来适应新的参数环境，这不仅增加了推理成本，也限制了实际应用中的灵活性。

**问题设置**：在测试阶段，模型不再接触 PDEs 的显式参数，而只能观察到同一环境下的少量示例轨迹。模型仅依赖这些上下文信息，预测在相同动力学但不同初始条件下的演化轨迹。



### 3.方法

<img src="https://gitee.com/li-zhumeng/typora_image/raw/master/img/20260113114518.png" alt="image-20260113114516302" style="zoom: 50%;" />



Zebra被引入用于解决上述问题。作为一种受大语言模型启发的生成式自回归 PDEs 求解框架，其核心思想是：将 PDE 轨迹视为“token 序列”，通过上下文学习来完成参数适应。

Zebra 的整体结构可以概括为 “编码–生成–解码” 三个阶段，整体框架如上图所示。

首先，利用向量量化变分自编码器（VQ-VAE）将连续的物理场 $u_t$ 映射到离散潜空间。具体而言，空间编码器 $E_\omega$ 将输入状态压缩为潜变量
$$
z_t = E_\omega(u_t),
$$
随后通过码本 $\mathcal{Z}$ 进行量化，得到离散表示 $z_t^q$，并由解码器重构原始物理场。这一步相当于学习到一个“物理词表”，使 PDEs 演化可以被表示为离散 token 序列。

在此基础上，Zebra 使用自回归 Transformer对 token 序列进行建模。训练阶段采用标准的 next-token 预测目标：
$$
\mathcal{L}_{\text{Transformer}} = -\mathbb{E}_{S}\sum_{i=1}^{N} \log p\big(S[i] \mid S[<i]\big),
$$
其中序列 $S$ 由多条共享相同动力学但初值不同的轨迹拼接而成，并通过特殊标记区分不同轨迹。通过预训练，模型学会如何从上下文轨迹中“推断”潜在动力学规律。

在推理阶段，Zebra 将若干上下文轨迹与一个新的初始条件拼接为提示，并通过自回归生成未来 token，再解码回物理空间。整个过程无需任何梯度更新，因此显著快于基于元学习或微调的方法。同时，由于模型是生成式的，Zebra 不仅能够给出单一预测，还可以通过采样刻画预测不确定性，并生成符合上下文动力学分布的新轨迹。

此外，我们注意到，在 VQ 阶段的训练过程中，损失函数主要依赖于数值层面的重构误差，这在一定程度上限制了模型对局部结构与多尺度特征的刻画能力。对于偏微分方程数据而言，解的梯度往往蕴含着关键的物理信息，例如边界层、激波以及快速变化区域，而仅依赖数值重构可能导致这些细节被过度平滑。为此，我们在原有重构损失的基础上，引入了一项梯度一致性损失，用以约束重构结果与真实解在梯度空间中的一致性，其形式定义为
$$
\mathcal{L}_{\nabla} = \left\lVert \nabla \hat{u} - \nabla u \right\rVert_2^2,
$$
通过引入梯度一致性约束，模型能够更有效地捕捉解的局部变化和多尺度结构特征，从而提升整体表示质量。



### 4.实验

**实验设置**：

在偏微分方程（PDEs）研究中，训练与评测数据通常来源于高精度数值方法生成的仿真数据，数据来自文献 [3]，可由此链接下载 [advection](https://huggingface.co/datasets/sogeeking/advection)，本报告的测试代码在 [Zebra_plus](https://github.com/JcLimath/Zebra_plus)。代码复现及测试运行均在一张 RTX 4090D（24GB）GPU 上完成。针对 Advection 方程，我们简单评估了在原有模型基础上引入梯度一致性损失所带来的改进效果以验证所提出的想法。模型结构与超参数与文献 [3] 保持一致，训练步数设为 1000 step。

| Method                 | Test Error |
| ---------------------- | ---------- |
| $\text{Zebra}$         | 0.0185     |
| $\text{Zebra}^{\star}$ | 0.0166     |

**结果分析**：

实验结果被展示在上表中，可以看出，引入梯度损失后的模型的重构精度取得了明显提升，验证了该改进在建模 Advection 动力学过程中的有效性。这表明，仅依赖数值重构误差可能不足以充分刻画解的局部变化特征，而梯度约束能够帮助模型更好地捕捉解的结构信息。此外，在训练步数和参数规模均保持不变的情况下获得性能提升，也说明该方法具有良好的效率与实用性。



### 5.结论与展望

本报告针对基于向量量化的表示学习方法在偏微分方程数据建模中的不足，在 Zebra 的基础上，分析了其在多尺度结构刻画方面的局限性，并在原有框架基础上引入了梯度一致性损失，以增强模型对局部变化特征的表达能力。该改进实现简单、物理直觉明确，有助于提升离散表示的质量。未来工作可进一步结合多尺度或物理约束信息，并探索其在生成式建模和参数化偏微分方程预测任务中的应用潜力。



### 参考文献

>[1] Li et al. Fourier Neural Operator for Parametric Partial Differential Equations. ICLR, 2021.
>
>[2] Raissi et al. Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. JCP, 378: 686-707, 2019.
>
>[3] Serrano et al. Zebra: In-Context Generative Pretraining for Solving Parametric PDEs. ICML, 2025.
>
>[4] Wu et al. Transolver: A Fast Transformer Solver for PDEs on General Geometries. ICML, 2024.
>
>[5] Wu et al. PINNsAgent: Automated PDE Surrogation with Large Language Models. ICML, 2025.
>
>[6] Yao et al. Guided Diffusion Sampling on Function Spaces with Applications to PDEs. NeurIPS, 2025.
>
>[7] Zhou et al. Unisolver: PDE-Conditional Transformers Towards Universal Neural PDE Solvers. ICML, 2025.
>
>[8] Zhou et al. SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs. AAAI, 2026.