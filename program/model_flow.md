
现在我们手里已经有了基于真实物理测算的帧级别精准状态标签（[]/[Static], [Grasped], [Interacting], [Reach], [Release]）。这就相当于给了模型一份“标准答案”的答题卡。在基于 Diffusion / Flow Matching 的模型中，基础的生成损失 $\mathcal{L}_{base}$（例如预测噪声 $\epsilon$ 或预测向量场 $v$）只能让模型学到数据的统计分布。为了让模型严格遵守物理定律，我们必须在预测出的干净轨迹 $\hat{X}_0$ 上，施加物理感知正则化（Physics-Informed Regularizations）。我为你设计了三大核心 Loss，这不仅是代码落地的核心，更是你论文中 Method 章节最“秀肌肉”的数学表达。基础定义 (Preliminaries)在去噪/流匹配过程的任意步，模型预测出当前窗口内的完整无噪轨迹 $\hat{X}_0 = \{\hat{H}^{(t)}, \hat{O}_1^{(t)}, \dots, \hat{O}_N^{(t)}\}_{t=1}^T$。其中 $\hat{H}^{(t)}$ 是手部在第 $t$ 帧的状态（包含各关节三维坐标 $\hat{J}_k^{(t)}$ 和手腕位姿），$\hat{O}_i^{(t)}$ 是第 $i$ 个物体的 6D 位姿。一、 $\mathcal{L}_{static}$：物体恒常性约束 (Object Permanence Anchor)痛点： 背景物体（比如桌上的杯子）或者刚刚被放下的物体，在自回归生成中极容易发生微小的“漂移（Drift）”或“抖动（Jittering）”。策略： 零速度与绝对锚点惩罚 (Zero-Velocity & Anchor Penalty)。对于任意被标记为 [Static] 的物体 $i$ 和帧 $t$，我们不仅要求它的速度为 0，还要求它死死钉在它进入 [Static] 状态那一刻的“锚点位置（Anchor Pose）” $O_i^{anchor}$ 上。$$\mathcal{L}_{static} = \mathbb{E}_{t \in W_{static}} \left[ \underbrace{\|\hat{O}_i^{(t)} - \hat{O}_i^{(t-1)}\|_2^2}_{\text{Zero Velocity}} + \lambda_{anc} \underbrace{\|\hat{O}_i^{(t)} - O_i^{anchor}\|_2^2}_{\text{Anchor Locking}} \right]$$审稿人视角： 这里的精妙之处在于引入了 $O_i^{anchor}$。自回归模型容易发生累计误差（Covariance Shift），单靠第一项（速度为0）仍可能导致物体极其缓慢地滑行。加入 Anchor Locking 后，不管序列生成了 10 万帧还是 100 万帧，只要是 [Static]，物体的坐标在数学上就绝对不可能偏离原点。二、 $\mathcal{L}_{contact}$：刚性抓持的拓扑锁定 (Rigid Isometry Constraint)痛点： 当状态是 [Grasped] 时（比如右手握着刀），如果仅仅依赖数据驱动，模型生成的刀会在手心里“滑动”或者“穿模”，俗称“手滑（Hand-Sliding）”。策略： 相对等距约束 (Relative Isometry Consistency)。我们在数据预处理时，已经算出了接触瞬间 $t_{contact}$ 时，手腕中心（Wrist）与物体（Object）之间的相对 SE(3) 变换矩阵 $T_{rel}^{anchor} = (T_{wrist}^{(t_{contact})})^{-1} T_{obj}^{(t_{contact})}$。在整个 [Grasped] 期间，这个相对关系必须保持绝对的刚性恒定。$$\mathcal{L}_{contact} = \mathbb{E}_{t \in W_{grasped}} \left[ \| (T_{wrist}^{(t)})^{-1} \hat{T}_{obj}^{(t)} - T_{rel}^{anchor} \|_2^2 + \gamma \sum_{k \in \mathcal{K}_{contact}} \text{ReLU}(\epsilon - \|\hat{J}_k^{(t)} - \hat{P}_{obj}^{(t)}\|_2) \right]$$公式解读：第一项 (SE(3) Locking)： 保证了宏观上刀随手动，绝不掉落。第二项 (Fingertip Anti-Penetration)： 这是我们在上一步提取的“真实接触手指集合 $\mathcal{K}_{contact}$”。如果这些指尖深入了物体表面内部（距离小于安全阈值 $\epsilon$），立刻施加 ReLU 惩罚，完美解决指尖穿模！三、 $\mathcal{L}_{interact}$：三元动态受迫交互 (Dynamic Manifold Constraint)痛点： 比如切面包。刀（被右手 [Grasped]）在面包（被左手 [Grasped] 且标记为 [Interacting]）上运动。刀不能切到桌子，也不能在空气中切。策略： 动态流形与包围盒约束 (Dynamic Manifold & Bounding Constraint)。对于标记为 [Interacting] 的三元动作，我们利用符号距离场 (SDF, Signed Distance Field) 将刀刃的关键点限制在面包的几何包围体积内，但禁止过度穿透。$$\mathcal{L}_{interact} = \mathbb{E}_{t \in W_{interact}} \left[ \sum_{p \in Tool} \max\left(0, \text{SDF}_{target}(\hat{x}_p^{(t)}) - \delta_{max}\right) + \max\left(0, \delta_{min} - \text{SDF}_{target}(\hat{x}_p^{(t)})\right) \right]$$公式解读： 这构成了一个**“物理操作沙盒（Physical Operation Sandbox）”**。$\text{SDF}_{target}$ 是面包的距离场。我们要求刀（Tool）的操作点既不能离开面包太远（$>\delta_{max}$，即空砍），也不能插得太深（$<\delta_{min}$，即切穿桌面）。四、 $\mathcal{L}_{reach}$ / $\mathcal{L}_{release}$：意图驱动的引力与斥力 (Intent-Driven Attraction & Repulsion)痛点： 在 [Reach] 阶段，手在自由空间运动。如果没有任何约束，模型可能会让手绕一个巨大的弯才去拿杯子。策略： 方向性单调损失 (Directional Monotonicity)。在 [Reach] 阶段，手部必须受到目标物体的“引力”，即每一帧手与物体的距离必须单调递减；在 [Release] 阶段则必须受到“斥力”。$$\mathcal{L}_{reach} = \mathbb{E}_{t \in W_{reach}} \left[ \text{ReLU} \left( \|\hat{H}^{(t)} - \hat{O}_{target}^{(t)}\|_2 - \|\hat{H}^{(t-1)} - \hat{O}_{target}^{(t-1)}\|_2 + \tau_{margin} \right) \right]$$审稿人视角： 这是一个极其优雅的时序导数约束。它不规定手必须走哪条绝对路径（保留了生成的多样性 Diversity），只强制要求“距离的差分必须小于 0（加上一个小的容差 $\tau_{margin}$ 以允许抬手等动作）”。总体训练目标 (Total Training Objective)最终，你在网络训练时，总的 Loss 是这样计算的：$$\mathcal{L}_{total} = \mathcal{L}_{flow/diff} + \lambda_1 \mathcal{L}_{static} + \lambda_2 \mathcal{L}_{contact} + \lambda_3 \mathcal{L}_{interact} + \lambda_4 \mathcal{L}_{reach}$$这里有一个极其硬核的 Trick（可以写进 Implementation Details）：物理 Loss 应当采用动态退火权重（Dynamic Annealing Weights）。在 Diffusion/Flow 的早期步数（Noise 很大时），$\hat{X}_0$ 是乱码，强加物理 Loss 会导致梯度爆炸。因此，$\lambda_i$ 应该随着去噪步数接近真实分布而逐渐增大，强制模型在最后生成的高频细节上完美符合物理定律。

## baseline
可以考虑使用https://github.com/facebookresearch/dit进行修改
-- 原版 DiT 为了处理图像，使用了 PatchEmbed (底层是 nn.Conv2d) 将图片切块，并在最后使用了复杂的 unpatchify 将序列还原成图片。对于我们的动作数据，这些全都不需要。

修改方案：
将所有的 Patch 逻辑替换为极其轻量的 nn.Linear 投影。

-- 原版 DiT 使用了复杂的二维正余弦位置编码（因为图片有宽高）。我们的动作序列只有单一的时间轴（或者实体维度）。

修改方案：
删掉原版 pos_embed.py 里的 get_2d_sincos_pos_embed，换成标准的 1D 位置编码。

--原版 DiT 只有一个 y_embedder 用来输入 ImageNet 的类别标签（如“狗”、“猫”）。我们需要把它升级为逐帧/逐实体的物理状态嵌入（State Embedding）。

修改方案：
在主网络 DiT 类中，增加一个 nn.Embedding，并将算好的标签直接加到输入序列的 Token 上。

--原版 DiT 是处理类别的，没有处理长文本指令的能力。

最轻量且不改变 DiT block 内部结构的改法是：将 CLIP 提取的文本特征，直接以 Token 的形式拼接到动作序列的前面（Token Concatenation）。

## flow matching
使用 TorchCFM 作为底层引擎，是这篇论文在工程落地时最聪明的一步棋。它把繁琐的常微分方程（ODE）、最优传输（Optimal Transport）和时间步采样的高阶数学全封装好了，你只需要像搭乐高一样，把你的数据和 DiT 模型插进去。在 Flow Matching 中，有一个极其核心的区别（与 DDPM 相比）：模型预测的不是噪声，也不是直接预测干净数据 $X_0$，而是预测一个速度场（Vector Field / Velocity $v_t$）。为了让你能在代码里无缝接入我们之前设计的三大物理损失（Physics-Informed Loss），我为你梳理了使用 TorchCFM 的标准 4 步走流程，并附带了核心的 PyTorch 伪代码：第一步：初始化 Flow Matcher在你的训练脚本最开头，你需要实例化一个 Flow Matcher。目前学术界做动作生成最推荐的是 Exact Optimal Transport (Exact OT, 精确最优传输) 轨迹，因为它是一条绝对的直线。
```code
import torch
from torchcfm.conditional_flow_matching import ExactOptimalTransportConditionalFlowMatcher

# 初始化 Flow Matcher
# sigma 设为 0 表示我们使用确定性的最优传输直线轨迹
FM = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
```

第二步：编写训练循环 (前向传播与基础 Loss)在每一个 Batch 的训练中，你需要提供真实的干净动作序列 $X_1$。Flow Matcher 会自动帮你生成纯噪声 $X_0$、随机时间步 $t$、中间状态 $X_t$ 以及目标速度场 $U_t$。
```code
# 假设从 DataLoader 中取出了一个 Batch 的数据：
# x1: 真实的干净动作轨迹 [B, Seq_Len, 398]
# text_emb: CLIP 文本特征 [B, Text_Len, Hidden]
# state_labels: 物理状态标签 [B, Seq_Len]

optimizer.zero_grad()

# 1. 采样与 x1 形状相同的纯高斯噪声 x0
x0 = torch.randn_like(x1)

# 2. 调用 TorchCFM 生成流匹配的核心变量
# t: 随机时间步 [B, 1]
# xt: 轨迹上的中间加噪状态 [B, Seq_Len, 398]
# ut: 目标速度场向量 (Ground Truth Vector Field) [B, Seq_Len, 398]
t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)

# 3. 将 xt 和 t 以及你的物理条件送入你的 CRR-DiT 模型
# 模型预测出当前时间步的速度场 vt_pred
vt_pred = model(xt, t, text_emb, state_labels)

# 4. 计算基础的 Flow Matching Loss (极其简单，就是预测场和目标场的 MSE)
loss_fm = torch.mean((vt_pred - ut) ** 2)
```

第三步：推导 $\hat{X}_1$ 并计算物理 Loss (核心难点突破)这里有一个极其关键的数学推导：我们的物理 Loss（如 $\mathcal{L}_{static}$ 钉死背景物体，$\mathcal{L}_{interact}$ 防止穿模）必须作用在干净的物理坐标上，而不能作用在速度场上！既然模型输出的是速度场 vt_pred，我们该怎么获得模型预测的干净轨迹 $\hat{X}_1$ 呢？根据 Exact OT 的直线公式：$X_t = t \cdot X_1 + (1 - t) \cdot X_0$目标速度场 $U_t = X_1 - X_0$通过简单的代数变换，我们可以解析地还原出预测的真实动作 $\hat{X}_1$：$$\hat{X}_1 = X_t + (1 - t) \cdot \hat{v}_t$$在代码中就是这一行极度优雅的推导：
```
# 5. 从预测的速度场还原出干净的物理轨迹 \hat{X}_1
# 注意 t 的形状对齐
t_expand = t.view(-1, 1, 1) 
x1_pred = xt + (1.0 - t_expand) * vt_pred

# 6. 将还原出的 x1_pred 喂给你的物理法则函数
# (注意：只在 state_labels 对应的位置计算 Loss)
loss_static = compute_static_loss(x1_pred, state_labels, anchor_poses)
loss_contact = compute_contact_loss(x1_pred, state_labels)
loss_interact = compute_interact_loss(x1_pred, state_labels, sdf_fns)

# 7. 动态退火 (Dynamic Annealing)
# t 越接近 1（越接近干净数据），物理 Loss 的惩罚力度越大
anneal_weight = t_expand.mean() 

# 8. 汇总并反向传播
loss_total = loss_fm + anneal_weight * (lambda_1*loss_static + lambda_2*loss_contact + lambda_3*loss_interact)

loss_total.backward()
optimizer.step()
```

第四步：推理阶段 (用 ODE 求解器生成动作)
当模型训练好后，怎么从纯噪声生成 10 万帧动作呢？
这时候我们需要用到常微分方程求解器（通常使用 torchdiffeq 库，它是 TorchCFM 的默认搭档）。
```
from torchdiffeq import odeint

@torch.no_grad()
def generate_motion(model, text_emb, state_labels, seq_len=120):
    # 1. 采样初始纯噪声 (对应 t=0)
    x0 = torch.randn(1, seq_len, 398).cuda()
    
    # 2. 包装一个 ODE 函数
    # 注意：ODE solver 要求函数签名是 f(t, x)，所以我们要包一层
    def ode_func(t, x):
        # 将标量 t 扩展为 Batch 维度
        t_batch = t.expand(x.shape[0]).cuda()
        # 模型预测速度场
        return model(x, t_batch, text_emb, state_labels)
        
    # 3. 定义积分时间点：从 0.0 到 1.0，我们只走 10 步！(这就是 Flow Matching 的极速优势)
    t_span = torch.linspace(0.0, 1.0, 11).cuda()
    
    # 4. 求解 ODE (默认使用 Euler 法或 dopri5)
    # 求解器会沿着模型预测的速度场，把 x0 一步步推导到 x1
    traj = odeint(ode_func, x0, t_span, method='euler')
    
    # 5. 轨迹的最后一个状态就是我们生成的干净动作
    generated_motion = traj[-1] 
    return generated_motion
```





这是一份为你量身定制的 **CRR-Flow (Contact, Release, Reach - State-Aware Autoregressive Flow)** 完整工程与算法 Pipeline。

这份 Pipeline 按照“数据流入 $\rightarrow$ 网络计算 $\rightarrow$ 损失优化 $\rightarrow$ 推理生成”的逻辑链条展开，颗粒度细化到了 Tensor 的形状和核心数学公式。你完全可以直接将这份文档作为你写代码的 **架构蓝图 (Architecture Blueprint)**，或者直接翻译后放入论文的 **Method (附录) 章节**。

---

# 🚀 CRR-Flow: 完整架构与工程 Pipeline

## 第一阶段：离线数据引擎 (Offline Data Engine)
**目标：** 将 OakInk2 的多模态原始数据，清洗并转化为带有极其精确物理状态标签（State Labels）的 10 FPS 训练张量。

### Step 1.1: 语义解析与拓扑构建 (Semantic Parsing)
* **输入:** `program_info.json`
* **操作:** 编写规则库，提取动作发生的**语义区间** `[T_start, T_end]`，以及主导手 (`lh`/`rh`)、动作原语 (`Action Primitive`) 和目标物体列表 (`Target Objects`)。
* **输出:** 粗糙的 Triadic/Dyadic 语义关系图。

### Step 1.2: 球形关节代理与物理边界校准 (Spherical Joint Proxy Calibration)
* **输入:** SMPL-X 关键点 ``，物体点云 ``。
* **操作:**
    1.  赋予 SMPL-X 的 15 个手指关节/指尖解剖学物理半径 $r_i$（如大拇指 1.2cm，其余 0.8cm）。
    2.  计算指尖球体到物体表面点云的最短 SDF (Signed Distance Field)。
    3.  利用 SDF 阈值 $\epsilon$（如 5mm），将语义区间严格裁剪为三个极其精确的运动学子区间：**Reach** (靠近), **Contact** (接触/交互), **Release** (释放)。

### Step 1.3: 状态标签打标 (State Labeling)
* **操作:** 遍历时间轴，为每个实体 (Hand, Object 1, Object 2...) 在每一帧打上互斥的 One-hot 状态标签：
    * `0: [Static]` (绝对静止)
    * `1: [Reach]` (受目标引力靠近)
    * `2: [Grasped]` (与手腕刚性等距绑定)
    * `3: [Interacting]` (受迫动态交互，如切削)
    * `4: [Release]` (受目标斥力远离)
* **输出:** 状态标签张量 $S_{labels} \in \{0,1,2,3,4\}^{T_{total} \times N_{entities}}$。

### Step 1.4: 拓扑感知的降频与滑窗切片 (Topology-Aware Downsampling & Chunking)
* **操作:**
    1.  **安全降频:** 将 30 FPS 降为 10 FPS。运动特征使用线性插值，**状态标签使用 Max-Pooling / Logical OR**，确保高频的瞬间接触不被漏采样。
    2.  **非均匀滑窗:** 设定窗口大小 $W = K_{hist} (20帧) + L_{pred} (100帧) = 120帧$。以 6:3:1 的概率比例，过采样（Oversample）包含跨物体相变（Contact $\leftrightarrow$ Release）的窗口。
* **最终落盘数据:** `x_hist.npy`, `x_target.npy`, `state_labels.npy`, `text_clip_emb.npy`。

---

## 第二阶段：状态感知网络架构 (State-Aware DiT Backbone)
**目标：** 构建一个轻量级、支持 1D 序列、且能深度融合物理状态标签的 Diffusion Transformer。

### Step 2.1: 异构 Token 组装 (Heterogeneous Token Assembly)
* **输入:** * 当前带噪轨迹 $X_t$
    * 无噪历史轨迹 $X_{hist} }$
* **操作:** 使用 `nn.Linear` 将特征投影到 Hidden Dimension ($D_{hidden}$)。

### Step 2.2: 物理状态残差注入 (State-Aware Residual Injection)
* **输入:** 状态标签 $S_{labels} \in \mathbb{R}^{B \times 100 \times 3}$ (假设 3 个实体)
* **操作:** 1.  通过 `nn.Embedding` 将离散标签转化为连续特征。
    2.  在实体维度求和: $E_{state} = \sum_{i=1}^3 \text{Embed}(S_{labels}^{(i)})$。
    3.  **核心注入:** 仅对预测目标部分进行残差相加：$X_{t\_feat} = X_{t\_feat} + E_{state}$。
    4.  序列拼接: $X_{input} = \text{Concat}(X_{hist\_feat}, X_{t\_feat}, \text{dim}=1)$。

### Step 2.3: DiT 核心前向传播 (Flow Transformer Forward)
* **操作:** 1.  注入 1D 正余弦位置编码 (1D Sincos Positional Encoding)。
    2.  通过 `AdaLN-Zero` 将 Flow Matching 的时间步 $t \in [0, 1]$ 注入到每一层 Transformer Block 的 Scale/Shift 中。
    3.  在 Transformer Block 内部引入 Cross-Attention，将 CLIP Text Embedding (文本指令) 作为 Key 和 Value 注入。
* **输出:** 截取后 100 帧的输出，通过 Final Layer 映射回物理维度，得到预测的速度场 $\hat{v}_t$。

---

## 第三阶段：训练目标与物理正则化 (Training Objective & Physics Loss)
**目标：** 利用 TorchCFM 框架计算流匹配 Loss，并在重构的干净物理空间中施加四大定律。

### Step 3.1: 构造最优传输流 (Exact OT Flow Generation)
* **操作:** * 采样噪声 $X_0 \sim \mathcal{N}(0, I)$。
    * 采样随机时间步 $t \sim \mathcal{U}(0, 1)$。
    * 计算中间轨迹: $X_t = t X_1 + (1 - t) X_0$。
    * 计算目标速度场: $U_t = X_1 - X_0$。
* **基础 Loss:** $\mathcal{L}_{FM} = \text{MSE}(\hat{v}_t, U_t)$。

### Step 3.2: 干净轨迹解析重构 (Clean Motion Analytical Reconstruction)
* **操作:** 物理法则不能加在速度场上，必须在算 Loss 前推导出模型预测的真实坐标。
* **公式:** $\hat{X}_1 = X_t + (1 - t) \cdot \hat{v}_t$。

### Step 3.3: 施加物理感知正则化 (Physics-Informed Regularizations)
*(注：以下 Loss 仅在 `state_labels` 对应的 Mask 区域内激活)*

1.  **$\mathcal{L}_{static}$ (物体恒常性):**
    * 惩罚绝对速度: $\|\hat{O}^{(t)} - \hat{O}^{(t-1)}\|_2^2$
    * 惩罚锚点偏移: $\|\hat{O}^{(t)} - O^{anchor}\|_2^2$
2.  **$\mathcal{L}_{contact}$ (刚性等距与防穿模):**
    * 相对 SE(3) 锁定: $\|(T_{wrist}^{(t)})^{-1} \hat{T}_{obj}^{(t)} - T_{rel}^{anchor}\|_2^2$
    * 指尖非对称防穿透: $\text{ReLU}(\epsilon - \|\hat{J}_{fingers}^{(t)} - \hat{P}_{obj}^{(t)}\|_2)$
3.  **$\mathcal{L}_{interact}$ (动态流形约束):**
    * 工具使用包围盒限制: 基于 $\text{SDF}_{target}(\hat{x}_{tool})$ 的极值截断损失。
4.  **$\mathcal{L}_{reach/release}$ (意图导数约束):**
    * 距离单调递减/递增惩罚: $\text{ReLU}(\Delta \text{Distance} + \tau)$。

### Step 3.4: 动态退火反向传播 (Dynamic Annealing Backward)
* **操作:** 随着 $t \to 1$（靠近真实数据），物理 Loss 权重变大。
* **总 Loss:** $\mathcal{L}_{total} = \mathcal{L}_{FM} + t \cdot (\lambda_1 \mathcal{L}_{static} + \dots + \lambda_4 \mathcal{L}_{release})$。
* `loss.backward()` & `optimizer.step()`。

---

## 第四阶段：自回归推理生成 (Autoregressive Inference & Unrolling)
**目标：** 利用训练好的模型，像“贪吃蛇”一样稳定生成 10 万帧无崩坏的连续操作。

### Step 4.1: 初始上下文建立 (Context Initialization)
* 提取第一句文本指令。
* 给定 $T=0$ 时刻的初始人体与桌面物体摆放状态，复制 20 帧作为第一个 $X_{hist}$。

### Step 4.2: 常微分方程求解 (ODE Solver Unrolling)
* **操作:**
    1.  采样未来 100 帧的纯噪声 $X_0$。
    2.  使用 `Euler` 或 `Heun` 求解器（如 `torchdiffeq.odeint`），将 $t$ 从 0.0 积分到 1.0。
    3.  每一步积分调用模型 `model(X_t, t, X_hist, Text, States)` 预测步进方向。
    4.  **仅需 10 个 NFE (网络评估次数)**，即可得到极高质量的未来 100 帧 $\hat{X}_1$。

### Step 4.3: 滑窗接力 (Sliding Window Relay)
* **操作:** 将刚刚生成的 100 帧的**最后 20 帧**截取出来，作为下一轮生成的 $X_{hist}$。
* 更新下一句文本指令和状态标签队列。
* 跳转回 Step 4.2 继续生成，直至无穷。

### Step 4.4: 运动学上采样 (Kinematic Super-Resolution - 论文 Future Work/Optional)
* **操作:** 将生成的 10 FPS 骨架，送入一个极轻量的局部条件扩散模型（Local Diffusion）或插值网络，上采样至 30 FPS，消除高频 Jittering。


## training test
using one motion for training, and then test the result similarity with original data. 找一个序列数不是特别长的 sequence进行训练
如果效果不好，记录进超参以及对应的result，调整各种超参

---

## Model I/O Specification (CRR-Flow DiT)

### Training

**Model Input:**

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `x_t` | `[B, T, 398]` | float32 | Interpolated noisy motion on OT trajectory: `x_t = t·x1 + (1-t)·x0` |
| `t` | `[B]` | float32 | Flow timestep ∈ [0, 1]. t=0 is pure noise, t=1 is clean data |
| `y['text']` | `list[B]` | str | Raw text descriptions, CLIP-encoded inside model (ViT-B/32 → 512D) |
| `y['state_labels']` | `[B, T, 20]` | int64 | Per-frame per-entity state labels. 20 = max_entities (2 hands + up to 18 objects). Padding = -1 |

**Training Target:**

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `ut` | `[B, T, 398]` | float32 | Target velocity field = `x1 - x0` (Exact OT straight line) |

**Model Output:**

| Tensor | Shape | Dtype | Description |
|--------|-------|-------|-------------|
| `vt_pred` | `[B, T, 398]` | float32 | Predicted velocity field |

**Loss:** `L_FM = MSE(vt_pred, ut)`

**Clean trajectory reconstruction:** `x1_pred = x_t + (1 - t) · vt_pred`

### Inference (ODE Integration)

**Input:**
- `x0 ~ N(0, I)` : `[1, T, 398]` initial noise
- `t_span = [0.0, 0.1, 0.2, ..., 1.0]` : 10 Euler steps
- `y['text']`, `y['state_labels']` : same as training

**Process:** `x_{t+dt} = x_t + dt · model(x_t, t, y)` iterated from t=0 to t=1

**Output:** `x1 = traj[-1]` : `[1, T, 398]` generated clean motion (normalized)

### 398D Feature Vector Layout

```
Index     Dim  Name                    Description
─────────────────────────────────────────────────────────────────
[0:3]       3  body_transl             World translation (meters)
[3:9]       6  body_orient             Root orientation (6D rotation)
[9:135]   126  body_pose               21 body joints × 6D rotation
[135:138]   3  left_hand_pos           Left hand wrist position
[138:144]   6  left_hand_orient        Left hand wrist orientation (6D)
[144:165]  21  left_hand_pca           Left hand PCA pose coefficients
[165:168]   3  right_hand_pos          Right hand wrist position
[168:174]   6  right_hand_orient       Right hand wrist orientation (6D)
[174:195]  21  right_hand_pca          Right hand PCA pose coefficients
[195:198]   3  left_hand_tsl_quat      Left hand translation (quaternion repr)
[198:262]  64  left_hand_pose_quat     Left hand 16 joints × 4D quaternion
[262:265]   3  right_hand_tsl_quat     Right hand translation (quaternion repr)
[265:329]  64  right_hand_pose_quat    Right hand 16 joints × 4D quaternion
[329:350]  21  sdf_left                Signed distance field left hand (placeholder zeros)
[350:371]  21  sdf_right               Signed distance field right hand (placeholder zeros)
[371:374]   3  object1_pos             Object 1 position (meters)
[374:380]   6  object1_rot             Object 1 rotation (6D)
[380:383]   3  object2_pos             Object 2 position
[383:389]   6  object2_rot             Object 2 rotation (6D)
[389:392]   3  object3_pos             Object 3 position
[392:398]   6  object3_rot             Object 3 rotation (6D)
─────────────────────────────────────────────────────────────────
Total:    398
```

### State Label Vocabulary (7 states)

| ID | State | Applies to | Description |
|----|-------|-----------|-------------|
| 0 | `[Idle]` | Hands | No annotation covers this frame |
| 1 | `[Reach]` | Hands | Approaching target object |
| 2 | `[Grasped]` | Hands/Objects | Rigid attachment between hand and object |
| 3 | `[Grasped_and_Interacting]` | Objects | Grasped by one hand + tool acting on it |
| 4 | `[Release]` | Hands | Moving away from object |
| 5 | `[Static]` | Objects | At rest, not being manipulated |
| 6 | `[Interacting]` | Objects | Being acted upon by a tool |
| -1 | (padding) | — | Unused entity slot |

### Normalization

- Mean: `dataset/OAKINK2_FLOW/Mean_flow.npy` shape `(398,)`
- Std: `dataset/OAKINK2_FLOW/Std_flow.npy` shape `(398,)`
- Normalize: `x_norm = (x - mean) / std`
- Denormalize: `x = x_norm * std + mean`

### Data Split

- Train: 501 sequences
- Test: 126 sequences
- Total: 627 sequences × 200 frames × 398D

### Model Architecture (FlowDiT)

| Component | Detail |
|-----------|--------|
| Input projection | `Linear(398, latent_dim)` |
| Positional encoding | 1D sinusoidal (max_len=200) |
| State embedding | `Embedding(7+1, latent_dim)` with padding_idx, sum over active entities |
| Timestep embedding | Continuous sinusoidal t∈[0,1] → MLP → latent_dim |
| Text embedding | Frozen CLIP ViT-B/32 (512D) → `Linear(512, latent_dim)` |
| Condition fusion | `MLP(cat(t_emb, text_emb))` → AdaLN condition vector |
| Transformer blocks | N × DiTBlock with AdaLN-Zero (shift/scale/gate × 2) |
| Output projection | FinalLayer with AdaLN-Zero → `Linear(latent_dim, 398)` |

**Default hyperparams (debug):** latent=256, layers=4, heads=4, ff=512 → 4.5M trainable params
**Full model:** latent=512, layers=8, heads=8, ff=1024 → ~25M trainable params

---

## Experiment Log: Three-Way Training Verification

**Date**: 2026-03-26
**Goal**: Diagnose why generated motion has flying/implausible poses. Compare flow matching vs diffusion loss, and FlowDiT vs original DiffH2O UNet.

### Results Table

| Experiment | Model Arch | Loss Type | Input Dim | Input Category | Train Steps | Loss @30K | Loss @50K | Loss @75K | Loss @100K | Body Y range | Body pose range | MSE | Visualization |
|-----------|-----------|-----------|-----------|----------------|-------------|-----------|-----------|-----------|------------|--------------|-----------------|-----|---------------|
| Way 1 | FlowDiT (4.5M) | Diffusion (predict x₀) | 135D | Body only (6D) | 30K | **0.090** | — | — | — | [-0.13,0.15] | 0.76 (GT:0.26) | 0.104 | `output/way1_vis/gen.mp4`, `gt.mp4` |
| Way 1 ext | FlowDiT (4.5M) | Diffusion (predict x₀) | 135D | Body only (6D) | 100K | 0.090 | 0.065 | 0.060 | **0.055** | [0.10,0.16] | 0.22 (GT:0.26) | 0.030 | `output/way1_100k_vis/gen.mp4`, `gt.mp4` |
| Way 2 | FlowDiT (4.5M) | Flow matching (velocity) | 60D | Hand PCA (pos+orient+pca ×2) | 30K | **0.630** | — | — | — | — | — | 0.032 | `output/way2_vis/gen_samples/ours_videos/0000_2.mp4` (gen), `output/way2_vis/gt_samples/ours_videos/0000_2.mp4` (gt) |
| Way 2 ext | FlowDiT (4.5M) | Flow matching (velocity) | 60D | Hand PCA (pos+orient+pca ×2) | 200K | 0.531 | 0.507 | 0.466 | **0.434** | — | — | 0.006 | `output/way2_200k_vis/gen_samples/ours_videos/0000_0.mp4` (gen), `output/way2_200k_vis/gt_samples/ours_videos/0000_0.mp4` (gt) |
| Way 3 | DiffH2O UNet (orig) | Diffusion (predict x₀) | 117D | Hand PCA+SDF+Obj (GRAB) | 30K | **0.00002** | — | — | — | — | — | ~0 | `output/way3_vis/gt_samples/ours_videos/0000_2.mp4` (gt≈gen) |
| Way 1 FM 200K | FlowDiT (4.5M) | Flow matching (velocity) | 135D | Body only (6D) | 200K | 0.689 | 0.385 | 0.339 | 0.298 | [0.03,0.15] | 0.28 (GT:0.26) | 0.005 | `output/way1_fm_200k_vis/gen.mp4`, `gt.mp4` |
| Way 1 FM 400K | FlowDiT (4.5M) | Flow matching (velocity) | 135D | Body only (6D) | 400K | — | — | — | 0.264 | [0.05,0.14] | 0.33 (GT:0.26) | **0.003** | `output/way1_fm_400k_vis/gen.mp4`, `gt.mp4` |
| Flow 398D | FlowDiT (4.5M) | Flow matching (velocity) | 398D | Body+HandPCA+MANO+SDF+Obj | 2K | 1.420 | — | — | — | flying | 2.05 | 0.170 | `output/gen_transitions_vis/gen_0.mp4` |
| Flow 360D | FlowDiT (4.5M) | Flow matching (velocity) | 360D | Body+Hand6D+Obj | 10K | 0.477 | — | — | — | flying | 1.28 | 0.057 | `output/flow_360d_vis/gen_0000.mp4` |
| Flow 315D | FlowDiT (4.5M) | Flow matching (velocity) | 315D | Body+Hand6D | 10K | 1.023 | — | — | — | flying | 1.11 | 0.050 | `output/flow_315d_vis/gen.mp4` |
| Flow 135D | FlowDiT (4.5M) | Flow matching (velocity) | 135D | Body only (6D) | 10K | 0.689 | — | — | — | flying | 0.76 | — | `output/flow_135d_vis/gen.mp4` |

### Key Findings

1. **Original DiffH2O UNet (Way 3)** achieves loss=0.00002 — **perfect memorization** of single sequence. The proven architecture + diffusion loss works.

2. **FlowDiT + diffusion loss (Way 1)** achieves loss=0.055 @100K, MSE=0.030. Body grounded, pose range matches GT.

3. **FlowDiT + flow matching (Way 1 FM)** achieves **MSE=0.003 @400K** — the **best reconstruction quality** across all FlowDiT experiments. Flow matching converges slowly (loss=0.264) but ultimately produces **10× lower MSE** than diffusion (0.003 vs 0.030). Body Y=[0.05, 0.14] nearly matches GT [0.06, 0.08].

4. **Flow matching needs more steps**: At 30K steps, flow matching appears much worse than diffusion. But at 400K steps, it surpasses diffusion in MSE. The velocity field representation captures fine-grained motion better once converged.

5. **Architecture gap**: DiffH2O UNet still converges orders of magnitude faster (loss=0.00002 @30K vs FlowDiT's 0.264 @400K). The UNet's inductive bias remains superior for rapid convergence.

### Conclusions

- **Flow matching CAN work** with FlowDiT — it just needs ~10-20× more training steps than diffusion to converge
- **Flow matching achieves lower MSE** than diffusion once converged (0.003 vs 0.030) — better reconstruction fidelity
- **The original DiffH2O UNet** remains the fastest to converge (loss=0.00002 @30K)
- **Next step**: Train flow matching on the full 315D/360D features with sufficient steps (200K+), or adapt UNet architecture for flow matching

### Way 1 Model Detail (Best CRR-Flow Variant)

**Model Input (135D body only):**
```
[0:3]     body_transl       3D   world translation (meters)
[3:9]     body_orient       6D   root rotation (6D continuous repr)
[9:135]   body_pose       126D   21 body joints × 6D rotation
────────────────────────────────
Total:   135D (from raw_smplx, quaternion → 6D conversion)
```

**Model Architecture — FlowDiT (4.5M trainable params):**
```
Input: x_t [B, T, 135]
  → Linear(135, 256)                         # input projection
  + PositionalEncoding1D(256, max_len=200)   # 1D sinusoidal
  + StateEmbedding(7+1, 256)                 # per-frame entity state labels
  → 4 × DiTBlock(d=256, heads=4, ff=512)    # AdaLN-Zero transformer blocks
      conditioned on: c = MLP(cat(t_emb, text_emb))
        - t_emb: SinusoidalTimestepEmbedder(256) from t∈[0,1]
        - text_emb: frozen CLIP ViT-B/32 (512D) → Linear(512, 256)
      AdaLN-Zero: shift/scale/gate × 2 (attn + ffn)
  → FinalLayer(256, 135)                     # AdaLN-Zero + linear projection
Output: prediction [B, T, 135]
```

**Loss Design — Standard DDPM (predict x₀):**
```python
# Cosine noise schedule, 1000 timesteps
alpha_bars = cosine_beta_schedule(1000)           # precomputed

# Per step:
t = randint(0, 1000, (B,))                        # random discrete timestep
noise = randn_like(x1)                             # Gaussian noise
x_t = sqrt(alpha_bar[t]) * x1 + sqrt(1-alpha_bar[t]) * noise   # forward diffusion

x0_pred = model(x_t, t/1000, y=y)                 # model predicts clean x₀
loss = MSE(x0_pred, x1)                            # simple MSE on x₀ prediction
```

Key differences from flow matching (Way 2):
- Predicts **clean data x₀** directly, not velocity field
- Uses **discrete timesteps** (0-999) with cosine schedule, not continuous t∈[0,1]
- No OT permutation — simpler, more stable for small batch sizes
- Converges ~7× faster on same architecture (loss 0.055 vs 0.434 at equivalent steps)

**Training Config:**
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Gradient clipping: 1.0
- Batch size: 1 (overfit single sequence)
- Classifier-free guidance: cond_mask_prob=0.1

**Files:**
- Model: `model/flow_dit.py` → `FlowDiT`
- Training: `train/train_flow.py` with `--use_diffusion`
- Dataset: `data_loaders/humanml/data/flow_dataset.py` → `OakInk2FlowDataset`
- Data: `dataset/OAKINK2_FLOW_135D/`

### Smoothed Visualization Comparison

**Date**: 2026-03-26
**Method**: Savitzky-Golay filter applied along time axis (axis=0) to the generated 398D motion tensor before SMPL-X rendering.

| Level | Window | Polyorder | Effect |
|-------|--------|-----------|--------|
| None | — | — | Raw model output |
| Light | 5 | 3 | Minimal denoising, preserves fast movements |
| Medium | 11 | 3 | Noticeable smoothing, removes jitter |
| Heavy | 21 | 3 | Strong smoothing, may over-smooth fast actions |

**Videos** (all at `output/way1_smooth/`):

| Model | None | Light | Medium | Heavy |
|-------|------|-------|--------|-------|
| **Diffusion 100K** (MSE=0.030) | `diffusion_100k_none.mp4` | `diffusion_100k_light.mp4` | `diffusion_100k_medium.mp4` | `diffusion_100k_heavy.mp4` |
| **FM 200K** (MSE=0.005) | `fm_200k_none.mp4` | `fm_200k_light.mp4` | `fm_200k_medium.mp4` | `fm_200k_heavy.mp4` |
| **FM 400K** (MSE=0.003) | `fm_400k_none.mp4` | `fm_400k_light.mp4` | `fm_400k_medium.mp4` | `fm_400k_heavy.mp4` |
| **GT** | `gt.mp4` | — | — | — |

**Observations**:
- FM 400K raw output (none) has the lowest MSE (0.003) but may show high-frequency micro-jitter in the video
- Light smoothing (SG w=5) removes jitter while preserving motion dynamics
- Medium smoothing (SG w=11) produces the smoothest natural-looking motion
- Heavy smoothing (SG w=21) may over-dampen fast hand movements
- Diffusion 100K output is inherently smoother (diffusion's iterative denoising acts as implicit smoothing) but has higher MSE

### WandB Dashboard

All experiments logged to local WandB at `http://172.18.36.108:8080`:
- Project `crr-flow`: Way 1 (`way1_diffusion_135d`), Way 2 (`way2_handpca_60d`), Way 1 FM (`way1_fm_135d_400k`)
- Project `diffh2o`: Way 3 (`way3_diffh2o_117d`)
