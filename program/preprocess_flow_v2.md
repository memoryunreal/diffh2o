参考之前preprocee data的操作，下面是一个新的处理方法，有些地方是可以利用之前的处理方式。

整个数据管道（Data Pipeline）可以划分为 **5 个核心阶段**，你可以直接将其作为代码仓库的模块划分或 GitHub/Jira 的开发任务。

---

### 阶段一：长序列过滤与“超级片段”重组 (Sequence Filtering & Reassembly)
**目标：** 解决 Complex Task 中无效等待时间过长的问题，将原始长视频切分为包含高密度连贯动作的“超级片段 (Continuous Segments)”。

1.  **加载元数据：** 读取 OakInk2 的 `/hhd4/lizhe/dataset/OakInk2/data/program/program_info` 和对应的 Primitive Task 注释。
2.  **计算间隔 (Gap Calculation)：** 遍历同一个 Complex Task 下的所有相邻 Primitive Task（如 $Task_A$ 和 $Task_B$）。
3.  **阈值切分 (Thresholding)：**
    * 如果 $Task_A$ 的结束帧与 $Task_B$ 的起始帧间隔 $\le 300$ 帧（原始 30FPS 下的 10 秒），则将它们视为连续，合并入同一个“超级片段”。
    * 如果间隔 $> 300$ 帧，则在此处斩断，打断连贯性，生成两个独立的“超级片段”。
4.  **输出：** 获得一系列时间区间列表 `[(start_1, end_1), (start_2, end_2), ...]`，每个区间内可能包含 1 到 N 个高度连贯的 Primitive Tasks。

### 阶段二：物理拓扑提取与状态绑定 (Kinematics & State Annotation)
**目标：** 在原始的 30FPS 频率下，提取最精确的 3D 物理特征并打上帧级别的 State Label。

1.  **SMPL-X 到 3D 坐标降维：** 调用 SMPL-X 前向传播，将人体的轴角/旋转参数转化为 77 个 Marker 的 3D 坐标（$77 \times 3$ 维）。
2.  **局部物体槽位过滤 (Object Slot Filtering)：** * 读取当前“超级片段”内涉及的所有物体。
    * 填入我们设定的 4 个 Object Slots（$4 \times 9$ 维）。无关的背景物体直接丢弃。提取物体的 BPS（特征基点集）作为静态形状特征。
3.  **距离探测与状态生成 (SDF & State Labeling)：**
    * 计算手部 10 个指尖 Marker 到目标物体表面的最短距离。
    * 利用物理阈值（如 $< 2\text{cm}$ 且文本指令吻合）判断 Contact，自动生成形状为 `[Num_Frames, 7]` 的 Entity State Map（取值范围 0~4）。

**核心规则库**
# 物理交互拓扑类型枚举
class Topology:
    TRIADIC = "triadic"          # 手 -> 工具 -> 目标物 (例: 切面包)
    DYADIC_RIGID = "dyadic"      # 手 -> 目标物 (例: 拧瓶盖)
    FLUID_CONTAINER = "fluid"    # 手 -> 容器 -> 内容物 (例: 倒水)
    STATIC_HOLD = "static_hold"  # 手 -> 目标物 (零相对运动，例: 拿着)

# 动作翻译规则库 (Action Translation Rule Base)
# 键为 JSON 中的 primitive (动词原形)，值为物理状态映射逻辑
ACTION_RULE_BASE = {
    # ==========================================
    # 类别 1: 三元工具交互 (手握工具，工具作用于目标)
    # ==========================================
    "cut": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "cut"},
    "scoop": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "scoop"},
    "scrape": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "scrape"},
    "stir": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "stir"},
    "spread": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "spread"},
    "wipe": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "wipe"},
    "brush": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "brush"},
    "write/draw": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "write"},
    "shear": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "shear"},
    "staple together": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "staple"},
    "stab": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "stab"},
    "knock": {"type": Topology.TRIADIC, "hand_to_obj1": "grasp", "obj1_to_obj2": "knock"},

    # ==========================================
    # 类别 2: 二元受迫交互 (手与目标物发生特定轨迹的相对运动或约束)
    # ==========================================
    "screw into": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_rotate"},
    "unscrew from": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_rotate"},
    "cap onto": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_translate"},
    "uncap from": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_translate"},
    "open": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_articulated"},
    "shut": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_articulated"},
    "close": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_articulated"},
    "connect to": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_insert"},
    "deconnect from": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_extract"},
    "turn": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_rotate"},
    "tighten": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_rotate"},
    "loosen": {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact_rotate"},

    # ==========================================
    # 类别 3: 容器/流体操作 (手握容器，容器姿态改变)
    # ==========================================
    "pour": {"type": Topology.FLUID_CONTAINER, "hand_to_obj1": "grasp", "obj1_constraint": "tilt"},
    "flow out": {"type": Topology.FLUID_CONTAINER, "hand_to_obj1": "grasp", "obj1_constraint": "tilt"},
    "squeeze out": {"type": Topology.FLUID_CONTAINER, "hand_to_obj1": "interact_deform", "obj1_constraint": "squeeze"},
    "shake": {"type": Topology.FLUID_CONTAINER, "hand_to_obj1": "grasp", "obj1_constraint": "periodic_motion"},

    # ==========================================
    # 类别 4: 静态保持 (强约束，零相对运动)
    # ==========================================
    "hold": {"type": Topology.STATIC_HOLD, "hand_to_obj1": "grasp"},
    "grip": {"type": Topology.STATIC_HOLD, "hand_to_obj1": "grasp"},
    "secure": {"type": Topology.STATIC_HOLD, "hand_to_obj1": "grasp"},
    "support": {"type": Topology.STATIC_HOLD, "hand_to_obj1": "grasp"},
    "be held": {"type": Topology.STATIC_HOLD, "hand_to_obj1": "grasp"},
}

# 兜底规则 (如果有些词没收录，默认当作静态抓持或简单二元交互)
DEFAULT_RULE = {"type": Topology.DYADIC_RIGID, "hand_to_obj1": "interact"}

## 参考翻译代码例子
def translate_to_token_states(primitive_name, main_obj, secondary_obj=None):
    """
    根据动词和操作对象，分配物理 Token 状态
    返回: {Token_ID: State_Label}
    """
    # 剥离尖括号和额外说明，比如 "<cut, sth>" 提取出 "cut"
    clean_primitive = primitive_name.strip("<>").split(",")[0].strip()
    
    rule = ACTION_RULE_BASE.get(clean_primitive, DEFAULT_RULE)
    
    token_states = {}
    
    if rule["type"] == Topology.TRIADIC:
        # 三元交互：手拿着主物体(刀)，主物体作用于副物体(面包)
        # 1. 主物体(刀)被手刚性握持 -> [Grasped]
        token_states[main_obj] = "[Grasped]"
        
        # 2. 副物体(面包)正在被主物体切削 -> [Interacting]
        if secondary_obj:
            token_states[secondary_obj] = "[Interacting]"
            
    elif rule["type"] == Topology.DYADIC_RIGID:
        # 二元受迫交互：手直接扭/拉/拧主物体
        # 主物体与手发生相对运动 -> [Interacting]
        token_states[main_obj] = "[Interacting]"
        
    elif rule["type"] == Topology.FLUID_CONTAINER:
        # 容器操作：手握着主物体(杯子)
        # 主物体与手刚性绑定，但姿态受到特定约束(如倾斜) -> [Grasped]
        token_states[main_obj] = "[Grasped]"
        
    elif rule["type"] == Topology.STATIC_HOLD:
        # 静态保持：手死死按住/拿着主物体
        # 主物体与手零相对运动 -> [Grasped]
        token_states[main_obj] = "[Grasped]"
        
    return token_states

# === 测试例子 (对应你截图中的切面包场景) ===
# 右手: "cut", 对象是 "S20005"(刀), 作用于 "002@0094@00004"(面包)
rh_states = translate_to_token_states("cut", main_obj="S20005", secondary_obj="002@0094@00004")
print("右手产生的状态:", rh_states)
# 输出: {'S20005': '[Grasped]', '002@0094@00004': '[Interacting]'}

# 左手: "hold", 对象是 "002@0094@00004"(面包)
lh_states = translate_to_token_states("hold", main_obj="002@0094@00004")
print("左手产生的状态:", lh_states)
# 输出: {'002@0094@00004': '[Grasped]'}

# 这里没有包含所有的动词 ，需要去OakInk2数据集进行提取所有动词，然后针对per primitive task的textual instruction进行部分基于LLM的顺序state编码，最终实现LLM 逻辑链 + 物理状态机双引擎
/hhd4/lizhe/dataset/OakInk2/data/object_affordance/affordance_label.json
宏观规划同时拆分解构在了 text_emb 和 state_labels 中，它们犹如“主谓宾”和“时态”的关系，分工极其明确。为了让你透彻理解这种数据结构的设计精妙之处，我们可以把这 180 帧的生成过程，想象成模型在“读剧本演戏”：1. text_emb: 宏观规划的“最终目的” (The Semantic Goal)本质： 它告诉模型**“你要干什么”**。体现： 比如 CLIP 特征编码了 "Grip the white fruit knife"。它是一个全局的、不随时间（在这 180 帧内）变化的常量。局限： 文本只包含高维语义，它不知道“刀现在离手有多远”，也不知道“手需不需要先张开”。如果没有状态标签，模型看着这句话，可能会试图在第 1 帧就让刀飞到手里。2. state_labels [180, 7]: 宏观规划的“物理执行阶段” (The Physical Phase)本质： 它告诉模型**“你现在正处于完成这个目的的哪一步”**。训练时的体现（你的打包数据）： 在你打包的这 [180, 7] 的张量中，它记录了真实的物理过渡。假设这 180 帧跨越了“触碰”的瞬间，右手（假设是索引 2）的状态序列可能是这样的：帧 0 - 50: 值全是 1 (Reach - 正在靠近)帧 51 - 180: 值全变成了 2 (Grasped - 已经抓住)这也就意味着，在训练数据里，state_labels 直接内置了“动作何时发生相变”的精准物理剧本！ 模型通过 Self-Attention 看到状态从 1 变成 2，就会强行学会在第 51 帧闭合手指并绑定物体。💡 训练与推理（Inference）时的错位与统一理清了上面两点，你就能明白我们上一轮讨论的“动态状态机”在 Inference 时是如何与这个数据结构完美闭环的了：在训练时 (Training)：你喂给模型的是上帝视角。这 [180, 7] 的标签是提前用真实距离算好的，模型在一次 Forward Pass 中，不仅看到了文本“拿刀”，还确切看到了第 51 帧状态变成了“抓取”。模型被强制要求拟合出一条完美的 $X_{target}$ 轨迹。在推理时 (Inference)：你没有未来 180 帧的上帝视角标签。此时，宏观 LLM 规划介入，它给你下发了一个逻辑队列：[Reach] -> [Grasped]。一开始，你手动构造一个形状为 [180, 7] 的 state_labels，把右手那一列全部填满 1 (Reach)。模型看着 text_emb ("拿刀") 和满眼的 1，开始预测速度场，手开始靠近刀。走着走着（比如走到了第 48 帧），你的后台代码算出：手和刀的距离 $< 2$cm 了！此时，你的动态状态机触发，瞬间把接下来喂给模型的 state_labels 张量里的右手列，全部替换成 2 (Grasped)。模型一看状态变成了 2，立刻改变速度场的预测策略，死死锁住手和刀的相对位置。总结text_emb 是宏观规划的语义指南针（指向最终动作）。state_labels 是宏观规划的物理离合器（控制当前处于 Reach、Contact 还是 Release 档位）。

### 阶段三：边界拓展与物理降频 (Boundary Expansion & Downsampling)
**目标：** 解决动作首尾缺失问题，并将数据压缩到 10FPS 且保证物理速度的绝对正确。

1.  **$\pm 60$ 帧边界拓展 (Boundary Padding)：**
    * 对每一个“超级片段”，向其前后各读取并拼接额外的 60 帧原始数据。
    * *注：拓展区域的文本指令置空或继承最近的动作指令，状态根据实际 SDF 距离自动延续。*
2.  **10FPS 降采样 (Decimation)：**
    * 对 3D 坐标和物体位姿进行严格的 `step=3` 均匀抽帧。
    * 对 State Labels 使用 `Max-Pooling` 或逻辑运算抽帧（确保瞬间的 Contact 不被漏掉）。
3.  **计算物理速度 (Kinematic Velocity)：**
    * **必须在降频后计算！**
    * $$V_t = X_t - X_{t-1}$$ 
    * 将计算出的 Velocity 拼接到对应的 3D Position 后方，形成最终的 Entity Tokens。

### 阶段四：5-3-2 混合采样引擎 (Mixed Window Sampling)
**目标：** 按照完美比例在降频后的“超级片段”上滑动截取 $W=200$ 帧的训练张量。

1.  **构建 50% 跨界滑窗 (Boundary-Centric)：**
    * 找到片段内两个 Primitive 的交界帧索引 $Idx_{boundary}$。
    * 在 $[Idx_{boundary} - 100, Idx_{boundary}]$ 范围内随机选取一个起点，截取 200 帧。
2.  **构建 30% 内部滑窗 (Intra-Primitive)：**
    * 找到长度大于 200 帧的单一 Primitive。
    * 在内部随机截取 200 帧，确保动作意图（Text）首尾一致。
3.  **构建 20% 冷启动滑窗 (Cold-Start)：**
    * 提取 Primitive 的第 0 帧（通常是绝对静止或准备状态）。
    * 将该帧的 Position 复制 200 遍，并将所有 Velocity 强行清零。
    * 作为 $X_{hist}$ 的前 20 帧，后 180 帧则是模型需要预测的启动轨迹。

### 阶段五：张量封装与 DataLoader 接口 (Tensor Packaging)
**目标：** 将截取好的 200 帧数据结构化为 PyTorch Dataset 可以直接吐出的 `dict` 格式。

对于每一条 200 帧的样本，打包生成以下字段，并保存为 `.pt` 或 `.npy` 格式：
* `x_hist`: 形状 `[20, 7, Dim]` (前 20 帧的无噪历史特征)
* `x_target`: 形状 `[180, 7, Dim]` (后 180 帧的干净目标特征，供 TorchCFM 加噪)
* `text_emb`: 当前时间步激活的文本指令（CLIP 特征）
* `state_labels`: 形状 `[180, 7]` (仅需要目标帧的物理状态标签)
* `padding_mask`: 形状 `[7]` (布尔值，标记哪些 Object Slots 是空的)
