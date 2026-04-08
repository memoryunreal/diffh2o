 
我们可以用一个**“异构图流匹配（Heterogeneous Graph Flow Matching）”**的概念来统一所有输入。假设一个场景里有：左手、右手、抹布、刀、面包、杯子、桌子。你可以设计一个统一的 Token 化策略（Unified Tokenization）：把一切实体都变成 Token。$Token_{HandL}, Token_{HandR}$$Token_{Obj1}, Token_{Obj2}, \dots$给每个 Token 贴上 状态标签（State Embeddings），这也是从历史帧中提取的：[Grasped] (被抓持，例如：刀)[Free] (自由/静止状态，例如：放在旁边的杯子)[Interacting] (受迫交互中，例如：正在被切的面包，或者正在被擦的桌子)送入 DiT / Flow Transformer：模型通过 Self-Attention 自动学习这些 Token 之间的相互影响。如果文本是 "cut the bread"，NLP 编码器会激活 $Token_{Knife}$ 和 $Token_{Bread}$ 的权重。模型看到 $Token_{Knife}$ 的历史状态是 [Grasped]，它就会把手的轨迹和刀绑定；看到 $Token_{Bread}$ 是 [Interacting]，就会让刀的运动轨迹去拟合面包的几何边界。
 
 Check how the preprocess of the oakink2 process, I need add a addtional preprocess for         
  flow-based model to train. in original oakink2 dataset, like the /hhd4/lizhe/dataset/OakInk2/d  
  ata/program/desc_info/scene_01__A001++seq__083f7a577484ba7929a9__2023-04-27-19-25-24.json and   
  /hhd4/lizhe/dataset/OakInk2/data/program/program_info/scene_01__A001++seq__083f7a577484ba7929a  
  9__2023-04-27-19-25-24.json there have two hands annotation times with interaction method and   
  interacted object list. 



有了这些 JSON，我们的数据预处理流程可以直接大换血，变得极其高效。你的脚本逻辑可以变成这样：Step 1: 读取 program_info 直接生成 Token 和关系图遍历 program_info.json 的每一个片段（Segment）。对于截图里的例子，你的代码可以直接生成这样的当前帧区间逻辑状态：$Token_{RightHand}$ + $Token_{BottleBody}$:关系是 "hold"。映射状态：[Grasped] (被刚性抓持，两者相对位置应该锁定，静止不动)。$Token_{LeftHand}$ + $Token_{Cap}$:关系是 "unscrew"。映射状态：[Interacting] (受迫交互，左手在动，瓶盖随着左手的扭矩在相对瓶身发生螺旋运动)。Step 2: 用物理信息做“微调对齐” (Kinematic Grounding)注意，这一步依然必不可少！为什么？因为 JSON 标注的时间戳 ((1746, 3852), ...) 通常是语义区间。人类在标注时，可能在第 1746 帧时，手才刚刚向瓶盖伸过去（Reach阶段），真正在物理上接触到瓶盖（Contact阶段）可能是在第 1760 帧。如果把 1746-1760 这一段也标成 [Interacting]，模型就会学到“手隔空拧瓶盖”的错误物理规律。所以，物理统计只需做一件事：修剪边界！拿着 JSON 给的 1746 帧，在 398D 物理轨迹中往后查。计算 $Distance(LeftHand, Cap)$。发现距离在第 1762 帧时突降到 0，且后续保持接触。那么：[1746, 1761] 帧： 左手状态为 [Reach]。[1762, 3852] 帧： 左手状态才是 [Interacting]。Step 3: 背景物体降维打击 (Static Labeling)在 1746 到 3852 帧期间，场景里除了瓶盖和瓶身，肯定还有其他物体（比如桌子上的杯子、刀）。因为这些物体的 ID 没有出现在 obj_list 里，我们可以非常放心地直接给它们全体打上 [Static / Free] 的标签。

## 核心规则库
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

## 计算距离表明reach contact release 三个阶段
二、 距离计算的新使命：裁剪出 Reach, Contact, Release结合 JSON 和距离，我们就可以完美且无歧义地切割出你最关心的 Reach，Contact 和 Release 阶段：假设 JSON 告诉我们：左手 lh 和 面包 Bread 在区间 [T_start, T_end] 发生了 "hold"。你需要在这个区间内提取 左手关键点 到 面包表面 的最小距离数组 $D[t]$。通过设置一个接触阈值 $\epsilon$（比如 1-3cm）：Reach (靠近期): 从 $T_{start}$ 开始，此时 $D[t] > \epsilon$ 且距离在不断缩小。打标策略： 此时 面包 的状态依然是 [Static]（它还在桌上没动），左手的状态是 [Reaching_Bread]。模型会学习到手在自由空间中做轨迹规划。Contact / Grasped (接触/交互期):找到距离首次降到阈值以下的帧 $T_{contact}$。在区间 [T_contact, T_release_start] 内，$D[t] \le \epsilon$。打标策略： 此时 面包 变为 [Grasped]。模型开始学习严格的物理绑定和动量传递。Release (释放期):在接近 $T_{end}$ 时，找到距离重新变大 $D[t] > \epsilon$ 的帧 $T_{release}$。打标策略： 面包 重新变回 [Static]，手回到自由空间。


# 最终目标 举个例子--为每个Oakink2的sequence做处理得到类似的时空状态轨迹图
{
  "metadata": {
    "semantic_window": [1431, 4025],
    "interaction_mode": "rh_main"
  },
  "entities": ["lh", "rh", "obj_Knife", "obj_Bread", "obj_BackgroundCup"],
  
  "token_state_trajectories": {
    
    "rh": [
      {"frames": [1431, 1455], "state": "[Reach]", "target": "obj_Knife"},
      {"frames": [1456, 4010], "state": "[Grasped]", "target": "obj_Knife"},
      {"frames": [4011, 4025], "state": "[Release]", "target": "obj_Knife"}
    ],

    "obj_Knife": [
      {"frames": [1431, 1455], "state": "[Static]"}, 
      {"frames": [1456, 1479], "state": "[Grasped]", "master": "rh"}, 
      {"frames": [1480, 3980], "state": "[Grasped_and_Interacting]", "master": "rh", "target": "obj_Bread"},
      {"frames": [3981, 4010], "state": "[Grasped]", "master": "rh"},
      {"frames": [4011, 4025], "state": "[Static]"}
    ],

    "lh": [
      {"frames": [1431, 1448], "state": "[Reach]", "target": "obj_Bread"},
      {"frames": [1449, 4015], "state": "[Grasped]", "target": "obj_Bread"},
      {"frames": [4016, 4025], "state": "[Release]", "target": "obj_Bread"}
    ],

    "obj_Bread": [
      {"frames": [1431, 1448], "state": "[Static]"},
      {"frames": [1449, 1479], "state": "[Grasped]", "master": "lh"},
      {"frames": [1480, 3980], "state": "[Grasped_and_Interacting]", "master": "lh", "tool": "obj_Knife"},
      {"frames": [3981, 4015], "state": "[Grasped]", "master": "lh"},
      {"frames": [4016, 4025], "state": "[Static]"}
    ],

    "obj_BackgroundCup": [
      {"frames": [1431, 4025], "state": "[Static]"}
    ]
  }
}

### 补充如何计算物理距离
二、 推荐方案：“多级包围盒与指尖球形代理” (Hierarchical Spherical Proxy)既然是离线处理，我们绝对不能在物理真实性上妥协，但也必须让代码跑得快。在机器人学和图形学碰撞检测中，最优雅的解法是球形代理（Spherical Proxy）。建议你在预处理脚本中采用以下三步走的计算逻辑：Step 1: 语义窗口粗筛 (Semantic Coarse Filtering)直接利用上一步从 program_info.json 中提取的语义区间（例如 [1431, 4025]）。在这个区间外，直接跳过所有精细计算，判定为 [Static] 或 [Reach]。Step 2: Wrist 级外包围盒剔除 (Wrist-level Bounding Rejection)在语义区间内，先用手腕（Wrist）坐标到物体包围盒（Bounding Box）或中心点算一个粗距离。逻辑： 如果 $Distance(Wrist, Object\_Center) > 30\text{cm}$，说明手还没伸过去，当前帧必定是 [Reach]，直接 continue 到下一帧，免去算手指的开销。Step 3: 指尖球形代理计算 (Fingertip Spherical Proxy) - 【核心精度所在】当手腕靠近物体后，展开精细计算。我们不渲染完整的 Mesh，而是提取 SMPL-X 的 5 个指尖关键点 (Fingertips) 和 10 个手指关节 (Finger Joints)。如何弥补关键点不准的问题？给每个关键点赋予一个“物理半径 $r$”（模拟手指的粗细）。比如，大拇指关键点半径设为 $r \approx 0.012\text{m}$ (1.2厘米)，其他指尖设为 $r \approx 0.008\text{m}$ (0.8厘米)。然后，提取物体表面的点云（Point Cloud，可以提前从物体的 Mesh 上均匀采样 1000 个点，存成一个 Tensor，这个你们之前提到的 BPS 表达其实就能直接用）。最终的接触判定公式 (Contact Metric)：对于每一帧，计算所有 15 个手指关键点 $J_i$ 到物体点云 $P_{obj}$ 的最短距离：$$D_{min} = \min_{i \in \{1..15\}} \left( \min_{p \in P_{obj}} ||J_i - p||_2 - r_i \right)$$如果 $D_{min} \le \epsilon$（$\epsilon$ 可设为 2mm~5mm 的极小容差）： 说明手指皮肤表面已经触碰到了物体！正式进入 [Grasped] 或 [Interacting] 状态。如果 $D_{min} > \epsilon$： 说明手还在微调姿态，依然是 [Reach]。