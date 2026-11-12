# SuperMimic 非人形适配：代码级调研

调研日期 2026-11-09。所有结论都标了证据来源。未标注「README」的都是我实际读到的代码或资产文件。

克隆位置（集群）：`/workspace/wzy_susie/refs/`

| 目录 | 上游 |
|---|---|
| `supermimic_wheelbipe/` | RadiantForever/supermimic_wheelbipe |
| `go2-supermimic/` | Zhengtianjin/go2-supermimic |
| `supermimic_manipulation/` | emoPointer/supermimic_manipulation |
| `Dex-Supermimic/` | CumulusAlpha/Dex-Supermimic |
| `in-hand-rotation-mjlab/` | Msornerrrr/in-hand-rotation-mjlab |
| `supermimic-quadruped/` | Houserz/supermimic-quadruped |

另有一份 `Huangw-NKU/supermimic-go2` 在 `/tmp/adapt/`，只作为对照读过 README。

---

## 1. 汇总表

| 项目 | 机器人 | 实测自由度 | 仿真后端 | 参考信号来源 | SuperMimic 的角色 | 真机 |
|---|---|---|---|---|---|---|
| wheelbipe | 轮腿 V14_2 | 6 主动（4 腿 + 2 轮） | Isaac Lab（训练）+ MuJoCo（模型） | 指令库 | 世界模型 + BPTT | 未见证据 |
| go2-supermimic | Unitree Go2 | 12 腿关节 + freejoint | mjlab | 任务奖励（追逐/闪避） | 世界模型 + BPTT | 无，README 明示仿真 |
| manipulation | ARX5 机械臂 | 7（6 臂 + 1 夹爪） | mjlab | 任务目标状态 | 世界模型 + BPTT | 未见证据 |
| Dex-Supermimic | Wuji Hand 1/2 | 20 执行器（21 铰链） | mjlab | 任务目标（方块/笔） | 世界模型 + BPTT | 仿真 |
| in-hand-rotation | LEAP Hand | 25 执行器 | mjlab | 任务目标（SO(3)/指尖） | 世界模型，与 PPO 并存 | 有真机，但归属 PPO |
| supermimic-quadruped | Dobot Rover X1 | 12 腿关节 + freejoint | mjlab | 动作库 665 条 + 规划器 | 世界模型 + BPTT | 仿真 |

三条一眼可见的规律：

1. **世界模型 + 想象 rollout + BPTT 这个核心循环，六个项目全部保留。**
2. **参考层全部被替换。** 没有一个是原封不动用 G1 那套人形动作库的。
3. **真机证据极弱。** 唯一有真机视频的 LEAP Hand 项目，明确把真机结果归给 PPO。

---

## 2. 逐个项目

### 2.1 supermimic_wheelbipe（轮腿）

**规模**：27 个 Python 文件，6261 行。独立包 `supermimic_wheelbipe/`。

**机器人**：轮腿 V14_2。MuJoCo 模型在 `assets/mujoco/wheelbipe_v14_2.xml`，加载处 `supermimic_wheelbipe/mujoco_env.py:24`、`:68`。

**自由度 6 个**，来自 `spec.py` 的关节分组（`spec.py:73-93`）：

- `act_leg_idx`：4 个主动腿关节，正则 `.*_(front1|rear1)_joint$`
- `act_wheel_idx`：2 个轮关节，正则 `.*_wheel_joint$`

代码里对数量做了硬断言（`spec.py:78-83`）：腿必须是 4 个，轮必须是 2 个，否则直接报错。

**动作缩放**（`constants.py:17-19`）：

```
ACTION_SCALE_LEG = 0.5      # rad / 单位动作
ACTION_SCALE_WHEEL = 10.0   # rad/s / 单位动作
ACTION_CLIP = 3.0
```

**参考层被替换得最彻底**。`command.py:2` 的原话：

> 速度指令库：替代原版的 `MotionLib`（参考动作库）

指令是 4 维 `(vx, vy, wz, height)`（`command.py:14`），程序生成，不落盘：

- `CommandLib.random` 默认 256 条 × 2000 帧（`command.py:60-63`）
- 每段 250 到 750 帧，即 5 到 15 秒（`command.py:63`）
- goal 特征是未来 K 帧指令，`CMD_DIM=4`，维度 `K*4`（`command.py:118-121`）

**训练配置**（`train.py`）：默认 4096 环境（`:19`），学习率衰减 12 万轮（`:39`），老师数据只混前 3000 轮（`:71`）。训练入口走 Isaac Lab（`train.py:16` 导入 `AppLauncher`）。

**老师**：`teacher.py:1` 写的是「wheeled-legged_RL 里已经训练好的 PPO 策略」，观测 35 维、动作 6 维（`teacher.py:13-14`）。不是脚本控制器。

**README 声称的结果**：前进 1.40 m/s，原地偏航 10.21 rad/s，16 条指令评测 0 摔倒、归一化回报 0.983。

---

### 2.2 go2-supermimic（四足）

**规模**：27 个 Python 文件，4859 行。包 `go2_supermimic/`，算法子包 `go2_supermimic/supermimic/`，含 `world_model.py`、`imagine.py`、`buffer.py`、`state.py`、`train.py`、`driver.py`。

**机器人**：`go2_supermimic/robot/go2.xml`。XML 里 12 个具名腿关节（`FL/FR/RL/RR` × `hip/thigh/calf`），加 1 个 `<freejoint/>`。另有 5 个无名铰链条目，是 mjlab 的辅助关节，不计为驱动自由度。

**后端**：mjlab，代码中出现 54 次。

**参考层**：整个仓库没有任何 motion library 相关代码，grep `reference motion|motion clip|retarget` 无结果。学习目标由任务定义，README 列出三种任务：

- `Go2-Chase-Flat`
- `Go2-Chase-Obstacles`
- `Go2-Chase-Terrain`（README:34）

地形任务的参数写在 README:37：

> On terrain, steps grow from 2 to 10 cm and box edges from 2 to 8 cm

地形观测是 11×7 的高度扫描（README:34）。**这条对上台阶任务最有参考价值**：台阶高度从 2 cm 起步做课程。

**真机**：README 明确是仿真，无硬件部署证据。

---

### 2.3 supermimic_manipulation（ARX5 机械臂）

**规模**：131 个 Python 文件，18573 行。包 `src/clab/`。

**机器人**：`src/clab/robots/arx5/assets/arx5.xml`。实测 7 个铰链（`joint1` 到 `joint6` 加夹爪），8 个执行器。

**SuperMimic 的位置**：`src/clab/algorithms/supermimic/`，含 `train.py`、`runner.py`、`agent.py`、`networks.py`、`buffer.py`。它被做成了算法插件，接入 `clab` 的任务框架。

**任务**：`src/clab/tasks/` 下有 `lemon`、`box_in_bin`、`mug_hang`、`drawer` 四个任务目录。

注意：需求里提到的 Square 任务，在这份代码里没有对应目录。README 和代码只到四个任务加一个 `pregrasp.py`。

**参考层**：没有动作库。任务用 mocap 物体定义目标状态，例如 `tasks/drawer/props.py:66` 的 `mocap=True`，`tasks/drawer/resets.py:33` 的 `write_mocap_pose_to_sim`。

**真机**：未找到硬件证据。

---

### 2.4 Dex-Supermimic（Wuji Hand）

**规模**：89 个 Python 文件，13455 行。两处包：`src/dex_supermimic/`（任务）和 `src/dex_supermimic/rl/dex_supermimic/`（算法），后者含 `mdp.py`、`networks.py`、`runner.py`、`cli.py`。

**机器人**：`packages/wuji-description/.../right_simplified.xml`。实测 21 个铰链、20 个执行器。

**任务**：`src/dex_supermimic/tasks/` 下 `dex_reorient`、`pen_spin`。

**参考层**：任务目标。`tasks/dex_reorient/mdp/event_impl/reset.py:37` 是对 mocap 根的重置逻辑。

**权重**：README:103 说明 `_8000` 后缀表示训练 8000 轮，仓库提供 `model_hand1_reorient_8000.pt` 等检查点。

**真机**：无。

---

### 2.5 in-hand-rotation-mjlab（LEAP Hand）

**规模**：71 个 Python 文件，12961 行。包 `src/in_hand_rotation_mjlab/`，含 `world_model/`、`robots/leap_hand/`、`tasks/`。

**机器人**：`robots/leap_hand/xmls/left_hand_custom.xml`。实测 25 个铰链、25 个执行器。README 说机器人模型包含在真手上辨识出的伺服模型。

**任务**：`tasks/fingertip_rotate`、`tasks/hand_cube`。

**这一份的方法划分最清楚**。README:3：

> Every task trains with **PPO**, and the yaw and fingertip tasks also train with a **learned differentiable world model** in the style of SuperMimic.

也就是说，SuperMimic 风格的世界模型是**附加的第二条训练路径**，PPO 是基线。README:19 有一张表，分列 Task / PPO / World model / Real robot。

**真机**：有。README:7-8 展示了真手旋转视频，但标注为「PPO, the released checkpoint」。**真机结果不是世界模型路线跑出来的。**

---

### 2.6 supermimic-quadruped（Dobot Rover X1）

**规模**：97 个 Python 文件，12705 行。

**这一份是唯一改造上游结构的**：

```
src/supermimic/          # 与上游同构：training/ world_model/ imagination/ policy/ envs/ data/ eval/
src/supermimic/robots/   # 新增：spec.py g1.py rover.py
src/supermimic/kinematics/robot_fk.py   # 通用 FK
src/supermimic/planner/  # 新增：rover.py rover_skills.py 等
configs/robots/          # go1.yaml rover.yaml
```

**机器人**：`robots/assets/rover/dobot.xml`，12 个腿关节（`joint_{front,rear}_{left,right}_{abad,thigh_pitch,calf_pitch}`），加 1 个 `joint_fixed_world` 和 1 个 freejoint。

`robots/rover.py` 的文档字符串写明了每个调参值的出处（Dobot 官方仓库、本地 RL 行走策略、Go2 的 SuperMimic 跟踪实验、mjlab 的 Go1 常量），这是我在六个仓库里见到的最严格的溯源写法。

**参考层保留**：仍然是动作库。`training/loop.py:15` 导入 `motion_library`，`imagination/rollout.py:10` 使用 `MotionLibrary`。

README:99 说明动作库规模：665 条片段、87 分钟（含镜像）。

**新增了在线规划器**，README 的说法是 SONIC 风格的 kinodynamic planner，把摇杆输入实时变成参考动作，每 0.1 秒重规划。代码在 `planner/rover.py` 和 `planner/rover_skills.py`。

**README 声称的跟踪结果**：之字形 0.89，原地转 0.84，跑 0.67，躺下起立 0.99（归一化回报，1.0 为满分）。

**真机**：仿真。

---

## 3. 横向对比

### 3.1 代码形态：五份重写，一份改造

| 形态 | 数量 | 仓库 | 代码量 |
|---|---|---|---|
| 独立重写 | 5 | wheelbipe、go2、manipulation、dex、in-hand | 4.9k 到 18.6k 行 |
| 改造上游 | 1 | quadruped | 12.7k 行，新增 2 个目录 |

重写派的共同点是包名直接带机器人名（`go2_supermimic`、`dex_supermimic`、`in_hand_rotation_mjlab`、`supermimic_wheelbipe`）。改造派把机器人差异收敛到 `robots/` 和 `configs/`。

### 3.2 参考层的三种替代方案

| 方案 | 代表 | 参考信号 | 是否需要离线数据 |
|---|---|---|---|
| 动作库 + 在线规划器 | quadruped | 665 条重定向动作 | 是 |
| 程序生成指令库 | wheelbipe | `(vx, vy, wz, height)`，256×2000 帧 | 否 |
| 任务目标 | go2、manipulation、dex、in-hand | 任务奖励 + mocap 目标 | 否 |

### 3.3 保留与替换的边界

保留（六份一致）：

- 世界模型：MLP 动力学 + 终止头
- 想象 rollout 与 BPTT 策略更新
- 基于回报的重置采样 / 指令调度
- 从仿真器收集 transition 的 buffer

替换（每份都改）：

- 机器人资产与关节分组
- 观测与动作空间
- 参考信号与奖励
- 终止条件

---

## 4. 对火星车的可迁移结论

| 来源 | 可直接借用 | 文件 |
|---|---|---|
| quadruped | 机器人规格抽象、通用 FK、多机器人配置 | `robots/spec.py`、`kinematics/robot_fk.py`、`configs/robots/*.yaml` |
| wheelbipe | 指令库、腿/轮混合动作缩放、关节分组正则 | `command.py`、`constants.py`、`spec.py` |
| go2 | 台阶地形课程（2 cm 起）、高度扫描观测 | `env/terrains.py`、README 地形参数 |
| in-hand | 世界模型与 PPO 双轨对比的评测组织方式 | README 对比表、`world_model/` |

三条判断：

1. **参考层必须换，世界模型可以留。** 六个项目无一例外。
2. **上台阶最适合走「任务目标 + 地形课程」路线**，因为 go2 已经证明了台阶地形可以按 2 到 10 cm 逐步加难，不需要参考动作。
3. **如果坚持用参考动作，quadruped 的规划器模式是唯一有代码先例的做法**：动作库离线，规划器在线生成，跟踪策略不变。

---

## 5. 未核实与局限

- 六个仓库我都没有实际运行，只做了静态阅读。README 里的效果数字属于仓库自述。
- `supermimic_manipulation` 首次 clone 得到空工作区，改用 `-b main` 才成功，说明其默认分支配置有坑。
- 需求里提到的 Square 任务，在 manipulation 代码中没有找到。
- 真机部署证据只有 LEAP Hand 一份，且归属 PPO，不是世界模型路线。
- 我没有逐一阅读每个项目的训练超参数，只提取了与迁移决策相关的部分。

---

## 6. 复现核实的命令

```bash
# 关节与执行器计数
python3 /tmp/dof.py

# 指令库替代动作库的证据
sed -n '1,20p' ~/refs/supermimic_wheelbipe/supermimic_wheelbipe/command.py

# 机器人规格抽象
sed -n '1,60p' ~/refs/supermimic-quadruped/src/supermimic/robots/spec.py

# 通用 FK
grep -n "^class\|^    def " ~/refs/supermimic-quadruped/src/supermimic/kinematics/robot_fk.py

# 方法划分（LEAP）
sed -n '1,8p' ~/refs/in-hand-rotation-mjlab/README.md
```

---

## 7. 六轮 rocker-bogie 平台参考（补充调研 2026-11-09）

这一节不是 SuperMimic 适配，是同平台的开源工程。目的只有两个：找可复用的六轮模型，核实差动机构的建模问题。

### 7.1 访问与内容核实

| 项目 | 仓库 | 访问结果 | 实际内容 | 对你有用吗 |
|---|---|---|---|---|
| NASA JPL Open Source Rover | `nasa-jpl/open-source-rover` | 已克隆，617 个文件 | 机械、电路、PCB、零件表、网页遥控 | 机械与选型参考，**没有 URDF 也没有 ROS** |
| Sawppy | `Roger-random/Sawppy_Rover` | 已克隆，263 个文件 | 3D 打印 STL 与装配文档 | 差动零件实体（见 7.3） |
| ToolBoxRobotics | `ToolBoxRobotics/mars_rover_ws` | 已克隆 | ROS 2 工作空间，含 `rover_description` | **有一份六轮摇臂 xacro**，见 7.2 |
| EPFLXplore ERC_NAV | `EPFLXplore/ERC_NAV` | 克隆失败，提示需要用户名，仓库私有或已改名 | 无 | 无法评估 |
| Perseverance 状态估计 | sharwinpatil.info 上的项目 | 未找到公开仓库 | 无 | 无法评估 |

对照组：NASA 那个仓库的名字容易让人以为是完整软件栈，实际 617 个文件里没有任何 `.urdf`、`.py` 之外的可运行仿真，主要目录是 `mechanical/`、`electrical/`、`parts_list/`。

### 7.2 唯一可用的六轮模型

`mars_rover_ws` 里有两种车：当前在用的四轮阿克曼（`src/rover_description/urdf/chassis.xacro`，轮半径 0.075 m、轮距 0.46 m），以及一份六轮摇臂参考模板：

```
src/rover_description/reference/opportunity_style_template.urdf.xacro
```

该文件的头注释说明它是「机遇号风格」的差分摇臂悬架，并明确写了它**没有接入当前模型**，只是迁移用的起点。

它的运动学拓扑（文件内 `:256`、`:282`、`:319`、`:355`）：

```
base_link
 ├─ left_differential_joint  (revolute)  → left_differential_link
 │    ├─ front_left_steering_joint (revolute) → front_left_wheel_joint (continuous)
 │    └─ left_bogie_joint (revolute)          → …后轮
 └─ 右侧对称
```

对你的建模直接有用的一点：**它和你现在的 URDF 是同一个做法**。左右差动各是一个从车体引出的独立转动副，没有把两者之间的差动杆建成闭环。

### 7.3 差动闭环：你的模型可能有这个缺口

Sawppy 的 STL 目录里有一整套差动零件，文件名本身就是证据：

```
STL/DiffBrace.stl  STL/DiffEnd.stl  STL/DiffLink.stl
STL/DiffLower.stl  STL/DiffUpper.stl
```

物理上，经典 rocker-bogie 的左右摇臂通过差动杆相连：一侧抬起时另一侧按比例下降，两摇臂角度被约束成一个关系。这在机构上是闭环运动链。

而 URDF 只能表达树。上面两份六轮模型都把差动拆成两个独立转动副，我们的 `mars_rover.urdf` 也一样（`left_rocker_joint` 与 `right_rocker_joint` 都直接挂在 `base_link` 上，全车 13 个可动刚体，没有任何差动连杆）。

**这对你的车意味着什么，取决于一个事实：**

| 实车结构 | 模型是否成立 | 处理方式 |
|---|---|---|
| 摇臂独立驱动，无差动杆 | 成立 | 现状即可 |
| 有被动差动杆 | 不成立 | MuJoCo 用 `equality` 关节耦合补约束 |

你 URDF 里摇臂关节的力矩上限是 100 N·m，是全车最大的，这更像是**主动摇臂**。如果实车确实是两个电机独立驱动，你现在这份模型是对的，可以继续。

如果实车上有一根连接左右摇臂的差动杆，仿真里必须补约束，否则策略会学到一个物理上不可能的动作：两个摇臂反向错开而无视差动比。

验证方法（不用拆车）：把车架空，手动抬起一侧摇臂，看另一侧是否反向运动。反向就是有差动。

### 7.4 可以借用的东西

| 来源 | 借什么 | 路径 |
|---|---|---|
| `mars_rover_ws` | 六轮摇臂的 xacro 拓扑与关节命名习惯 | `src/rover_description/reference/opportunity_style_template.urdf.xacro` |
| `mars_rover_ws` | ROS 2 导航栈（Nav2 + EKF + 轮速里程计），部署阶段用 | `src/rover_*` |
| `Sawppy_Rover` | 差动零件几何，若需要补闭环约束可参考安装尺寸 | `STL/Diff*.stl` |
| `nasa-jpl/open-source-rover` | 底盘选型、转向舵机布局、零件清单 | `mechanical/`、`parts_list/` |

### 7.5 这一轮的局限

- ERC_NAV 和 Perseverance 项目都没拿到代码，无法评价。
- 用户资料里「Sawppy 指出闭环运动链无法用普通 URDF 表达」这句话，我在这份仓库的 README 里**没有找到原文**（只找到指向维基的 rocker-bogie 链接）。结论本身与机构事实相符，但出处需要另行标注。
- 三个仓库都没有实际运行。

