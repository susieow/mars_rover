# 20 cm 爬楼实验经验

更新：2026-11-12。负责人 wzy。本文件是交接的统一经验入口，旧方案和结果 Markdown 已合并到这里。清理前证据版本 `c956d58cac626bef70a44c2128a40f8d23995437`；原始 JSON/JSONL、轨迹、视频、模型及冻结源码仍保留。参考资料另见 [SuperMimic 调研](docs/references/supermimic_nonhumanoid_code_survey.md)。

## 当前结论

最终测试版本为 **台阶摩擦 4/6/8 × 两种触发 × 两种 ±20° 顺序 × 六轮基础轮速 3～15 rad/s**。已完成 112 条，其中 68 条曾局部越阶。用户把所有曾越阶的记录作为效果最好的参考组，观察其效果相近，本次不再强行选出单一“冠军”。完整爬楼、过平台、下楼并停稳的合格成功样本仍为 **0**；这些批次没有完成完整路线评估，不能报成完整路线的 0/112 成功率。

只继续 20 cm；过去的 4/10/12/15 cm 等仅作历史经验。当前任务是整理交接，没有新增动力学实验，也没有补齐缺测参数。

## 已测范围：不要重复相同组合

| 摩擦 | 触发 | 已测轮速 rad/s | 条数 | 曾局部越阶 | 末 1 秒仍保持越阶等级 | 最大穿透 mm |
|---:|---|---|---:|---:|---:|---:|
| 4 | 首次接触 | 6～15 | 20 | 10 | 7 | 65.16 |
| 4 | 卡住后 | — | 0 | — | — | — |
| 6 | 首次接触 | 6～15 | 20 | 11 | 8 | 76.05 |
| 6 | 卡住后 | 3～15 | 26 | 17 | 16 | 41.00 |
| 8 | 首次接触 | 6～15 | 20 | 15 | 14 | 80.49 |
| 8 | 卡住后 | 3～15 | 26 | 15 | 14 | 80.43 |

每档包含 SN、SP 两种顺序。“末 1 秒保持”仍是几何条件，不保证接触力支撑或最终稳定。

未测试的 44 个组合：摩擦 4 的“卡住后触发”3～15 rad/s 两方向（26 个）；摩擦 4/6/8 的“首次接触触发”3～5 rad/s 两方向（18 个）。未测不等于失败，也不等于要求队友自动补跑。

### 参数含义与时序

- 起点为地面、仿真 t=0；距首阶前轮外缘约 8 m。摩擦从起步生效，只修改台阶的滑动摩擦，不改地面/轮胎及其他接触参数。
- 六轮同一档基础目标，启动变化率 30 rad/s²；保留航向反馈与前后轮转向。实际目标会做左右修正，不能把基础档位当作六轮逐时实际速度，也不能认为转向已锁定。
- `contact`：任一前轮与上行台阶首次产生正法向力即触发，下一 1 ms 物理步开始更新摇臂目标。
- `stall`：前轮已经接触，首次达到连续 2 秒车身向前净位移小于 0.02 m 的时刻触发；不是统一固定在 22 s。
- 左右摇臂同向。SP 的一个 8 秒周期：0～1 s 从 0 到 +20°；1～2 s 保持；2～4 s 到 −20°；4～5 s 保持；5～6 s 回 0；6～8 s 保持。SN 把全部角度反号。目标变化速度 20°/s，3 周期后回中立。
- 物理/控制步长均 1 ms。触发后观察 30 s，未触发最多 120 s。审核违规只记事件继续观察；非有限数值作为技术失败。
- 车轮/转向/摇臂执行器力矩限制沿用 50/40/100 N·m。没有使用早期“无限力矩”诊断模型。

### 局部有效经验与完整成功

“曾越阶”沿用保存结果中的几何判断：双后轮轮心都越过某阶前棱 5 mm，近似轮底高于该阶高度减 10 mm，连续至少 100 ms。它不要求当时两轮都有可靠支撑；应同时检查实际接触力、最大穿透、保持等级、姿态和后续回落。

最大穿透是简化碰撞几何的重叠深度，不能直接解释成真实轮胎压缩量。20 mm 是现有审核阈值，尚无硬件标定依据。当前高轮速/高摩擦局部进展常伴随速度违规或较大穿透，不能仅凭越阶标记认定真实可执行。

所有 112 条均未登顶，保留过程原因。完整成功还需要上楼、平台支撑、下楼、全部轮回到地面并正常停稳，保存 state/action/实际力矩与接触，并独立重放及检查视频后再入库。这里的“局部有效”用于继续研究，不代表已接纳为成功训练数据。

## 检索已试参数

机器索引 `docs/experiment_index.jsonl` 当前收录 1198 条带来源的记录，包含最终版本、近期历史诊断、早期 768 次候选评估、29 条较长历史诊断和 1 秒附加验证。**这是证据条目数，不是独立策略数或物理试验总次数**；对照、复测、技术失败和重复实际控制序列须分别理解。早期原始交付中只有汇总而无逐例文件的项目，以下面的历史摘要为准，不伪造精确参数记录。

```bash
cd /home/agilex/wzy/rover
.venv/bin/python tools/find_experiment.py --friction 6 --trigger stall --speed 6 --direction SP
.venv/bin/python tools/find_experiment.py --crossed
.venv/bin/python tools/find_experiment.py --id T22_WMR_RSP_dt1000us --json
.venv/bin/python tools/find_experiment.py --text A20_C20 --scope history --json
```

相同参数必须同时核对初态/检查点、模型与源码哈希、物理步长、触发、动作时序和停止方式；相同 ID 在不同批次不一定同一实验。精确匹配可直接引用已有证据。新模型、更小步长等有明确目的的复核应标成对照，不能仅凭“同轮速曾失败”禁止有实质变化的验证。

索引保留原报错英文、视频绝对路径、原始结果位置和哈希。历史视频不再出现在前端，按索引路径在 AGX 自行查看。早期媒体中一部分是冻结 backend 重建，原记录未保存全部逐步 controls；不能冒称每一帧均来自原始逐步保存。对应来源限制仍在原清单中。

## 已淘汰路线及原因

| 路线 | 实际执行与结果 | 淘汰/不复用原因 | 原始证据/视频位置 |
|---|---|---|---|
| 早期低台阶、DeepSeek 交付与 4 cm 种子 | 旧版本有低台阶“成功”表述，4 cm 数据保留在 regression；无 20 cm 完整成功 | 几何、目标与审核版本不符，不能用来代表当前 20 cm 能力；不再测试其他高度 | `datasets/45deg_v3/`；旧交付 Markdown 可从清理前 Git 提取 |
| 15/20 cm 的 2 秒短序列搜索 | 768 次评估；四工况各 192 次；704 个按工况去重目标序列，跨四组数值去重 546；局部越首阶 0 | 有的分数奖励后退后回墙，低轮速短动作没形成后轮减载/越阶；不能继续原样重复刷相同候选 | `review_v4/dynamic_search15/`、`dynamic_search20/`；视频 `.../h15/`、`.../h20/` |
| 29 条较长历史诊断与附加 1 秒保持 | 15 cm 16 条、20 cm 13 条，无完整成功；`h20_near_contact_tail_hold_1s` 接父候选 round2_candidate13 末状态，仍未越阶 | 主动支撑/锚定等未形成可执行上楼；部分历史 EMA 字段有引用保存问题。1 秒延续独立计数，不与 768/29 混计 | `review_v4/`；`docs/media_manifest.json` 的 historical/additional |
| 原始高轮速矩阵 | 6 条平地校验、36 条楼梯主试验、4 条分配对照；无登顶 | 35/36 主试验因速度瞬态截断；36 参数仅 15 条不同实际控制，不能解释成所有爬楼目标都充分执行失败 | `runs/speed20_15steps_20261111/`；视频 `.../speed20_20261111/` |
| 超速记录不停的三步长诊断 | 6 参数组 × 1/0.5/0.25 ms＝18 条；A40_C40 另 3 条 | A03_C03 从约 22 s 卡住；高接近速度仍有角限位、穿透、姿态等问题；缩小步长没有直接得到成功 | `runs/overspeed20_20261111/`、`overspeed20_A40_C40_20261111/` |
| A20_C20 取消力矩上限的小实验 | 3.3975 s 前与父轨迹一致，之后去掉力矩限幅；完整有效诊断到 30 s | 轮/转向/摇臂峰值约 960.52/715.09/893.91 N·m，翻覆、穿透和限位异常；仅演示后续，不用于后续硬件约束实验。中间技术失败/限幅未完全去除的尝试也保留 | `runs/unrestricted20_a20c20_20261111/`；`docs/unrestricted20_a20c20_20261111.json` |
| 22 s 卡点轮组加速＋±10°摇臂 | 35 新动作＋5 对照＝40 条；0 越首阶，37 条过程无违规 | 3→6 rad/s 的不同单/双/三轮组组合与小幅摆动未解除首阶阻挡 | `runs/recovery20_t22_20261111/`；视频同名目录 |
| 22 s 卡点 ±20° 摇臂 | 32 新记录，另引用 8 条旧中立对照；T22_WMR_RSP 曾越首阶但末段回落且有违规 | 有局部进展但没有登顶；固定 22 s 检查点和轮组加速方案已由地面起步、事件触发版本替代 | `runs/recovery20_t22_amp20_20261111/` |
| −30/−40/−50/−60/−70/−80/−90°到 +20°非对称范围 | 计划 224；139 完成、3 中断、82 未启动；完成者 0 越首阶 | 更大摆幅没有带来本批越阶；用户已叫停，不自动续跑。未完成的角度不能标成已失败 | `runs/recovery20_t22_asymmetric_20261111/`；`docs/recovery20_asymmetric_stopped_20261111.json` |
| 只锁后轮转向、前后轮都锁 | 2 条、各从 22 s 续跑 30 s，过程无违规但未越首阶 | 没有解决阻挡；使用等式约束，反力不受真实锁止装置电机限制，不能当实车锁止验证 | `runs/rear_steer_lock_20261111/`、`all_steer_lock_20261111/` |
| T22 检查点后才增摩擦至 6/8 | 先正后负两条均因穿透提前停止；先负后正：摩擦 6 有局部越阶，摩擦 8 原版提前停止，另有不停的继续观察 | 摩擦只在 22 s 切换，不能与起步即使用摩擦 6/8 等同；提前停止看不到后续。由当前地面恒速版本替代 | `runs/friction20_t22*`；对应 JSONL 登记 |
| 前/中轮接触台阶才给该组 6 rad/s | 1 条；同组任一侧接触就两侧加速，另一档 3 rad/s；曾局部越阶、后轮转向违规 | 接触频繁抖动，前/中组切换 938/153 次；用户选择取消这套接触切速，改从起步恒定基础档位 | `runs/friction20_contact_wheels_mu6_20261112/` |

`.../` 视频根目录统一为 `/home/agilex/wzy/videos/review_high_stairs/`。当前最终版本中未越阶的条件也保留在前端，供比较；“淘汰”表示旧方案不再作为当前入口，不表示车辆物理上绝对不可能爬上。

### 数值和模型经验

- URDF 摇臂角速度上限为 1.57 rad/s，角度范围约 −90°～+20°；它是上限而非固定运动速度。六轮均主动自转，中轮没有转向自由度。轮速上限 52.359878 rad/s（约 500 rpm），轮电机额定力矩字段 50 N·m；不等于实车能在最高转速持续输出最大力矩。
- 速度审核使用实际 `qvel`，阈值为 URDF 上限 +0.01 rad/s。例 A03_C06 在 1 ms 下 t=17.751 s 首次左摇臂超速，峰值约 1.7018 rad/s、持续约 15 ms；0.5/0.25 ms 对照也出现并恢复。硬件来源尚未核实，不凭空放宽允许百分比或时长。
- A03_C03 的 180 s 是观察上限，约 22 s 就已卡住；从检查点继续应保留真实父轨迹与检查点，不能声称从地面开始执行续跑动作。
- 当前模型：轮半径 0.163 m；质量数值约 126.746 kg，注释写 122.5 kg；地球重力 9.81 m/s²；接触时间常数 0.004 s；内部自碰撞关闭；轮胎碰撞全宽 0.2 m、视觉约 0.103 m。均非已标定实车事实，本次未改模型。
- 20 cm 静力辅助分析中，加入原关节摩擦后 6 个抽样中立状态有 5 个存在额定静态平衡解；5/6 不是爬楼成功率，也不证明动作动态可达。权威结论 `review_v4/quasistatic20_final_conclusions.json`，不要复用被 superseded 的无摩擦旧结论。

## 当前脚本入口与复现

| 触发/摩擦/速度 | 实验脚本（`scripts/rover/`） | 记录目录（`runs/`） |
|---|---|---|
| 卡住，6，3～6 | `ground_mu6_constant_wheels_20261112.py` | `ground_mu6_constant_wheels_20261112/` |
| 卡住，8，3～6 | `ground_mu8_constant_wheels_20261112.py` | `ground_mu8_constant_wheels_20261112/` |
| 卡住，6，7～15 | `ground_mu6_v7to15_20261112.py` | `ground_mu6_v7to15_20261112/` |
| 卡住，8，7～15 | `ground_mu8_v7to15_20261112.py` | `ground_mu8_v7to15_20261112/` |
| 接触，4，6～15 | `ground_mu4_contact_v6to15_20261112.py` | `ground_mu4_contact_v6to15_20261112/` |
| 接触，6，6～15 | `ground_mu6_contact_v6to15_20261112.py` | `ground_mu6_contact_v6to15_20261112/` |
| 接触，8，6～15 | `ground_mu8_contact_v6to15_20261112.py` | `ground_mu8_contact_v6to15_20261112/` |

参数接口为 `--speed <整数档> --direction SN|SP`。原脚本会拒绝已存在候选目录，不要删除旧目录强行重跑。媒体/核验入口保留对应 `media`/`verify` 脚本。重放或新实验前先查索引，修改运行目录并记录新版本，不自动新增物理实验。

从 AGX 取数据，不能只克隆 Git。环境、服务和协作地址见 README。冻结旧源码仍有原绝对路径，供追溯；删除的活动脚本可从 Git `c956d58cac626bef70a44c2128a40f8d23995437` 恢复，清单 `docs/cleanup_manifest_20261112.json` 记录路径与哈希。SuperMimic 参考原文未改。

## 最终版本逐条记录

SN＝先负后正；SP＝先正后负。全部曲线、首次接触、触发时间和英文违规原因在前端及索引可查。表内“最高/末段”等级仅指几何等级。

| 编号 | 摩擦 | 触发 | 速度 rad/s | 顺序 | 阶段结果 | 最大穿透 mm |
|---|---:|---|---:|---|---|---:|
| `ground_mu4_contact_v6_rocker_SN` | 4 | 接触 | 6 | SN | 未越阶 | 8.77 |
| `ground_mu4_contact_v6_rocker_SP` | 4 | 接触 | 6 | SP | 未越阶 | 7.09 |
| `ground_mu4_contact_v7_rocker_SN` | 4 | 接触 | 7 | SN | 曾越阶（最高1，末段1） | 10.50 |
| `ground_mu4_contact_v7_rocker_SP` | 4 | 接触 | 7 | SP | 未越阶 | 9.16 |
| `ground_mu4_contact_v8_rocker_SN` | 4 | 接触 | 8 | SN | 未越阶 | 23.21 |
| `ground_mu4_contact_v8_rocker_SP` | 4 | 接触 | 8 | SP | 未越阶 | 25.73 |
| `ground_mu4_contact_v9_rocker_SN` | 4 | 接触 | 9 | SN | 曾越阶（最高1，末段1） | 11.05 |
| `ground_mu4_contact_v9_rocker_SP` | 4 | 接触 | 9 | SP | 曾越阶（最高2，末段2） | 14.48 |
| `ground_mu4_contact_v10_rocker_SN` | 4 | 接触 | 10 | SN | 未越阶 | 34.05 |
| `ground_mu4_contact_v10_rocker_SP` | 4 | 接触 | 10 | SP | 未越阶 | 12.84 |
| `ground_mu4_contact_v11_rocker_SN` | 4 | 接触 | 11 | SN | 未越阶 | 23.38 |
| `ground_mu4_contact_v11_rocker_SP` | 4 | 接触 | 11 | SP | 曾越阶（最高1，末段0） | 19.63 |
| `ground_mu4_contact_v12_rocker_SN` | 4 | 接触 | 12 | SN | 未越阶 | 13.32 |
| `ground_mu4_contact_v12_rocker_SP` | 4 | 接触 | 12 | SP | 未越阶 | 39.55 |
| `ground_mu4_contact_v13_rocker_SN` | 4 | 接触 | 13 | SN | 曾越阶（最高1，末段1） | 65.16 |
| `ground_mu4_contact_v13_rocker_SP` | 4 | 接触 | 13 | SP | 曾越阶（最高1，末段1） | 12.55 |
| `ground_mu4_contact_v14_rocker_SN` | 4 | 接触 | 14 | SN | 曾越阶（最高2，末段2） | 15.45 |
| `ground_mu4_contact_v14_rocker_SP` | 4 | 接触 | 14 | SP | 曾越阶（最高1，末段0） | 16.05 |
| `ground_mu4_contact_v15_rocker_SN` | 4 | 接触 | 15 | SN | 曾越阶（最高2，末段2） | 21.88 |
| `ground_mu4_contact_v15_rocker_SP` | 4 | 接触 | 15 | SP | 曾越阶（最高2，末段0） | 14.60 |
| `ground_mu6_contact_v6_rocker_SN` | 6 | 接触 | 6 | SN | 曾越阶（最高1，末段1） | 21.00 |
| `ground_mu6_contact_v6_rocker_SP` | 6 | 接触 | 6 | SP | 曾越阶（最高1，末段0） | 17.63 |
| `ground_mu6_contact_v7_rocker_SN` | 6 | 接触 | 7 | SN | 未越阶 | 22.58 |
| `ground_mu6_contact_v7_rocker_SP` | 6 | 接触 | 7 | SP | 未越阶 | 24.41 |
| `ground_mu6_contact_v8_rocker_SN` | 6 | 接触 | 8 | SN | 未越阶 | 32.74 |
| `ground_mu6_contact_v8_rocker_SP` | 6 | 接触 | 8 | SP | 曾越阶（最高3，末段3） | 28.76 |
| `ground_mu6_contact_v9_rocker_SN` | 6 | 接触 | 9 | SN | 曾越阶（最高2，末段2） | 25.55 |
| `ground_mu6_contact_v9_rocker_SP` | 6 | 接触 | 9 | SP | 未越阶 | 30.23 |
| `ground_mu6_contact_v10_rocker_SN` | 6 | 接触 | 10 | SN | 未越阶 | 28.42 |
| `ground_mu6_contact_v10_rocker_SP` | 6 | 接触 | 10 | SP | 曾越阶（最高1，末段0） | 32.80 |
| `ground_mu6_contact_v11_rocker_SN` | 6 | 接触 | 11 | SN | 曾越阶（最高1，末段1） | 29.53 |
| `ground_mu6_contact_v11_rocker_SP` | 6 | 接触 | 11 | SP | 未越阶 | 29.52 |
| `ground_mu6_contact_v12_rocker_SN` | 6 | 接触 | 12 | SN | 未越阶 | 31.92 |
| `ground_mu6_contact_v12_rocker_SP` | 6 | 接触 | 12 | SP | 曾越阶（最高1，末段1） | 31.91 |
| `ground_mu6_contact_v13_rocker_SN` | 6 | 接触 | 13 | SN | 未越阶 | 34.28 |
| `ground_mu6_contact_v13_rocker_SP` | 6 | 接触 | 13 | SP | 曾越阶（最高2，末段2） | 34.27 |
| `ground_mu6_contact_v14_rocker_SN` | 6 | 接触 | 14 | SN | 曾越阶（最高2，末段2） | 36.87 |
| `ground_mu6_contact_v14_rocker_SP` | 6 | 接触 | 14 | SP | 未越阶 | 76.05 |
| `ground_mu6_contact_v15_rocker_SN` | 6 | 接触 | 15 | SN | 曾越阶（最高1，末段1） | 39.23 |
| `ground_mu6_contact_v15_rocker_SP` | 6 | 接触 | 15 | SP | 曾越阶（最高1，末段0） | 39.22 |
| `ground_mu6_v3_rocker_SN` | 6 | 卡住 | 3 | SN | 未越阶 | 22.52 |
| `ground_mu6_v3_rocker_SP` | 6 | 卡住 | 3 | SP | 未越阶 | 18.04 |
| `ground_mu6_v4_rocker_SN` | 6 | 卡住 | 4 | SN | 未越阶 | 14.69 |
| `ground_mu6_v4_rocker_SP` | 6 | 卡住 | 4 | SP | 未越阶 | 17.11 |
| `ground_mu6_v5_rocker_SN` | 6 | 卡住 | 5 | SN | 未越阶 | 17.21 |
| `ground_mu6_v5_rocker_SP` | 6 | 卡住 | 5 | SP | 未越阶 | 17.21 |
| `ground_mu6_v6_rocker_SN` | 6 | 卡住 | 6 | SN | 未越阶 | 19.12 |
| `ground_mu6_v6_rocker_SP` | 6 | 卡住 | 6 | SP | 曾越阶（最高2，末段2） | 19.12 |
| `ground_mu6_high_v7_rocker_SN` | 6 | 卡住 | 7 | SN | 未越阶 | 22.54 |
| `ground_mu6_high_v7_rocker_SP` | 6 | 卡住 | 7 | SP | 未越阶 | 22.54 |
| `ground_mu6_high_v8_rocker_SN` | 6 | 卡住 | 8 | SN | 曾越阶（最高1，末段1） | 23.44 |
| `ground_mu6_high_v8_rocker_SP` | 6 | 卡住 | 8 | SP | 曾越阶（最高2，末段2） | 23.44 |
| `ground_mu6_high_v9_rocker_SN` | 6 | 卡住 | 9 | SN | 曾越阶（最高2，末段2） | 25.02 |
| `ground_mu6_high_v9_rocker_SP` | 6 | 卡住 | 9 | SP | 曾越阶（最高1，末段1） | 25.02 |
| `ground_mu6_high_v10_rocker_SN` | 6 | 卡住 | 10 | SN | 曾越阶（最高1，末段1） | 27.21 |
| `ground_mu6_high_v10_rocker_SP` | 6 | 卡住 | 10 | SP | 曾越阶（最高1，末段1） | 27.21 |
| `ground_mu6_high_v11_rocker_SN` | 6 | 卡住 | 11 | SN | 曾越阶（最高3，末段3） | 29.52 |
| `ground_mu6_high_v11_rocker_SP` | 6 | 卡住 | 11 | SP | 曾越阶（最高1，末段0） | 29.75 |
| `ground_mu6_high_v12_rocker_SN` | 6 | 卡住 | 12 | SN | 曾越阶（最高2，末段2） | 31.91 |
| `ground_mu6_high_v12_rocker_SP` | 6 | 卡住 | 12 | SP | 曾越阶（最高1，末段1） | 31.91 |
| `ground_mu6_high_v13_rocker_SN` | 6 | 卡住 | 13 | SN | 曾越阶（最高2，末段2） | 34.27 |
| `ground_mu6_high_v13_rocker_SP` | 6 | 卡住 | 13 | SP | 曾越阶（最高1，末段1） | 34.27 |
| `ground_mu6_high_v14_rocker_SN` | 6 | 卡住 | 14 | SN | 曾越阶（最高2，末段2） | 36.87 |
| `ground_mu6_high_v14_rocker_SP` | 6 | 卡住 | 14 | SP | 曾越阶（最高1，末段1） | 36.87 |
| `ground_mu6_high_v15_rocker_SN` | 6 | 卡住 | 15 | SN | 曾越阶（最高2，末段2） | 39.22 |
| `ground_mu6_high_v15_rocker_SP` | 6 | 卡住 | 15 | SP | 曾越阶（最高1，末段1） | 41.00 |
| `ground_mu8_contact_v6_rocker_SN` | 8 | 接触 | 6 | SN | 未越阶 | 38.15 |
| `ground_mu8_contact_v6_rocker_SP` | 8 | 接触 | 6 | SP | 未越阶 | 41.49 |
| `ground_mu8_contact_v7_rocker_SN` | 8 | 接触 | 7 | SN | 曾越阶（最高1，末段1） | 42.44 |
| `ground_mu8_contact_v7_rocker_SP` | 8 | 接触 | 7 | SP | 未越阶 | 41.31 |
| `ground_mu8_contact_v8_rocker_SN` | 8 | 接触 | 8 | SN | 曾越阶（最高1，末段1） | 46.55 |
| `ground_mu8_contact_v8_rocker_SP` | 8 | 接触 | 8 | SP | 曾越阶（最高1，末段1） | 60.07 |
| `ground_mu8_contact_v9_rocker_SN` | 8 | 接触 | 9 | SN | 曾越阶（最高1，末段0） | 51.25 |
| `ground_mu8_contact_v9_rocker_SP` | 8 | 接触 | 9 | SP | 曾越阶（最高1，末段1） | 51.08 |
| `ground_mu8_contact_v10_rocker_SN` | 8 | 接触 | 10 | SN | 曾越阶（最高1，末段1） | 56.76 |
| `ground_mu8_contact_v10_rocker_SP` | 8 | 接触 | 10 | SP | 曾越阶（最高1，末段1） | 55.85 |
| `ground_mu8_contact_v11_rocker_SN` | 8 | 接触 | 11 | SN | 曾越阶（最高1，末段1） | 60.61 |
| `ground_mu8_contact_v11_rocker_SP` | 8 | 接触 | 11 | SP | 未越阶 | 60.48 |
| `ground_mu8_contact_v12_rocker_SN` | 8 | 接触 | 12 | SN | 未越阶 | 65.49 |
| `ground_mu8_contact_v12_rocker_SP` | 8 | 接触 | 12 | SP | 曾越阶（最高1，末段1） | 65.36 |
| `ground_mu8_contact_v13_rocker_SN` | 8 | 接触 | 13 | SN | 曾越阶（最高1，末段1） | 70.30 |
| `ground_mu8_contact_v13_rocker_SP` | 8 | 接触 | 13 | SP | 曾越阶（最高1，末段1） | 70.18 |
| `ground_mu8_contact_v14_rocker_SN` | 8 | 接触 | 14 | SN | 曾越阶（最高1，末段1） | 75.81 |
| `ground_mu8_contact_v14_rocker_SP` | 8 | 接触 | 14 | SP | 曾越阶（最高1，末段1） | 75.70 |
| `ground_mu8_contact_v15_rocker_SN` | 8 | 接触 | 15 | SN | 曾越阶（最高1，末段1） | 80.49 |
| `ground_mu8_contact_v15_rocker_SP` | 8 | 接触 | 15 | SP | 曾越阶（最高1，末段1） | 80.38 |
| `ground_mu8_v3_rocker_SN` | 8 | 卡住 | 3 | SN | 未越阶 | 22.87 |
| `ground_mu8_v3_rocker_SP` | 8 | 卡住 | 3 | SP | 未越阶 | 22.87 |
| `ground_mu8_v4_rocker_SN` | 8 | 卡住 | 4 | SN | 未越阶 | 37.81 |
| `ground_mu8_v4_rocker_SP` | 8 | 卡住 | 4 | SP | 未越阶 | 29.09 |
| `ground_mu8_v5_rocker_SN` | 8 | 卡住 | 5 | SN | 未越阶 | 31.33 |
| `ground_mu8_v5_rocker_SP` | 8 | 卡住 | 5 | SP | 未越阶 | 31.33 |
| `ground_mu8_v6_rocker_SN` | 8 | 卡住 | 6 | SN | 未越阶 | 36.14 |
| `ground_mu8_v6_rocker_SP` | 8 | 卡住 | 6 | SP | 曾越阶（最高1，末段1） | 36.14 |
| `ground_mu8_high_v7_rocker_SN` | 8 | 卡住 | 7 | SN | 未越阶 | 51.46 |
| `ground_mu8_high_v7_rocker_SP` | 8 | 卡住 | 7 | SP | 未越阶 | 51.46 |
| `ground_mu8_high_v8_rocker_SN` | 8 | 卡住 | 8 | SN | 未越阶 | 46.45 |
| `ground_mu8_high_v8_rocker_SP` | 8 | 卡住 | 8 | SP | 曾越阶（最高1，末段1） | 46.45 |
| `ground_mu8_high_v9_rocker_SN` | 8 | 卡住 | 9 | SN | 曾越阶（最高1，末段1） | 51.18 |
| `ground_mu8_high_v9_rocker_SP` | 8 | 卡住 | 9 | SP | 未越阶 | 51.18 |
| `ground_mu8_high_v10_rocker_SN` | 8 | 卡住 | 10 | SN | 曾越阶（最高1，末段1） | 55.93 |
| `ground_mu8_high_v10_rocker_SP` | 8 | 卡住 | 10 | SP | 曾越阶（最高1，末段1） | 55.93 |
| `ground_mu8_high_v11_rocker_SN` | 8 | 卡住 | 11 | SN | 曾越阶（最高1，末段1） | 60.54 |
| `ground_mu8_high_v11_rocker_SP` | 8 | 卡住 | 11 | SP | 曾越阶（最高1，末段1） | 60.54 |
| `ground_mu8_high_v12_rocker_SN` | 8 | 卡住 | 12 | SN | 曾越阶（最高1，末段1） | 65.43 |
| `ground_mu8_high_v12_rocker_SP` | 8 | 卡住 | 12 | SP | 曾越阶（最高2，末段2） | 65.43 |
| `ground_mu8_high_v13_rocker_SN` | 8 | 卡住 | 13 | SN | 曾越阶（最高1，末段1） | 70.24 |
| `ground_mu8_high_v13_rocker_SP` | 8 | 卡住 | 13 | SP | 曾越阶（最高1，末段1） | 70.24 |
| `ground_mu8_high_v14_rocker_SN` | 8 | 卡住 | 14 | SN | 曾越阶（最高1，末段1） | 75.76 |
| `ground_mu8_high_v14_rocker_SP` | 8 | 卡住 | 14 | SP | 曾越阶（最高1，末段0） | 75.76 |
| `ground_mu8_high_v15_rocker_SN` | 8 | 卡住 | 15 | SN | 曾越阶（最高1，末段1） | 80.43 |
| `ground_mu8_high_v15_rocker_SP` | 8 | 卡住 | 15 | SP | 曾越阶（最高1，末段1） | 80.43 |
