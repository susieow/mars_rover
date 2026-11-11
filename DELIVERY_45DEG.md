# 45° 台阶攀爬交付文档

日期：2026-11-10
机器：agilex@172.30.2.15（以下所有路径都在 AGX 上，Mac 上没有副本）
项目根目录：`~/wzy/rover`

## 0. 如何 SSH 登录

这台机器在 172.30.2.15，账号 agilex，SSH 默认 22 端口。在 Mac 终端执行：

```bash
ssh agilex@172.30.2.15
```

这台 Mac 已经配置了公钥免密登录。换一台新机器时先执行一次：

```bash
ssh-copy-id agilex@172.30.2.15
```

登录后进入项目：

```bash
cd ~/wzy/rover
```

从本机拷文件到 AGX：

```bash
scp 本地文件 agilex@172.30.2.15:~/wzy/
```

前端在 AGX 上跑，浏览器访问 `http://172.30.2.15:8701/`。本文档所有命令都是登录 AGX 之后执行的。不要用 conda。

## 1. 目标与约束

目标：六轮摇臂车在 MuJoCo 中爬上 45° 台阶，级数超过 10 级，台阶高度覆盖多档。

硬约束（用户原话）：
- 不能改悬架模型。
- 摩擦力增加到最大。
- 只要 45°，不做 30°。
- 不要倒角，只用正常台阶。
- 不在 Mac 上保存任何东西。
- 不用 conda。

## 2. 结论摘要

**当前控制器只是基线，不是正确解。用户明确表示控制器写得不对，将换 Astra 重写。本文档的价值在模型、评测接口、已验证边界和负面结论，不在控制策略。**

无倒角、正常台阶、当前控制器下的真实边界：

| 台阶高度 | 是否通过 | 备注 |
|---|---|---|
| 4 cm | 通过 | μ=4.0 |
| 5 cm | 通过 | μ=4.0 |
| 6 cm | 通过 | μ=4.0 |
| 8 cm | 通过 | μ=4.0，抬升角 20° |
| 10 cm | 通过 | μ=4.0 |
| 12 cm | 通过 | 需要 μ=4.5 |
| 13 cm | 失败 | 卡在第一级中轮；只在临时 ±120 N·m XML 下测过 |
| 15 cm | 失败 | 无倒角无解；曾用 3 cm 倒角通过，按新约束作废 |
| 20 cm | 失败 | 几何上过不去，见 4.2 |

## 3. 环境与目录

Python 解释器：`~/wzy/rover/.venv/bin/python`。不要用 conda。

| 路径 | 作用 |
|---|---|
| `assets/mars_rover/urdf/mars_rover.urdf` | 模型源头 |
| `assets/mars_rover/rover_actuated.xml` | 实际仿真模型，由 URDF 生成一次后固定 |
| `scripts/rover/climb.py` | 建模辅助、`build_robot`、`make_controller` |
| `scripts/rover/course.py` | 台阶场景 `build_course` + 主循环 `simulate` |
| `scripts/rover/render_course.py` | 渲染 CLI，内部调用同一个 `simulate` |
| `scripts/rover/verify_winners.py` | 逐高度复核获胜配置 |
| `scripts/rover/check_h12_plain.py` | h12 无倒角 μ4.0/4.5 对比 |
| `scripts/rover/check_large_plain.py` | h15/h20 无倒角对比 |
| `scripts/rover/check_torque.py` | 实测轮关节力矩钳制 |
| `scripts/rover/check_torque_120.py` | 临时放宽 XML 到 ±120 的对照实验 |
| `scripts/rover/render_winners.py` | 重建前端卡片与 `runs.jsonl` |
| `~/wzy/web/app.py` | 前端服务，端口 8701 |
| `~/wzy/videos/` | mp4 与 `runs.jsonl` |

注意：`build_robot()` 只有在 `rover_actuated.xml` 不存在时才从 URDF 重新生成。手工修改 `rover_actuated.xml` 会保留，除非把它删掉。仿真实际读的是这个 XML。

## 4. 模型事实

### 4.1 关节与执行器

- 6 个轮关节，执行器 `ctrlrange` 为 ±120。
- 4 个转向关节，执行器 ±40，控制目标为 0。
- 2 个摇臂关节，执行器 ±100，关节范围 **−90° 到 +20°**（`range="-1.5708 0.3491"`），轴 `0 1 0`。
- 共 12 个执行器。

**关键坑：轮关节 XML 里写着 `actuatorfrcrange="-50 50"`。MuJoCo 会把总执行器力钳制到 ±50 N·m。** 实测 `ctrl=120` 时 `actuator_force=120`，但 `qfrc_actuator=50`。也就是说之前所有标注“120 N·m”的实验，实际轮上力矩一直是 50 N·m。想真正测 120 N·m，需要把 XML 中轮关节的 `actuatorfrcrange` 改到 ±120（该实验已做，见第 8 节，结论是 120 N·m 也救不了 13 cm 以上）。

### 4.2 几何

轮半径 `r = 0.163 m`。轮相对车体的 x 偏移：前轮 +0.661，中轮 +0.016，后轮 −0.676。

后轮与车体刚性连接，摇臂只驱动前轮和中轮。摇臂角对应的轮 z（相对车体，左侧）：

| 摇臂角 | 前轮 z | 中轮 z |
|---|---|---|
| 0° | −0.368 | −0.369 |
| +20°（上限） | −0.454 | −0.234 |
| −20° | −0.237 | −0.458 |
| −35° | −0.117 | −0.487 |
| −90°（下限） | +0.332 | −0.314 |

**中轮在 +20° 时最多抬高 0.135 m。这是无倒角台阶高度的准静态上限。** 12 cm 能过，13 cm 起过不去，与实测一致。

20 cm 台阶还有一个更硬的限制：台阶高度大于轮半径 0.163 m，前轮在竖直面上无法滚上去，相当于撞墙。

### 4.3 摩擦

尖角台阶翻越所需最小摩擦系数（`μ_min = tan(acos((r−h)/r))`）：

| 台阶高 | μ_min |
|---|---|
| 4 cm | 0.87 |
| 5 cm | 1.04 |
| 6 cm | 1.23 |
| 8 cm | 1.69 |
| 10 cm | 2.39 |
| 12 cm | 3.66 |
| 15 cm | 12.5 |
| 20 cm | 不存在（h > r） |

可用上限：μ=4.0 稳定；μ=4.5 因工况而异（h12 通过）；μ≥6 接触求解器失效，轮子下陷 10 到 19 cm。MuJoCo 里静摩擦与滑动摩擦都设成同一个值，`friction="μ 0.02 0.001"`。

## 5. 仿真与评测接口

### 5.1 场景

`build_course(height, stairs, depth, friction, start_x=1.6, top_len=2.0, nosing=0.0, wheel_radius=0.163, middle_collision=True) -> Path`

- 45° 台阶用 `depth = height`。
- 12 级上、2 m 平台、11 级下，台阶宽 5 m。
- 第一级立面在 x=1.6；车从 x=0 出发，先在平地上走 1.6 m。
- `nosing` 是台阶上沿的 45° 倒角，新约束下应保持 0。

### 5.2 主循环

`simulate(height, stairs, depth, seconds=90.0, speed=6.0, friction=1.4, tau_max=50.0, deg=12.0, front_lead=0.95, mid_lead=0.10, mid_trail=0.70, slew=90.0, assist=True, on_frame=None, on_step=None, pitch_limit=12.0, pitch_fade=20.0, const_deg=None, rocker_free=False, init_yaw_deg=0.0, init_x=0.0, stop_when_done=True, mid_deg=None, mode="roll", nosing=0.0, wheel_radius=0.163, middle_collision=True, lift_deg=35.0, lift_lead=0.90, lift_hold=0.20, land_trail=0.70, rocker_gain=80.0, rocker_damp=3.0, alt_amp=0.0, alt_freq=1.0, alt_bias=0.0, heading_hold=True, alt_front_scale=1.0, recovery=False, stall_dx=0.01, stall_time=1.0, recovery_time=2.0, recovery_freq=1.5, recovery_amp=20.0, dwell_amp=0.0, dwell_hold=1.5) -> dict`

- `SIM_DT = 0.005`，`CTRL_DT = 0.02`。
- 控制律直接写在 `simulate` 里，**没有控制器回调接口**。`on_step(m, d, t)` 是每个物理步之后调用，只适合记录；`on_frame` 每 4 步调用，用于渲染。
- 新控制器要么替换 `simulate` 里的控制块，要么先重构出 `controller(m, d, t) -> ctrl` 钩子。

返回字典字段：

`height, stairs, depth, friction, deg, assist, x, z, max_z, plat_z, max_pitch_deg, reached_top, back_down, past_end, stopped_at_s, recovery_count, success`

成功判定：

- `reached_top = max_z > h0 + plat_top − 0.06`
- `back_down = z < h0 + 0.06`
- `past_end = x > course_end + 0.3`
- `success = reached_top and back_down and past_end`

也就是必须爬上平台并从另一侧下来。

### 5.3 确定性

同一参数、同一机器，结果逐位可重复（h15 的 x=8.74 两次完全一致）。但参数敏感：h12 倒角 2 cm 失败，3 cm 和 4 cm 通过。评测时不要因为一次通过就认定鲁棒。

## 6. 当前控制器（基线，待替换）

- 轮子：目标轮速 6 rad/s，P 控制 `kp=25`，输出限幅 `tau_max`。左右轮按航向误差做差速修正：`corr = 4.0*wrap(y_ref−yaw) + 0.6*yaw_rate`，`y_ref = yaw0 + clip(−0.4*y, −0.25, 0.25)`。`heading_hold=True` 是默认值，h10 在关闭时会失败，很关键。
- 转向：PD 回零，`−20*q − qvel`，限幅 ±40。
- 摇臂：开环位置排程 `schedule(x)`，用**已知的**立面位置 `risers` 计算：
  - 当 `r − front_lead < x < r − mid_lead`，目标 `−deg`（前轮抬起、中轮下压）。
  - 当 `r − mid_lead ≤ x < r + mid_trail`，目标 `+deg`（前轮下压、中轮抬起）。
  - 其余为 0。
  - 目标经 90°/s 斜率限幅，摇臂 PD 增益 80、阻尼 3，限幅 ±100。
- 每个高度用不同的 `deg` 和窗口，见第 7 节。

这个控制器的根本问题：

1. 它知道每一级立面的位置，是“背台词的课程播放器”，不是感知驱动的控制器。
2. 每个高度单独调参，12 级对称课程上过拟合。
3. 没有接触力、IMU、打滑估计的闭环。
4. 对时序和相位敏感，参数一动结果就变。

`--recovery` 里保留了堵转自救（反相正弦或单侧交替保持），默认关闭。实测无效且可能有害，见第 8 节。

## 7. 已验证结果（无倒角，正常台阶）

统一条件：45°、`depth = height`、12 级、120 s 到 150 s、μ 如下、轮力矩标注 120 但实际 50 N·m、`pitch_limit=0`、`heading_hold=True`。

| 高度 | μ | deg | front_lead | mid_lead | mid_trail | 结果 | 终点 x |
|---|---|---|---|---|---|---|---|
| 4 cm | 4.0 | 12 | 0.95 | 0.10 | 0.70 | 通过 | 5.54 |
| 5 cm | 4.0 | 12 | 0.95 | 0.10 | 0.70 | 通过 | 5.53 |
| 6 cm | 4.0 | 12 | 0.95 | 0.10 | 0.70 | 通过 | 5.79 |
| 8 cm | 4.0 | 20 | 0.95 | 0.08 | 0.56 | 通过 | 6.17 |
| 10 cm | 4.0 | 12 | 0.95 | 0.10 | 0.70 | 通过 | 10.66 |
| 12 cm | 4.5 | 12 | 0.95 | 0.10 | 0.70 | 通过 | 10.89 |
| 12 cm | 4.0 | 12/20 | 上述 | 上述 | 上述 | 失败 | 0.7 到 0.9 |
| 13 cm | 4.0 | 12/20 | 0.95 或 9.5h | h | 7h | 失败 | 1.0 到 1.1 |
| 15 cm | 4.5 | 12/20 | 0.95 或 9.5h | h | 7h | 失败 | 1.45 |
| 15 cm | 4.0 | 20 | 9.5h | h | 7h | 失败 | −1.9 到 1.5 |
| 20 cm | 4.0/4.5 | 12/20 | 9.5h | h | 7h | 失败 | 0.5 到 1.5 |

失败判据一致：`max_z` 只有 0.9 到 1.1 m（平台 1.7 到 2.9 m），车在第一级或前几级反复抬头。

h13/h15/h20 在“真实 120 N·m”下依然失败（临时 XML，实验后未修改交付模型）。

## 8. 失败路线与负面结论

这些结论可以帮下一任省时间，不建议重复。

### 8.1 左右交替摇臂

试过三种：

1. 反相正弦：左右摇臂 `±amp*sin(2πft)`，amp 10 到 25，bias −5 到 0，freq 1 到 2。
2. 单侧保持（dwell）：一侧抬到 +amp 保持 `hold` 秒，另一侧为 0，交替。
3. 抬起侧前中轮空转、落地侧驱动、后轮全程出力。

全部失败，包括 h10 这种同相排程本来能过的工况。原因：45° 已接近附着极限，单侧抬起会卸掉一侧支撑，整车滑回去。

### 8.2 卡住自救

严格触发（2.5 s 内位移 < 5 mm）下：

- h10 仍通过（触发 2 次，x 从 10.66 降到 9.41）。
- h12 通过（触发 1 次）。
- h8 触发 41 次原地不动。
- h15 一次都不触发。
- h20 触发 25 到 32 次仍失败。
- deg=20 时自救反而有害：h8 无自救 x=6.17 通过，加自救触发 64 次失败。

自救代码保留在 `recovery=True`，默认关闭。

### 8.3 倒角

倒角能显著改变结果：

- h12：3 cm 或 4 cm 倒角，μ=4.0 可通过。
- h15：3 cm 倒角，μ=4.0、deg=20、窗口 1.425/0.15/1.05 可通过。
- h20：8/10/12/14 cm 倒角全部失败。

按新约束“不要倒角”，这些结果作废，只作为“接触几何确实能改变可爬高度”的证据保留。

### 8.4 加大轮力矩

把 wheel joint 的 `actuatorfrcrange` 临时改到 ±120 后，h13/h15/h20 仍然失败。瓶颈是几何与附着，不是力矩。

### 8.5 斜向接近

h20 用 15° 和 30° 航向接近，仍然失败。

## 9. 给下一个控制器的建议

1. **先做接触事件检测，不要再按 x 排程。** 仿真里可以直接读每个轮子与地形 geom 的接触力，geom 名有 `up{i}`、`down{i}`、`platform`、`ground`，轮子是 `left_front_wheel_col_*` 等。建议状态机：接近、前轮接触立面、抬前轮、前轮落台阶、中轮接触立面、抬中轮、中轮落台阶、后轮跟上。
2. **中轮抬高是核心瓶颈。** +20° 摇臂上限对应 0.135 m 抬高。13 cm 以上必须换机制：动态冲量、以前轮为支点把车体抬起来、利用后轮驱动抬头，或者接受 12 cm 上限。任何方案都不能改悬架几何。
3. **把 `simulate` 重构出控制器钩子。** 建议签名 `controller(model, data, t) -> np.ndarray[12]`，在 `CTRL_DT` 调用，物理循环和其他评测逻辑不变。这样新控制器可以独立迭代，渲染和前端继续复用。
4. **评测协议固定。** 高度 4 到 20 cm、步长 1 cm；12 级；`depth=height`；μ=4.0 和 4.5 两档；每档跑一次。记录 `success/max_z/x/max_pitch/stopped_at`。通过标准沿用现有的“上平台并从另一侧下来”。
5. **不要再相信 `ctrlrange`。** 读 `d.qfrc_actuator` 确认真实力矩；轮关节 XML 里是 ±50。
6. **保留 `heading_hold`。** 关掉它 h10 就失败，说明航向保持对对称台阶是必要的。
7. **前端卡片要清理。** 现有 h12/h15 卡片用的是倒角，按新约束无效。新控制器跑出的结果写到同一个 `runs.jsonl` 即可，字段见 `~/wzy/videos/runs.jsonl`。

## 10. 复现命令

```bash
cd ~/wzy/rover/scripts/rover

# 7 个配置逐条复核（h12、h15 带 3 cm 倒角，按新约束仅作参考）
~/wzy/rover/.venv/bin/python verify_winners.py

# h12 无倒角，μ4.0 与 4.5 对比
~/wzy/rover/.venv/bin/python check_h12_plain.py

# h15/h20 无倒角
~/wzy/rover/.venv/bin/python check_large_plain.py

# 实测轮关节力矩钳制
~/wzy/rover/.venv/bin/python check_torque.py

# 渲染单个工况
~/wzy/rover/.venv/bin/python render_course.py \
  --height 0.10 --stairs 12 --depth 0.10 --friction 4.0 --tau-max 120 \
  --deg 12 --front-lead 0.95 --mid-lead 0.10 --mid-trail 0.70 \
  --pitch-limit 0 --seconds 150 --out ~/wzy/videos/test.mp4
```

前端重启（当前进程 PID 会变）：

```bash
cd ~/wzy/web
setsid nohup ~/wzy/rover/.venv/bin/python app.py --port 8701 \
  > ~/wzy/web/app.log 2>&1 < /dev/null &
```

访问 `http://172.30.2.15:8701/`。

## 11. 已知坑

1. **轮力矩被 XML 钳到 50 N·m。** 所有“120 N·m”的标签都不真实。
2. **摇臂上限 +20°，中轮最多抬 0.135 m。** 13 cm 以上准静态无解。
3. **μ≥6 会让接触求解器失效。** 轮子下陷 10 到 19 cm，结果不可信。
4. **`build_robot()` 有 early return。** 删掉 `rover_actuated.xml` 才会从 URDF 重新生成，手工改动会被保留。
5. **参数时序敏感。** 同一高度换一个倒角尺寸或窗口就从通过变失败。不要用单次结果下结论。
6. **`heading_hold` 默认开，别关。**
7. **前端 `runs.jsonl` 按 `out` 去重。** 重渲染时用新文件名或先清空。
8. **不要用 conda。** 统一用 `~/wzy/rover/.venv/bin/python`。
9. **悬架模型不能改。** 这是用户红线，所有“通过”都建立在现有 URDF 上。
