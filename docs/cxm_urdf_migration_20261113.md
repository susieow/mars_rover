# cxm 接触恢复诊断的新 URDF 适配（2026-11-13）

当前 URDF：`c12222dfd7b770b21f5620be33d55ac2ab974fb499f32b7d5c16045b1f5ca061`。
历史 15 条诊断使用 `cca26f5be19e02a21feed1dc7b4b9d0147fa0f81db9079b049679ffb3dccde8a`。

相对原 URDF，37 个 link 和 2 个 joint 的 XML 有变化，主要是结构对称化、碰撞体与安装描述。
当前 XML 总质量为 126.746112 kg。12 个运动关节的 limit、dynamics 字段未变：
轮速度 52.359878 rad/s、力矩 50 N·m；转向 ±1.5708 rad、3.1416 rad/s、40 N·m；
摇臂 −1.5708～0.3491 rad、1.57 rad/s、100 N·m。

## 为什么同时修改代码

原控制诊断从旧 `rover_actuated.xml` 生成场景，并锁定旧 URDF 哈希。只改网页或哈希会把旧几何误当新模型。
新增 `scripts/rover/recovery_urdf_model.py`，复用 Phase B 的 `urdf_slope_model.py` 编译新 URDF，
在每条记录目录保存编译模型和原始 URDF 快照。模型检查包含 link/frame、惯性张量、关节轴、限值、阻尼、碰撞体、摩擦与求解设置。
控制器的硬限幅读取 URDF，控制增益、阶段时序和策略保持原值；运行审核采用零越限容差。

新模型采用 Phase B 的接触位掩码，不延续旧诊断的 90 对碰撞排除。机器人摩擦采用 Phase B 配置；
地面和平台滑动摩擦为 4，台阶为命令行指定值。轮胎 0.6 是用户要求，不能说是 URDF 的轮胎字段。
这些与旧诊断的接触设置差异已经公开，因此新旧结果不能当作只改变单个控制变量的对照。

历史 15 条结果、轨迹、视频不改写。新运行使用独立目录 `runs/urdf_slope_control_c12222df/`，
case ID 增加 `_c12222df`，视频也写入独立目录。渲染器核对记录的 URDF 哈希，不用当前模型重画旧轨迹。

## 网页判定

`/experiments/contact-recovery` 显示当前 URDF 哈希、XML 参数、每条记录的模型版本、关节限值是否一致、记录时审核与当前模型审核。
历史关节审核不变，但旧模型对当前模型的完整合规性为“未验证”，不能重标为新模型成功。
目录生成器同时收集历史与新模型的已生成视频记录；模型没有媒体的短测不计入网页视频统计。
顶层 `urdf_compliant` / `full_success` 保留历史 API 兼容含义，表示各记录原审核结果，绝非当前模型认证数量；新网页不再拿这些字段宣称完整通过。

初始化阶段、独立重放、步长复核和新版平台判据尚未统一到这套诊断控制器，因此新记录即使模型字段与保存阶段关节检查通过，仍只属于诊断。

## 扩大摆幅与平台辅助诊断（2026-11-13）

新增 `adaptive_zigzag`，只在 30°、35°、40°、45° 上验证。前段将微 Z 的约 ±6° 航向 / ±8° 摇臂扩大到约 ±10° / ±14°；检测到任一前轮对平台产生超过 5 N 的法向支撑后，切换 3 s 的平台辅助：前/中/后轮组目标约为 4.0/3.6/2.8 rad/s，左右轮速差以 1 rad/s 交替，摇臂在 +10° 与 −4° 间切换。

结果：30° 未到平台，最大穿透 24.25 mm 且有运行越限；35° 未到平台但保持限值和数值质量；40° 未到平台但保持限值和数值质量；45° 未到平台并有运行越限。30° 触发过平台辅助阶段但未完成停稳，说明当前动作不足以解决末级上平台。扩大摆幅没有证明 35° 或 40° 可爬，也不能把失败写成车辆绝对物理上限。

新版页面现在把新版 19 条记录和旧版 15 条记录分开；状态颜色为绿色（到平台且质检通过）、黄色（到平台但质检未通过）、红色（未到平台）。

## 使用

在 `/home/agilex/wzy/rover-chaixm` 下执行：

```bash
/home/agilex/wzy/rover/.venv/bin/python -B scripts/rover/urdf_slope_control_20261112.py --friction 6 --check-model
# 按需新跑，拒绝覆盖已存在的同名 case：
/home/agilex/wzy/rover/.venv/bin/python -B scripts/rover/urdf_slope_control_20261112.py --friction 6 --strategy zigzag --direction SN --angle-deg 25 --max-seconds 240
for angle in 30 35 40 45; do /home/agilex/wzy/rover/.venv/bin/python -B scripts/rover/urdf_slope_control_20261112.py --friction 6 --strategy adaptive_zigzag --direction SN --angle-deg $angle --max-seconds 240; done
MUJOCO_GL=egl /home/agilex/wzy/rover/.venv/bin/python -B scripts/rover/urdf_slope_control_media_20261112.py --case urdf25_mu6_zigzag_SN_c12222df
/home/agilex/wzy/rover/.venv/bin/python -B scripts/rover/urdf_slope_catalog_20261112.py
```

共享网页服务仍为 8701，接触恢复目录改为读取 cxm 的 `docs/urdf_slope_review.json`。
目录目前在服务启动时读取，重新生成目录后需重启该服务；浏览器刷新只重新读取服务已加载的数据。
部署时保留共享页面已有的 Phase D/E 改动，只更新接触恢复页及主页的诊断摘要。
源码修改只写入 cxm，未合并或提交到 wzy，未推送远程。

## 验证

检查新模型 1,633 项字段；人工扰动质量、碰撞尺寸、安装位置和惯量均被拒绝，旧场景也不能通过新模型检查。
短时运行检查仅含初始化和 50 个保存物理步，不是爬楼结果，不计入视频目录。
验证证据和修改前源码在 `runs/urdf_migration_20261113/`。
