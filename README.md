# 火星车 20 cm 台阶实验

本项目为火星车爬楼数据集寻找可用经验。当前只测试单级高/深均为 **20 cm、45°、上行 15 级**的直角楼梯。最终目标仍是上楼、过平台、下楼，最后在地面以正常车形态停稳。

目前有 112 条最终版本的地面起步诊断，**68 条曾满足双后轮局部越阶条件，尚无登顶或完整路线成功样本**。这些局部越阶记录是当前优先参考的动作，用户观察其效果相近；不能解释成已经完成爬楼或已经通过真实硬件验证。

## 交接先读

- [经验文档](EXPERIENCE.md)：最终参数、已测/未测范围、局部有效经验、淘汰路线和逐条编号。
- [协作说明](AGENTS.md)：AGX、Git 和工作习惯。
- [SuperMimic 非人形适配调研](docs/references/supermimic_nonhumanoid_code_survey.md)：保留原文的参考资料。

## 最终测试版本

| 项目 | 设置 |
|---|---|
| 台阶滑动摩擦系数 | 4、6、8；从仿真 0 秒生效 |
| 六轮基础目标速度 | 3～15 rad/s，每整数档；六轮同档 |
| 触发条件一 | 任一前轮首次对上行台阶产生正法向接触力 |
| 触发条件二 | 前轮已接触，连续 2 秒车身向前净位移不足 2 cm |
| 摇臂动作 | 左右同向，SN：先 −20° 再 +20°；SP：先 +20° 再 −20° |
| 摆动时序 | 8 秒/周期、3 周期，之后回中立；触发前中立 |
| 观察时长 | 触发后 30 秒；未触发最多 120 秒 |
| 审核行为 | 违规记录后继续观察；不因审核失败提前截断 |

这张表是版本范围，不代表 156 个组合全部完成。已测 **112**，未测 **44**；详见经验文档。基础轮速经过变化率限制和航向修正，目标速度、实际轮速和车身速度要分别看。

## 在 AGX 上查看

```bash
ssh agilex@172.30.2.15
cd /home/agilex/wzy/rover
.venv/bin/python tools/find_experiment.py --friction 4 --trigger contact --speed 7 --direction SN
.venv/bin/python tools/find_experiment.py --crossed
```

浏览器页面只展示最终版本 112 条视频和曲线，可按全部参数筛选，并勾选“只看曾越阶”。本机建立隧道后访问 `http://127.0.0.1:8701/`：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 127.0.0.1:8701:127.0.0.1:8701 agilex@172.30.2.15
```

隧道命令保持前台且无输出是正常现象；如本机 8701 已占用，先用现有隧道，或换本机端口。AGX 服务启动方式：

```bash
cd /home/agilex/wzy/rover
.venv/bin/python web/app.py --host 127.0.0.1 --port 8701
```

当前运行代码由仓库 `web/` 部署到 `/home/agilex/wzy/web/`。后台日志 `/tmp/rover-web-8701.log`。旧页面已下架，历史视频保留在 `/home/agilex/wzy/videos/review_high_stairs/`，用经验索引查文件名、参数及路径。

## 代码与数据

| 位置 | 用途 |
|---|---|
| `scripts/rover/ground_*.py` | 最终版本原始实验、媒体和核验程序；保留原内容和哈希 |
| `speed20_experiment.py`、`overspeed20_stairs.py`、`recovery20_amp20_experiment.py` | 当前实验仍依赖的控制、接触、超速统计和摇臂时序 |
| `scripts/rover/{course,closed_loop_controller,episode_audit,dataset_io,...}.py` | 场景、数据集、审核和视觉渲染依赖；数据集基线尚无高阶成功 |
| `tools/find_experiment.py` | 查询全部已整理经验，确认已测组合，支持 `--json` 输出完整字段 |
| `tools/verify_handoff.py` | 只读核对文档编码、最终目录、参数覆盖和媒体文件；不推进仿真 |
| `docs/final_experiments.json` | 最终 112 条的前端目录及 44 个缺测组合 |
| `docs/experiment_index.jsonl` | 最终和历史实验的逐条参数、原因、哈希和证据路径 |
| `docs/diagnostic*.jsonl`、其他 JSON | 冻结原始登记、批次核验，供追溯 |
| `review_v4/`、`datasets/` | 历史冻结证据，不是新的运行入口 |
| `review_v3/model_sha256.txt`、`scripts/rover/course.py.before_closed_loop_20261110.py` | 审核器按哈希读取的原始模型/碰撞策略证据，名称虽旧仍必须保留 |
| `runs/` | AGX 原始轨迹、场景、结果和曲线；不提交 Git |
| `assets/`、`render_assets/` | 原模型及视觉资源；本次整理未改变物理模型 |

最终版本保留原脚本名，避免破坏历史哈希和依赖。入口参数及对应批次见经验文档。删除的探针、旧扫描和页面可通过 Git 历史恢复；清理清单见 `docs/cleanup_manifest_20261112.json`。

Git：`https://github.com/susieow/mars_rover.git`，默认分支 `wzy`。需要隔离工作目录时自行 `git worktree add`；无需预建固定的协作 worktree。只克隆 Git 不含大轨迹和视频，复现时从 AGX 按索引取所需文件并核对哈希。

当前环境使用已有 `.venv/bin/python`、MuJoCo 和系统 ffmpeg/ffprobe，依赖记录在 `requirements-simulation.txt`。模型仍有质量注释不一致、轮胎视觉/碰撞宽度差异、自碰撞关闭等已知限制，使用地球重力；详见经验文档。
