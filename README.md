# Mars rover：15 cm / 20 cm 台阶实验

项目负责人：wzy（GitHub @susieow）。

**先阅读 [完整交接文档](HANDOFF_WZY_2026-11-11.md)。目前15/20 cm完整合格成功样本为0，768次局部候选均未完成双后轮越首阶。**

- 后续仅允许15 cm、20 cm，两侧各15级、45°，目标是上楼→平台→下楼→地面正常停稳。
- 其他高度不再进行新动力学试验；旧4 cm回归与17/18 cm计划不是当前运行授权。
- AGX 项目：`/home/agilex/wzy/rover`；GitHub 分支：`wzy`。
- 仿真、数据集记录和审核源码：`scripts/rover`；本批冻结证据：`review_v4`；核实值：[verified_facts](docs/verified_facts_2026-11-11.json)。
- 大批NPZ、视频和venv留在AGX，没有上传Git；完整清单及SHA-256见`docs/artifact_inventory_2026-11-11.jsonl`。克隆仓库不能替代取得原始轨迹。
- 当前模型存在质量注释不一致、自碰撞关闭、轮胎视觉/碰撞宽度差异；现有重力为9.81 m/s²。交接文档已明确，勿将其视为已校准实车/火星模型。

GitHub 默认分支为 `wzy`。
