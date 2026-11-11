火星车 45°楼梯成功经验数据集：使用与验收说明

目录：/home/agilex/wzy/rover/datasets/45deg_v3
目的：提供具有完整动作、状态和来源证据的仿真成功经验。失败尝试保留用于排查，不进入训练成功集。

当前采集目标更新（2026-11-11，按用户最新要求）
- 优先开发15 cm、20 cm两档：15级上行、平台、15级下降（含落地），最后地面停稳；每条楼梯内部等高且深度等于高度，保持45°。
- 10/12 cm不再是当前优先目标。4 cm/12级合格种子仅作回归参考，不能充抵15/20 cm训练样本。
- 当前15/20 cm尚无正式接纳样本；review_v4中的控制试验是诊断，失败不进入成功数据集。当前正式控制器仍为旧中立策略，高台阶控制原型尚未合入。
- train.jsonl是当前目标成功训练清单；accepted.jsonl是包含旧回归种子的全量接纳归档，不应整表当作新目标训练集。
- 验证集采用15/20 cm的独立初态；17/18 cm、15级留作未调参测试。测试尚未执行，正式release尚未冻结，不能据此声称成功。
- 生成器默认已改15/20 cm、15级、base初态。--dry-run仅列计划；测试执行需显式匹配的冻结release。
- 每次尝试先按几何/初态确定划分，失败和异常保留在原划分，不能按结果重新分配。详细规则与命令见scripts/rover/dataset_split_plan.txt。

一、当前状态与训练准入

【metadata 状态快照开始】
统计时间（Asia/Shanghai）：2026-11-11T08:43:07.241767+08:00
状态说明：正式接纳后的实际 metadata 统计；后续新增样本须重新核验。
有 metadata 的尝试：12 条
自动审核与独立复审均通过：1 条
待视觉复核：0 条
正式接纳（dataset_accepted=true）：1 条
自动拒收或不合格：11 条
自动与独立通过样本：seed1000_h04_base
待视觉样本：无
正式接纳样本：seed1000_h04_base
【metadata 状态快照结束】

上面的数量只反映统计时读取到的 episodes/*/metadata.json。后续复审或接纳可能改变状态，应重新读取 metadata；不要从文件夹名、视频数量、attempts.jsonl 或 success=true 推断成功样本数。

训练成功集必须同时满足以下条件：
1. dataset_accepted 的值严格为 true。
2. automated_checks_passed=true，quality_rejection_reasons 为空，independent_verification.verify_ok=true。
3. 对当前保存文件重新运行独立 verify 后退出码为 0，且每条结果 verify_ok=true。
4. 有实际执行的视觉复核记录 visual_review.json，明确标注 sampled_pass（抽样画面）或 full_video_pass（完整视频），并绑定轨迹/视频哈希；pending 不算通过。不得将抽样检查写成全视频逐帧检查。

dataset_accepted=false 的任何样本均不可用于训练成功经验。自动通过只表示可进入视觉复核；独立 verify 不执行视觉复核，也不会自动接纳。success/simulation_success 是历史仿真结果字段，不能替代以上条件。失败、超速、末态不稳、来源不完整、复审失败或尚未审核的尝试不得进入成功训练清单。episodes 目录保留这些尝试是为了追溯，不代表已经入库接纳。

二、完整路线定义

新目标场景必须为水平地面、15个上行台阶方箱、水平顶平台和14个下行台阶方箱，最后一级落到地面，共15次下降。历史4 cm种子仍保留12个上行方箱、11个下行方箱和共12次下降。实际级数必须匹配各自保存的参数。每级高度等于踏面深度，整体坡度为45°；方箱轴对齐、连续、底面落地，无额外地形、倒角或圆角辅助。地形接触 margin/gap 必须为 0，碰撞掩码 contype=2、conaffinity=1。

独立复审读取保存场景的实际几何，并按真实接触依次证明：
起始位于楼梯前且六轮落地 → 上行台阶有效承重接触 → 六轮在顶平台同时有效承重 → 下行台阶有效承重接触 → 楼梯后六轮落地 → 保持正常末态连续稳定至少 1 秒。

全程车体中心 |y|≤0.5 m，所有车轮碰撞包围球均不得越出场景实际宽度；全程 |yaw|≤30°、|roll|≤15°、|pitch|≤60°。末态车体 x 必须超过楼梯末端至少 0.3 m，所有车轮碰撞几何中心均超过末端。几何及路线判断不使用控制器自报 events 或 metadata.success。

当前 h04 场景：单级高/深均 0.04 m，顶高 0.48 m；楼梯始于 x=1.60 m，平台区间 x=2.08～4.08 m，楼梯末端 x=4.52 m，总宽 5 m。其他高度必须独立验证，不能由 h04 成功推导通过。

三、固定物理验收门槛

每一个物理步都进行检查。任何历史超限均保留；末态恢复正常不能消除超限记录。

关节速度来自原始 URDF，各关节固定增加 0.01 rad/s 数值容差：
- 6 个轮关节：原始上限 52.359878 rad/s，实际拒收阈值 52.369878 rad/s。
- 4 个转向关节：原始上限 3.1416 rad/s，实际拒收阈值 3.1516 rad/s。
- 2 个摇臂关节：原始上限 1.57 rad/s，实际拒收阈值 1.58 rad/s。
任一物理步的绝对速度高于对应阈值即拒收。审计保存逐关节原始上限、加容差阈值、峰值、累计超限时长，以及所用 URDF 路径与内容哈希。

任务姿态质量门槛：4 个转向关节全程实际绝对角度均须≤30°，逐物理步读取 qpos；任一步超过即永久记录 route_steer_over_30deg，保存全局及逐关节峰值、发生时刻和关节名。实际姿态必须正常，不能仅因控制命令很小、转向角速度未超限或最后回正就通过。当前命令幅度约±8.6°，30°是用于正常直线爬楼成功示范的宽松任务质量边界，不是 URDF 机械限位；模型的约±90°机械转向范围保持不变。更新此门槛后，旧的自动通过结论必须重新复审，旧状态快照不能覆盖当前验收规则。

其他全程门槛：
- 机器人与地形最大接触穿透深度不得超过 0.020 m。
- 非轮部件与地形发生大于 5 N 的接触力时，累计持续时间不得超过 0.020 s；同一步多个接触不重复累计总时长。
- 有限关节相对模型角度上下限的最大越界量不得超过 1°。摇臂范围为 [-1.5708, 0.3491] rad；转向范围为 [-1.5708, 1.5708] rad；轮转动为连续关节。
- 实际广义执行器力矩峰值：轮≤50 N·m、转向≤40 N·m、摇臂≤100 N·m，各允许 0.001 N·m 数值容差。检查 qfrc_actuator，不只检查发出的 ctrl 命令。
- qpos、qvel、qacc、ctrl、qfrc_actuator、actuator_force、有效接触信息及时间均须为有限数值；不得出现 MuJoCo warning 或非单调仿真时间。
- 必须识别出完整 6 个轮关节，不能用三轮落地或车体触地代替六轮承重。

末态以下条件必须在逐物理步审核中同时、连续成立至少 1.0 s；任一步不满足即重新计时：
- 六轮各自与 ground 的有效承重法向力均 >5 N，接触法向的竖直分量绝对值 >0.70。
- |roll|<3°、|pitch|<3°、|yaw|<5°；yaw 目标为世界 +x 方向。
- 车体中心 |y|<0.25 m。
- 车体线速度范数 <0.03 m/s，角速度范数 <0.08 rad/s。
- 每个轮关节绝对速度 <0.2 rad/s。
- 4 个转向和 2 个摇臂关节的绝对角度均 <3°，绝对速度均 <0.1 rad/s。
5 N 仅是“有效承重接触”的最低门槛，不代表轮间载荷均匀，也不能证明实际硬件稳定裕度。

独立重放要求相同 MuJoCo 版本、场景 SHA256、关节/执行器名称和顺序、物理时间步匹配，且具有逐物理步记录。控制动作对齐必须正确；时间、qpos、qvel、逐步 qfrc_actuator 的最大绝对重放误差均≤1e-8。所有拒收原因必须为空，才能通过独立物理及路线复审。

四、文件与 1000 Hz 数据语义

每条尝试保存在 episodes/<case>/：
- metadata.json：参数、审计、独立复审、来源哈希、视觉状态和接纳标志。
- scene.xml：本次仿真场景。外部 mesh 不重复复制，路径及内容哈希记录于 metadata。
- trajectory.npz：完整控制动作、状态、物理步及重放初始状态。
- sources/：历史种子冻结7份源码：course.py、closed_loop_controller.py、episode_audit.py、dataset_io.py、generate_dataset.py、verify_dataset.py、climb.py；新采集另冻结dataset_plan.py、publish_dataset.py，共9份。
- contact_sheet.jpg 等可视化产物：用于人工检查，不能替代 NPZ 和复审结果。

当前 v5 控制器与物理仿真周期均为 0.001 s，即 1000 Hz；场景 XML 显式保存 timestep，Recorder 读取 course.CTRL_DT。旧 servo500/reseat500 使用 500 Hz，均未接纳。历史 seed004 尝试的控制周期为 0.02 s，这些失败尝试不可混入当前成功集。读取数据时始终使用 NPZ 的 simulation_dt、control_dt、time、action_dt、action_steps，不自行假定频率。

N 条 ctrl 动作对应 N+1 行 time/qpos/qvel：
ctrl[i] 从 time[i] 开始生效，持续到 time[i+1]，目标后继状态为 qpos[i+1]/qvel[i+1]。action_dt[i] 为实际持续时间，action_steps[i] 为所覆盖物理步数；当前正常 1000 Hz 配置下一条动作覆盖一个物理步。最后一条动作也必须有真实积分后的末态，不能丢掉终点或补造状态。phase、desired_rocker、commanded_rocker 等按动作索引保存；关节和执行器维度以 joint_names、joint_qposadr、joint_dofadr、actuator_names 为准。

physics_time、physics_qpos、physics_qvel 保存每次 mj_step 完成后的状态。physics_step_start_time 给出该步起点，physics_action_index 指向实际使用的 ctrl，physics_ctrl 保存该步命令。仿真时间来自 d.time，不能用控制调用次数或视频帧数替代。

接触和力缓存不等同于积分后位置的重新求解结果：
- physics_qfrc_actuator 是该积分区间最近一次力求值产生的缓存，不是针对 physics_qpos 重新计算的力。
- contact_mask 是控制记录时已有的轮-世界接触缓存；contact_cache_source_step 指出对应的先前物理步，-1 表示初始缓存早于记录、求值时间未知。
- ground_contact_mask 来自控制器当时的接触诊断，不能直接当作对同一行积分后状态重新计算的承重标签。
训练或分析若需要严格同相位接触标签，应按照这些时间关系处理，或在单独实例明确重新求解并另存派生标签；不得改写原始记录混淆语义。

initial_integration_state 与 integration_state_spec 保存完整积分状态，包括影响确定性重放的 warmstart 等信息。独立重放按原 ctrl 和 action_steps 执行，不运行控制器、不由视频反推动作，也不在恢复初始积分状态后插入会改变缓存的额外 forward。

五、来源完整性

生成前先读入并保存源码快照。source_sha256 记录起点快照哈希，source_sha256_end 记录结束哈希，source_loaded_sha256 记录本 Python 进程导入前的源码基线；三者必须一致。长驻进程导入旧代码后文件被修改、或生成期间源码变化，均拒收。独立 verify 逐文件检查 sources 内容与清单，使用当前独立审计重新判断，而不是运行保存的旧审核结论。

mesh_sha256 为场景实际引用的每个外部 mesh 的绝对路径→内容 SHA256 清单。生成前记录，独立 verify 在重放前后核对。文件缺失、路径集合变化或内容哈希变化都拒收。只记录哈希不会自带 mesh 文件；迁移数据必须保留这些资源及可解析路径，不能声称单独 NPZ 已包含完整视觉资产。

NPZ 另外记录 scene_sha256、MuJoCo 版本和模型/关节/执行器信息；metadata 保留固定机器人 model_sha256。缺少新来源清单的旧尝试不能靠补写 success=true 升级为合格样本。

六、独立复审与离线渲染命令

在 AGX 上执行，使用项目已有 venv，不使用 conda。

复审一条或多条候选：
cd /home/agilex/wzy/rover/scripts/rover
PYTHONDONTWRITEBYTECODE=1 /home/agilex/wzy/rover/.venv/bin/python verify_dataset.py /home/agilex/wzy/rover/datasets/45deg_v3/episodes/seed1000_h04_base/metadata.json

复审当前所有已经接纳的样本：
PYTHONDONTWRITEBYTECODE=1 /home/agilex/wzy/rover/.venv/bin/python verify_dataset.py --accepted --dataset-root /home/agilex/wzy/rover/datasets/45deg_v3

JSON 顶层 verify_ok 与每条 episodes[].verify_ok 必须均为 true，退出码必须为 0。任一拒绝、读取/哈希/格式错误或空的已接纳集合都会以非零码退出；空集合不能显示“全部通过”。这些命令不修改接纳标志。

渲染自动合格候选的完整复核视频与关键帧：
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/agilex/wzy/rover/.venv/bin/python render_dataset.py /home/agilex/wzy/rover/datasets/45deg_v3/episodes/seed1000_h04_base/metadata.json

只刷新诊断关键帧，供失败分析使用：
MUJOCO_GL=egl PYTHONDONTWRITEBYTECODE=1 /home/agilex/wzy/rover/.venv/bin/python render_dataset.py /home/agilex/wzy/rover/datasets/45deg_v3/episodes/seed1000_h04_base/metadata.json --contact-sheet-only

渲染从保存的 physics_qpos 抽取近邻真实姿态，不插值，不再执行控制器或推进仿真。视频为 50 fps，含准确初态和末态，并保存帧来源时间、抽样误差等信息。双视角近景用于检查轮胎、摇臂、车体形态和接触。诊断关键帧可以为失败尝试生成，视频或缩略图的存在不代表接纳；视觉复核必须注明抽样范围；当前种子审核了全时间轴每 0.5 s 的 43 个侧视画面、19 个最差姿态等关键帧，以及上楼/下楼/准确末态的双视角视频帧，范围为 sampled_frames，未声称完整视频逐帧检查。

七、只读前端的角色

前端用于浏览展示清单中的已接纳视频，没有仿真、生成、审核、训练入口。当前实现读取 /home/agilex/wzy/videos/runs.jsonl，按 dataset_accepted=true 过滤，兼容旧 accepted=true 字段。它不会逐次重读 episode metadata，也不会独立重放或重新判断 visual_review_status。

因此 runs.jsonl 和前端只负责展示，不是训练准入权威。发布清单必须由已完成物理、来源和视觉验收的记录生成；训练入口仍须逐样本检查 metadata 并重新独立 verify。不得通过手工改展示标志把失败或待审核样本变成成功样本。

审核页只绑定 AGX 的 127.0.0.1:8701。可在 AGX 本机浏览器访问 http://127.0.0.1:8701。若要从个人电脑访问，可由用户自行在个人电脑终端执行：
ssh -L 8701:127.0.0.1:8701 agilex@172.30.2.15
保持该 SSH 会话运行，再在个人电脑打开 http://127.0.0.1:8701。该方式不将审核页开放到局域网所有地址，也没有授权公开下载数据。

八、模型与适用范围限制

机器人结构、几何尺寸、质量、惯量、关节参数和执行器限幅经派生场景逐项核对保持不变，总质量约 126.746 kg；URDF 内旧的 122.5 kg 注释不作为验收依据。数值步长与接触响应明确调整，不能声称全部物理设置与原 XML 逐项相同。

- 派生场景沿用旧 course 的碰撞策略：原 XML 物理 geom 默认 contype/conaffinity=1/1，旧场景将其映射为 1/2，地形为 2/1，视觉 geom 保持 0/0，因此关闭机器人自碰撞。model_identity 记录这 82 个精确掩码映射及旧 course 源码哈希。地形接触审核不能证明车体、轮胎、摇臂之间没有自穿插，必须用完整视频与关键姿态人工复核。
- 物理轮胎碰撞宽度约 0.20 m，而视觉外胎宽度约 0.103 m。碰撞与外观并非完全一致；六轮接触判据依据物理几何，不能把可视网格直接当作接触测量。
- 摩擦系数、接触软化与求解响应未经过真实硬件标定。当前使用的摩擦/接触参数属于仿真条件，不能据此推导真实火星地面、真实楼梯或实车成功率。
- 控制器使用模拟器真实接触、世界位姿等特权状态，不是仅凭真实传感器可直接部署的控制器。状态可观测性差异必须在后续训练和部署中明确处理。
- 本集的自动审核只覆盖明列的物理、路线、时序与来源条件。滑移等诊断量没有额外硬门槛；自动通过不等于舒适、能耗最优、载荷均衡或硬件安全认证。
- 一个 h04 工况成功不能证明其他初始扰动、台阶高度或参数鲁棒性。数据集规模以正式接纳记录为准，不为凑数量放宽阈值或混入失败经验。

维护约定：验收规则和来源变化应重跑独立复审；更新接纳状态后应重新统计本说明的状态快照。保留旧失败记录和明确拒因，避免历史 success 字段再次被误用。


九、当前成功种子与后续生成

唯一正式接纳种子：seed1000_h04_base。
单级高度/深度：4 cm / 4 cm；12 级上行，2 m 平台，12 级下降（含最后落地）；无倒角。
仿真时长 20.577 s；视频 20.600 s（50 fps，包含准确末帧）。
20,577 条动作、20,578 行状态，全部 20,577 个物理步均通过独立审核。
最大转向角 4.6802°，最大侧倾 1.4339°，最大轮地穿透 8.6257 mm，非轮部件触地时长 0。
最后六轮分别承重、关节回正，连续稳定 1.201 s。
正式轨迹与已重复运行的固定原型逐位一致；独立动作重放的时间、位置、速度与实际力误差均为 0。
视觉范围为抽样通过，完整 1280×720/50 fps 双视角视频留供使用者复核。
这些值只描述当前种子，不能作为其它高度、初态或真实硬件的成功率。

v5 当前控制律：
- 中立摇臂目标（front_lift=middle_lift=0）；位置外环比例4、速度内环150、速度目标限0.7 rad/s、力矩限100 N·m。
- 转向位置外环8、速度内环50、速度目标限1.2 rad/s、力矩限40 N·m。
- 常规轮速P10；平台几何减速约束生效或DESCENT阶段使用P40，正反方向误差均适用；实际轮矩限50 N·m。
- 转向误差超过10°时渐减该轮正向驱动力矩，25°时降至零；保持制动力矩。
- 六轮落地后回正停车，连续稳定满足条件才结束；支撑丢失有最多3次有界低速重新落位。
- 中立策略沿用接触阶段名称，但 FRONT_LIFT/MIDDLE_LIFT 不代表发出了非零抬升角，训练应读 desired_rocker/commanded_rocker 及实际状态。

当前目标成功训练入口（15/20 cm、15级）：
/home/agilex/wzy/rover/datasets/45deg_v3/train.jsonl
全量接纳归档（含旧4 cm回归种子）：
/home/agilex/wzy/rover/datasets/45deg_v3/accepted.jsonl
请勿将 attempts.jsonl 或整个 episodes 目录直接当作成功经验集。

显式重跑旧4 cm回归工况（不作为新目标训练数据；使用全新prefix）：
cd /home/agilex/wzy/rover/scripts/rover
/home/agilex/wzy/rover/.venv/bin/python -B generate_dataset.py --split regression --heights 4 --variants base --prefix next001

相同初态和参数会产生重复轨迹，重复生成不会增加经验多样性。可使用 base,yaw_left,yaw_right 变体或调整场景生成新的候选，但每条必须重新通过完整验收；当前未验证这些变化会成功。生成器不会自动加入 accepted.jsonl。

自动合格后生成视频和审核图：
/home/agilex/wzy/rover/.venv/bin/python -B render_dataset.py /home/agilex/wzy/rover/datasets/45deg_v3/episodes/next001_regression_h04_n12_base/metadata.json --review-sheets

完成实际视觉检查后，按当前种子的 visual_review.json 结构记录检查人、范围、notes、reviewed_images、trajectory_sha256、video_sha256，再调用：
/home/agilex/wzy/rover/.venv/bin/python -B publish_dataset.py /home/agilex/wzy/rover/datasets/45deg_v3/episodes/next001_regression_h04_n12_base/metadata.json

publish_dataset 会重新独立复审，检查视频、视觉证据和哈希，再更新训练/展示清单；不能复制当前审核结论来冒充新轨迹已被查看。控制器、来源或验收规则变化后必须重新验证。


十、视觉显示修正（2026-11-11）

旧渲染器默认同时显示82个机器人碰撞体和13个视觉网格，轮子的宽圆柱与车架盒子遮住了本体细节。旧视频和旧视觉审核已归档在该episode的render_archive_collision_overlay_20261111/，其审核被新的visual_review.json替代。

当前视频只显示原URDF视觉资产和地形：碰撞体在独立渲染模型中分配到隐藏组5，原URDF visual位于可见组1。每帧检查渲染场景，拒绝碰撞体混入或视觉部件缺失。机器人碰撞体继续参与物理仿真；隐藏仅影响显示。

旧MuJoCo模型的主车架视觉资产为base_link_slim.stl（34,898面）。原base_link.stl有202,697面，超过当前MuJoCo STL导入限制。现在将原STL无损转换为OBJ供离线显示，完整保留202,697个三角面，三角顶点与法向float32逐项一致，位置误差0；未修改原文件。其他12个视觉mesh保持URDF原始引用，13个视觉节点的link、位置、缩放与URDF一致。

独立渲染资产：/home/agilex/wzy/rover/render_assets/base_link_original.obj
转换来源：/home/agilex/wzy/rover/render_assets/provenance.json
episode/render_scene.xml 是显示专用场景，不能冒充原物理场景；episode/scene.xml 和trajectory.npz的SHA256在修正前后完全不变。metadata.render.visual_geometry记录渲染场景、全部显示mesh、URDF和渲染源码哈希，render_sources/保存此次源码与转换来源。

新视频仍是相同的1,030帧、50 fps、20.600 s。重新检查了43个全程半秒抽样、19个关键姿态和初态/上楼/下楼/末态双视角帧，范围仍为sampled_frames。原来的物理独立重放证据保留，重新发布还会再次独立审核。

十一、高台阶几何审核与取景
几何审核从实际场景识别连续台阶，并强制匹配metadata.parameters声明的级数、高度与深度。新场景15级，旧种子12级；不通过改元数据来伪装级数。等高、45°、直角台阶、无间隙或重叠及其他全部物理门槛保持不变。
高台阶上楼使用楼梯下方的斜侧机位，平台上随实际位置平滑转向下楼机位；按原视觉网格包围范围取景，关键帧采用居中裁剪，避免高楼梯遮住整车或把车体顶端裁掉。这些更改仅影响后续离线显示，旧已接纳视频和物理轨迹未改写。
