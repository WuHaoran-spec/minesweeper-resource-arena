# 数据说明与复现

数据全部为本项目自建环境中的程序策略合成轨迹；**真实玩家数据0条，待采集**。未读取第三方游戏或排行榜。自产源码、合成数据和训练权重按MIT公开；没有外部预训练权重或外部玩家记录。

## V1分区与隔离

|分区|独立基础图|对局 / 状态|
|---|---:|---|
|教师训练|200|400局，5722去重状态|
|验证|30|60局，891去重状态|
|训练图纠正|同200|400局，3217筛选去重状态|
|初测|100|1200局|
|独立复测|100|2800局|
|随机对手测试|100|1200局|

按基础图划分，换座、换先手、镜像和快照不能跨分区。`arena.audit`核验530张基础图及其旋转/镜像无重复。补训只从训练图的失败及动作分歧挖掘，不从初测/复测失败取标签。教师集与纠正集统一使用train-mapNNN标识同一地图。各轮内部以去掉game_id/revision的完整公开状态去重；两轮拼接可能重复，8939不称跨轮唯一数量。

`data/protocol.json`在训练前冻结。生成种子只存仓库外私人目录 `map_manifest.json`，公开协议只含各分区数量和私有manifest哈希。环境私有真值、模型观察和未来终局结果分离；已公开踩雷可在观察中以-1线索出现。策略不读取私有manifest，不接收环境对象；合法动作掩码不使用未知雷真值。训练随机种子和B0动作随机种子不是基础地图种子，可以公开。

## V1数据schema和质量

- `*_public_replays.jsonl.gz`：每行完整对局、map_id、来源、replay。教师源为heuristic_teacher_data；纠正源为learning_policy_rollout_with_teacher_corrections。没有把程序纠正称作人工纠正。
- `*_features.npz`：X和X_no_opponent为[N,5,27]，y为B2动作索引，mask为[N,5]公开合法动作；用allow_pickle=False读取。
- `evaluation/*/games.csv`：逐局策略对、匿名基础图ID、换座、先后手、得分、胜者、淘汰、行动数、原因、计时。
- 每条回放record含规则版本、game_id、step、actor、observation、action、metadata、feedback、next_observation。observation包含双方位置/分数/存活、资源、已揭线索、安全条件、合法动作。metadata含策略版本、81格风险及计算状态、真实cost/logit、目标及其来源。终局仅在replay.result及终局观察中保存。
- `summary.json`、`paired_effects.json`按基础图分组产生，不把行动数或座位变体当独立地图；全量失败、平局和超时均保留。
- `verified_cases.json`及完整回放可核验显式争夺、目标切换、分歧、学习失败与近似风险；自动核验不冒充人工审核。

本次5200正式局全部两次逐动作重放一致。`final_integrity_audit.json`核验模型形状/有限值/哈希、验证最优选择、分区、标签合法性、所有CSV终局及公共回放字段；还从860局公开教师、验证及纠正回放逐条重新生成全部特征、掩码和教师标签，与三份NPZ每个元素完全相同。规则环境还会拒绝含私有真值字段的metadata。私人快照只供可信本地重放分支，不能作为策略输入；实际人类纠正数据尚未采集。

## V1复现命令

已有公共数据可直接检查模型与统计；`python -m arena.analyze`重算配对统计和案例，`python -m arena.report`生成报告。逐步再演算隐藏环境需要私人提交包里的manifest，这一限制是主动隔离测试真值。公共回放可直接在网页导入查看，无须地图真值。

从公开克隆开始一套**新的**同规模实验（新基础图，因此不承诺逐数相同），PowerShell：

```powershell
$env:ARENA_OUTPUT_DIR = Join-Path (Get-Location) 'runs/fresh'
$env:ARENA_PRIVATE_DIR = Join-Path (Split-Path (Get-Location)) 'arena-private-fresh'
python -m arena.learning initial
python -m arena.experiment initial --limit 5
python -m arena.experiment initial
python -m arena.learning refine
python -m arena.experiment retest
python -m arena.experiment random
python -m arena.analyze
python -m arena.audit
```

模型与数据写入runs/fresh，私有manifest保存在仓库外；设置ARENA_OUTPUT_DIR时游戏也从该目录的models加载权重。对本次完整私人交付进行精确重算时，指定所附同一私有manifest所在目录和新的输出目录；不要覆盖已有结果。协议不同会明确报错，不能默默重用已有日志。不同NumPy/BLAS版本可能产生微小浮点与决策差异。

初测前5图60局pilot与正式初测重叠，单列保存，不另算独立图。训练期间只有一个模型随机种子，未声称跨训练种子稳定。

## V2大棋盘数据与质量边界

V2单独使用`data/challenge_v2/`、`models/challenge_*.npz`和仓库外`challenge_manifest.json`。中级16×16/40雷/7资源与高级30列×16行/99雷/11资源各自按基础地图划分训练、验证和测试；换座、先手、镜像、快照与回放分支仍属于同一图。V1的530张图、5200正式局及27维特征表不能当作V2数量。

本轮已完成68张基础图的小规模验证（每档20训练、6验证、8测试）：80局教师采集、24局验证、80局训练图纠正及384局正式测试。三份特征文件分别有9265、2385、4463条选中状态；全部16113条特征、合法掩码和教师标签从公开转移逐元素重建一致，568局完整回放再次重执行一致。另48局pilot与正式测试重叠，单列不算独立样本。真实结果和局限见[大棋盘实验报告](challenge_v2_experiments.md)，原始审计见`data/challenge_v2/integrity_audit.json`。

- `protocol.json`在采集前冻结每档地图数量、规则、训练预算与六组测试策略对；模型与策略文件使用哈希标识。
- `*_transitions.jsonl.gz`每行一条完整公共转移，含preset、map_id、split、来源、实际行动与反馈、B2教师动作及其来源、是否入选特征及是否实际用于训练。验证特征不计作训练样本。终局标签另存，不作为特征输入。
- `*_public_replays.jsonl.gz`保留完整对局，失败、平局、超时均不因训练采样而丢弃。
- `*_features.npz`的X为[N,5,32]，mask为[N,5]公共合法动作，y为启发式教师动作索引。先对公开状态去除不透明ID/revision后去重，再对每局合格候选最多等距保留192条；记录数量不等于独立地图或跨轮唯一状态。
- `evaluation`按尺寸和策略对分别记录实际终局及轨迹长度；同图四个座位/先手变体为相关样本。`integrity_audit.json`用于核对地图重复、模型哈希与验证选择、公共字段、逐局重放，以及从公开转移重建全部已选特征与标签。

大棋盘、更多轨迹与可重算标签都不等于最优数据。风险可能因计数预算变为近似，教师可能失败，稀疏或重复状态会影响学习；应分别报告来源、精确/近似状态、实际训练样本、留出结果及失败。完整训练/评测审计完成后才依据日志标记验收，不能把流程代码当作执行证据。V2目前同样没有真人数据或外部预训练模型。

在新的输出目录复现V2可运行`python -m arena.challenge_training all`，或依次执行`collect`、`initial`、`refine`、`evaluate`、`audit`、`report`。沿用前述`ARENA_OUTPUT_DIR`和仓库外`ARENA_PRIVATE_DIR`设置，避免覆盖既有结果；协议冲突会报错。无需真实雷图的公开复盘可直接导入网页；从头生成同一隐藏环境仍需私人manifest。

## E探索轨迹

`data/exploration_v2`与上述教师模仿数据分开。E搜索轨迹来自程序参数候选及对手池，记录参数版本/哈希、实际评分、探索概率、动作与结果，没有教师动作标签。终局目标用于训练候选排序；验证图选择冻结权重，独立测试图仅作确认。协议为6张训练、2张验证、4张测试中级图；实际完成量见[探索报告](exploration.md)。

E的数据支持分析具体发生过的等待、风险进入、绕路和推断目标切换，不能仅因自动化统计命名而称作“涌现策略”。发现阶段与独立确认阶段分别报告，失败、循环、超时及未发现稳定新行为都属于有效结果。它同样不是人类玩家数据。
