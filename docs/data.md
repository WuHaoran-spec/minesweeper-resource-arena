# 数据说明与复现

数据全部为本项目自建环境中的程序策略合成轨迹；**真实玩家数据0条，待采集**。未读取第三方游戏或排行榜。自产源码、合成数据和训练权重按MIT公开；没有外部预训练权重或外部玩家记录。

## 分区与隔离

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

## 数据schema和质量

- `*_public_replays.jsonl.gz`：每行完整对局、map_id、来源、replay。教师源为heuristic_teacher_data；纠正源为learning_policy_rollout_with_teacher_corrections。没有把程序纠正称作人工纠正。
- `*_features.npz`：X和X_no_opponent为[N,5,27]，y为B2动作索引，mask为[N,5]公开合法动作；用allow_pickle=False读取。
- `evaluation/*/games.csv`：逐局策略对、匿名基础图ID、换座、先后手、得分、胜者、淘汰、行动数、原因、计时。
- 每条回放record含规则版本、game_id、step、actor、observation、action、metadata、feedback、next_observation。observation包含双方位置/分数/存活、资源、已揭线索、安全条件、合法动作。metadata含策略版本、81格风险及计算状态、真实cost/logit、目标及其来源。终局仅在replay.result及终局观察中保存。
- `summary.json`、`paired_effects.json`按基础图分组产生，不把行动数或座位变体当独立地图；全量失败、平局和超时均保留。
- `verified_cases.json`及完整回放可核验显式争夺、目标切换、分歧、学习失败与近似风险；自动核验不冒充人工审核。

本次5200正式局全部两次逐动作重放一致。`final_integrity_audit.json`核验模型形状/有限值/哈希、验证最优选择、分区、标签合法性、所有CSV终局及公共回放字段；还从860局公开教师、验证及纠正回放逐条重新生成全部特征、掩码和教师标签，与三份NPZ每个元素完全相同。规则环境还会拒绝含私有真值字段的metadata。私人快照只供可信本地重放分支，不能作为策略输入；实际人类纠正数据尚未采集。

## 复现命令

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
