"""Generate experiment and data documentation from the recorded final artifacts."""
import json
from .learning import DATA, ROOT
from .policies import MODEL_DIR


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def table(summary):
    text=['|被评策略|对手|胜 / 平 / 负|胜率|胜+半平|95%基础图区间|钻石差|淘汰|超时|',
          '|---|---|---:|---:|---:|---|---:|---:|---:|']
    for p in summary['pairs']:
        lo,hi=p['map_bootstrap_95ci_win_plus_half_draw']
        text.append(f"|{p['policy']}|{p['opponent']}|{p['wins']} / {p['draws']} / {p['losses']}|"
                    f"{p['win_rate']:.2%}|{p['win_plus_half_draw']:.3%}|{lo:.3%}–{hi:.3%}|"
                    f"{p['mean_diamond_difference']:+.4f}|{p['eliminations_policy']}/{p['games']}|{p['timeouts']}|")
    return '\n'.join(text)


def report():
    initial,retest,random=[read(DATA/'evaluation'/s/'summary.json') for s in ['initial','retest','random']]
    integrity=read(DATA/'final_integrity_audit.json')
    train,valid,corr=[read(DATA/(s+'_collection.json')) for s in ['train','validation','train_corrections']]
    refine=read(DATA/'refinement.json')
    effects=read(DATA/'evaluation/retest/paired_effects.json')
    cases=read(DATA/'evaluation/retest/verified_cases.json')
    models=[read(MODEL_DIR/(s+'_training.json')) for s in ['L_initial','L_final','L_no_opponent_initial','L_no_opponent']]
    rows=['|检查点|训练样本|轮数 / 优化步数|验证选中轮|选中训练损失|选中验证损失|验证教师动作准确率|',
          '|---|---:|---:|---:|---:|---:|---:|']
    for m in models:
        q=m['selected']
        rows.append(f"|{m['file']}|{m['train_examples']}|{m['epochs']} / {m['optimizer_steps']}|{m['selected_epoch']}|"
                    f"{q['train_loss']:.6f}|{q['validation_loss']:.6f}|{q['validation_accuracy']:.2%}|")
    effect_rows=['|比较（左减右）|固定对手|胜+半平差 / 百分点|配对95%区间 / 百分点|',
                 '|---|---|---:|---|']
    for e in effects:
        lo,hi=e['paired_map_bootstrap_95ci']
        effect_rows.append(f"|{e['left']} − {e['right']}|{e['opponent']}|"
                           f"{100*e['difference_win_plus_half_draw']:+.3f}|{100*lo:+.3f} 至 {100*hi:+.3f}|")
    case_rows=['|类型|真实对局 / 记录步号|动作|目的地风险|状态|最终胜者|',
               '|---|---|---|---:|---|---|']
    for c in cases:
        case_rows.append(f"|{c['kind']}|{c['episode_id']} / {c['step']}|{c['action']}|"
                         f"{c['risk_of_destination']:.5f}|{c['risk_status']}|{c['result']['winner']}|")
    stage_times=[read(DATA/(s+'_stage_timing.json'))['elapsed_seconds_including_collection'] for s in ['initial','refine']]
    timeout_count=sum(p['timeouts'] for s in [initial,retest,random] for p in s['pairs'])
    initial_b2=next(p for p in initial['pairs'] if p['policy']=='B2' and p['opponent']=='B1')
    ci=initial_b2['map_bootstrap_95ci_win_plus_half_draw']
    initial_note=('B2对B1每张图聚合胜+半平均为0.5，故该有限样本bootstrap区间退化为[0.5,0.5]；这不意味着总体效果已知无不确定性。'
                  if ci==[.5,.5] else '区间描述当前基础图分组重采样，不代表已识别全部总体不确定性。')
    refinement_note=' '.join(f"对{e['opponent']}，补训相对初训胜+半平差为{100*e['difference_win_plus_half_draw']:+.3f}个百分点，"
                              +('区间跨0，未获得明确改善证据。' if e['paired_map_bootstrap_95ci'][0]<=0<=e['paired_map_bootstrap_95ci'][1]
                                else '方向与区间以表中数据为准，不外推跨训练种子稳定性。')
                              for e in effects if e['right']=='L_initial')
    ablation_note=' '.join(f"对{e['opponent']}，含对手特征减匹配消融为{100*e['difference_win_plus_half_draw']:+.3f}个百分点，"
                            +('区间跨0，未显示明确收益。' if e['paired_map_bootstrap_95ci'][0]<=0<=e['paired_map_bootstrap_95ci'][1]
                              else '方向与区间见表，不外推稳定性。')
                            for e in effects if e['right']=='L_no_opponent')
    learned_b2=next(p for p in retest['pairs'] if p['policy']=='L' and p['opponent']=='B2')
    ll,lh=learned_b2['map_bootstrap_95ci_win_plus_half_draw']
    teacher_note=f"L在本复测对B2的胜+半平为{learned_b2['win_plus_half_draw']:.3%}，区间[{ll:.3%},{lh:.3%}]。"
    teacher_note+=('本实验不支持学习策略优于教师。' if ll<=.5 else '这一分区结果仍需多训练种子和更多独立环境复核。')
    text=f'''# 实际训练、正式评测与局限

本报告由 `python -m arena.report` 读取本次新运行数据生成。旧交接目录的实验产物已私人归档，以下数字不沿用旧报告。环境为Windows、Python 3.12.5、NumPy 2.3.5、CPU，无GPU、付费API或真实玩家数据。规则固定arena-v1.0：9×9、10雷、3钻石、总200行动，不筛选可达地图。

## 方法和真实训练

共200训练基础图、30验证基础图；初测、复测、随机对手测试各有100张互不重叠的基础图。私有生成种子与公开观察隔离；530图及其八种旋转/镜像等价形式经审核无重复。地图生成种子只存私人manifest，公开协议记录计数及manifest哈希。

B0随机合法动作；B1按风险路线代价选资源；B2在相同风险与动作预算上加入公开对手到达代价。学习方案为B2教师的监督模仿，27动作特征、32个tanh隐藏单元、929参数；不是强化学习。Adam学习率0.002，batch256；训练随机种子仅1个，未评估跨训练种子稳定性。每阶段选固定轮数内**验证交叉熵最小**的检查点，不使用测试胜率选型。

{chr(10).join(rows)}

初训真实采集{train['games']}局、{train['unique_states']}去重状态，验证{valid['games']}局、{valid['unique_states']}状态。训练集上的初始学习策略复跑{corr['games']}局，包含{corr['failures']}输局和{corr['disagreements']}个动作分歧，按冻结判据选出{corr['unique_states']}纠正状态；与原数据拼接得到{refine['combined_examples']}条样本。两轮之间可能有重复，不将拼接数称为跨轮唯一状态。匹配消融共享样本、标签、网络、训练随机种子和45+20轮预算，只移除显式对手特征；共享公开历史线索仍保留。

初训及补训阶段（含采集、文件记录）分别实耗{stage_times[0]:.3f}秒和{stage_times[1]:.3f}秒。四次纯优化耗时依次为{', '.join(f"{m['elapsed_seconds']:.3f}" for m in models)}秒，不能把纯优化时间当成全部实验成本。网页实际加载 `models/L_final.npz`，SHA256：`{models[1]['sha256']}`。逐轮损失、选中轮数、预算和哈希均保存在 `models/*_training.json`。

验证准确率仅衡量对启发式教师动作的模仿，不是胜率、最优性或智能涌现证据。一次实际雷位没有用作概率标签；无奖励塑形，也未预设“转移目标”标签。

## 初次正式测试：100基础图、1200局

每策略对在每图交换起点身份及先行动者，共4变体。胜/平/负均从被评策略视角；“胜+半平”只用于分析，游戏胜负仍只按钻石。区间对基础图分组重采样2000次，不把4个变体当4张独立地图。

{table(initial)}

实耗{initial['elapsed_seconds']:.3f}秒，1200局逐动作回放一致。{initial_note}先跑前5图60局pilot，单列保存，不加进正式局数或独立图数。

## 独立复测：另100基础图、2800局

复测包括B1/B2/L交叉1200局、L_initial同图复测800局和匹配消融800局。

{table(retest)}

实耗{retest['elapsed_seconds']:.3f}秒，2800局逐动作回放一致。同图配对基础图bootstrap结果如下：

{chr(10).join(effect_rows)}

{refinement_note} {ablation_note} {teacher_note} B2相对B1的结果需结合两分区与小样本局限，不能推广为普遍优势。所有负结果全部保留；未删图、重改规则或按测试成绩挑权重。

## 随机对手外部测试：另100基础图、1200局

{table(random)}

实耗{random['elapsed_seconds']:.3f}秒。B0动作随机流由评测程序固定，且独立于隐藏地图种子；各策略对使用不同随机流。因此该表用于考查不同对手，不能仅按其微小胜率差宣称B1、B2、L的优劣排序。

## 全量核验、规则现象与计算状态

5200正式局基于300张独立测试基础图，保留{integrity['draws']}平局、{timeout_count}个200步超时。所有局在评测时逐步回放，并由独立audit入口再核对一遍；`data/final_integrity_audit.json`记录原始表及轨迹哈希。实际{sum(integrity['risk_status_counts'].values())}次决策中，完整全局模型计数{integrity['risk_status_counts']['exact_global_model_count']}次，预算截断近似{integrity['risk_status_counts'].get('approximate_cutoff',0)}次。不能把全部概率标成精确。

300测试图中{integrity['test_base_maps_with_adjacent_diamond']}图至少一颗钻石与某起点正交相邻，均被保留。先行动者胜{integrity['first_player_wins']}局、后行动者胜{integrity['second_player_wins']}局、平{integrity['draws']}局；这是混合策略描述量，不是独立识别的因果先手效应。{integrity['eliminated_final_winners']}局最终赢家自身已淘汰，符合得分保留规则；不证明策略有意自我淘汰获益。

推断和决策计时为 `perf_counter` 热进程共享缓存墙钟，受顺序与机器负载影响，不能当成冷启动性能优势。每侧决策数、均时、淘汰和行动数见原始CSV与summary。所有策略共用风险预算，路径风险是加性启发式罚项，不是各格边缘概率相乘所得的精确存活率。

## 可复核真实案例

{chr(10).join(case_rows)}

对应完整回放在 `data/evaluation/retest/verified_cases/`，判据和事件状态在 `verified_cases.json`。规则策略显式输出目标；学习策略没有目标头，目标只是动作的事后路线归因。争夺候选为双方存活且当前目标等于对手上次目标；精选显式争夺另要求至少两个资源。转移要求旧、新资源当时都还存在。算法分歧对同一公开观察重算B1/B2。自动判据核验不等于真人复核或意图解释，不将案例计数当独立实验数。

## 尚未得到的结论

已完成真实环境、三基线、真实可加载模型、训练图纠正一轮、固定分区评测、匹配消融、完整失败/超时保存与复盘。未证明学习策略优于启发式、对手特征稳定有益或任何专利新颖性。只有一个训练种子、CPU小模型和自建仿真，没有真人数据、强化学习、十人扩展或真实排雷硬件。项目可用于本游戏内的教学与策略研究，不外推为工业排雷成果。
'''
    (ROOT/'docs/experiments.md').write_text(text,encoding='utf-8')
    (ROOT/'docs/experiment.md').write_text('# 实验报告\n\n请阅读[完整实验报告](experiments.md)。所有数字由当前原始日志生成。\n',encoding='utf-8')
    data=f'''# 数据说明与复现

数据全部为本项目自建环境中的程序策略合成轨迹；**真实玩家数据0条，待采集**。未读取第三方游戏或排行榜。自产源码、合成数据和训练权重按MIT公开；没有外部预训练权重或外部玩家记录。

## 分区与隔离

|分区|独立基础图|对局 / 状态|
|---|---:|---|
|教师训练|200|400局，{train['unique_states']}去重状态|
|验证|30|60局，{valid['unique_states']}去重状态|
|训练图纠正|同200|400局，{corr['unique_states']}筛选去重状态|
|初测|100|1200局|
|独立复测|100|2800局|
|随机对手测试|100|1200局|

按基础图划分，换座、换先手、镜像和快照不能跨分区。`arena.audit`核验530张基础图及其旋转/镜像无重复。补训只从训练图的失败及动作分歧挖掘，不从初测/复测失败取标签。教师集与纠正集统一使用train-mapNNN标识同一地图。各轮内部以去掉game_id/revision的完整公开状态去重；两轮拼接可能重复，{refine['combined_examples']}不称跨轮唯一数量。

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
'''
    (ROOT/'docs/data.md').write_text(data,encoding='utf-8')
    case_text='# 真实案例复核\n\n'+chr(10).join(case_rows)+'\n\n'
    for c in cases:
        case_text+=f"## {c['kind']}\n\n{c['description']} 记录步号{c['step']}，行动者{c['actor']}。核验方式：{c['verification']}。人工审核：否。\n\n"
    case_text+='导入对应完整公共回放，可检查原观察、动作、风险和终局。网络没有显式目标头；自动案例不是意图解释。见[实验报告](experiments.md)。\n'
    (ROOT/'docs/case_review.md').write_text(case_text,encoding='utf-8')
    model_text='# 模型来源与加载\n\n自产合成游戏轨迹上的B2教师监督模仿模型，MIT许可；无外部模型或玩家数据。模型均为929参数NumPy网络。\n\n'
    for m in models:
        model_text+=f"- {m['file']}：SHA256 `{m['sha256']}`，验证最优epoch {m['selected_epoch']}；真实优化日志见对应_training.json。\n"
    model_text+='\n默认L加载L_final.npz，缺权重明确报错，不退回规则策略冒充学习。只有一个训练随机种子。训练与胜率负结果见../docs/experiments.md。\n'
    (MODEL_DIR/'README.md').write_text(model_text,encoding='utf-8')
    print('Regenerated docs/experiments.md, experiment.md, data.md, case_review.md and models/README.md')


if __name__=='__main__': report()
