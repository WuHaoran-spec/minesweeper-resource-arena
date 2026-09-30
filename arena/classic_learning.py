"""Small feedback-trained classic scorer; independent secondary race benchmark.

C1 predicts observed local action utility, not exact mine probability. Targets
come only from the cell actually revealed, never unvisited hidden cells.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import time
import numpy as np

from .classic_race import ClassicRace, prepare, FEATURE_NAMES, choose

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('CLASSIC_OUTPUT_DIR', str(ROOT)))
DATA = OUT/'data'/'classic_v3'
MODELS = OUT/'models'
PRIVATE = Path(os.environ.get('CLASSIC_PRIVATE_DIR', str(ROOT.parent/'private_delivery'/'revision3'/'experiment_truth')))
COUNTS = {'train': 24, 'validation': 6, 'test': 12}
TRAINING_SEED = 20261003
_MODELS = {}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def public_only(value):
    if isinstance(value, dict):
        if {'seed', 'map_seed', 'mines', 'hidden_mines', 'private_snapshot'} & value.keys():
            raise AssertionError('Private truth in public classic data')
        for item in value.values(): public_only(item)
    elif isinstance(value, list):
        for item in value: public_only(item)


def canonical(race):
    board = race._boards[0]; forms = []
    for reflect in (False, True):
        for turns in range(4):
            def transform(cell):
                r,c = cell; width,height = board.width,board.height
                if reflect: c = width-1-c
                for _ in range(turns): r,c,width,height = c,height-1-r,height,width
                return r,c
            forms.append(json.dumps(sorted(transform(cell) for cell in board._mines)))
    return hashlib.sha256(min(forms).encode()).hexdigest()


def freeze():
    if PRIVATE.resolve().is_relative_to(ROOT.resolve()) or PRIVATE.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Private map manifest must be outside public output')
    path = PRIVATE/'classic_manifest.json'
    if path.exists(): truth = json.loads(path.read_text(encoding='utf-8'))
    else:
        used = set(); splits = {}; fingerprints = {}
        for split, count in COUNTS.items():
            values = []; hashes = []
            while len(values) < count:
                seed = secrets.randbits(63); fingerprint = canonical(ClassicRace(seed))
                if fingerprint in used: continue
                used.add(fingerprint); values.append(seed); hashes.append(fingerprint)
            splits[split] = values; fingerprints[split] = hashes
        truth = {'private': True, 'preset': 'beginner', 'splits': splits, 'canonical_fingerprints': fingerprints}
        write(path, truth)
    assert {k: len(v) for k,v in truth['splits'].items()} == COUNTS
    protocol = {'version': 'classic-race-feedback-v3-pilot', 'preset': 'beginner', 'dimensions': [9,9], 'mine_count': 10,
        'secondary_benchmark': True, 'same_map_independent_public_observations': True,
        'automatic_protected_first_cell': [4,4], 'active_reveal_count_excludes_automatic_first': True,
        'ranking': ['cleared', 'safe_revealed', 'fewer_active_reveals'], 'computation_time_affects_rank': False,
        'independent_maps': COUNTS, 'manifest_sha256': digest(path),
        'collection': 'one R0 versus R1 race per training/validation map; every actually selected action retained',
        'architecture': [10,16,1], 'activation': 'tanh hidden, linear output',
        'feature_names': FEATURE_NAMES, 'training_seed': TRAINING_SEED, 'epochs': 100,
        'optimizer': 'full-batch Adam, learning rate 0.005',
        'target': 'safe_result_indicator + 0.25*min(new_safe_revealed,10)/10',
        'target_is_mine_probability': False, 'teacher_action_labels_used': False,
        'unselected_hidden_cell_labels_used': False,
        'selection': 'lowest validation MSE among epoch 0 and every 10 epochs; earlier on ties',
        'test': '12 maps x three unordered pairs of R0/R1/C1 x two seat assignments x two first players = 144 races',
        'planned_test_races': 144, 'failures_retained': True, 'human_records': 0,
        'source': 'automated generated classic boards, selected-action observations and feedback'}
    public_path = DATA/'protocol.json'
    if public_path.exists() and json.loads(public_path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Protocol mismatch; use a new output directory')
    write(public_path, protocol)
    return truth, protocol


def observed_utility(feedback):
    return float(not feedback['exploded']) + .25*min(feedback['new_safe_revealed'],10)/10


def forward(X, model):
    normalized = (X-model['mean'])/model['scale']
    hidden = np.tanh(normalized @ model['W1']+model['b1'])
    return (hidden @ model['W2']+model['b2']).reshape(-1)


def choose_model(observation, rng=None):
    path = MODELS/'classic_v3_final.npz'; key = (str(path), path.stat().st_mtime_ns)
    if key not in _MODELS:
        with np.load(path, allow_pickle=False) as data: model = {k: data[k].copy() for k in data.files}
        _MODELS[key] = model, digest(path)
    model, sha = _MODELS[key]
    X,cells,risk,status,detail = prepare(observation); scores = forward(X,model); index = int(np.argmax(scores))
    diagnostic = {'method': 'observed local utility regression', 'architecture': [10,16,1], 'model_sha256': sha,
        'score_is_mine_probability': False, 'selected_score': float(scores[index]),
        'selected_features': X[index].tolist(), 'feature_names': FEATURE_NAMES,
        'candidate_scores': [{'cell':cell,'score':float(score)} for cell,score in zip(cells,scores)], 'risk':detail}
    if observation.get('preset') != 'beginner': diagnostic['out_of_distribution'] = 'C1 trained only on classic beginner 9x9/10'
    return {'cell':cells[index], 'action':cells[index], 'policy':'C1', 'risk_status':status,'risk':risk,'diagnostics':diagnostic}


def play(seed, policies, first, identifier, map_id, stream, collect=False):
    race = ClassicRace(seed, first=first); rng = random.Random(91001+first)
    examples = []; began = time.perf_counter()
    while not race.done:
        actor = race.turn; obs = race.current_observation(); start = time.perf_counter()
        decision = choose(obs, policies[actor], rng)
        duration = time.perf_counter()-start
        X,cells,*_ = prepare(obs) if collect else (None,None)
        selected = X[cells.index(decision['cell'])].copy() if collect else None
        race.step(decision['cell'], decision, duration)
        if collect:
            feedback = race._history[-1]['feedback']
            examples.append({'map_id':map_id,'episode_id':identifier,'step':race.revision,'actor':actor,
                'behavior_policy':policies[actor],'features':selected.tolist(), 'target':observed_utility(feedback),
                'feedback':feedback,'risk_status':decision['risk_status'], 'label_source':'actual_selected_action_feedback'})
    final = race.observe(); row = {'episode_id':identifier,'map_id':map_id,'policies':policies,'first':first,
        'winner':final['winner'],'results':final['results'],'active_reveals':final['active_reveals'],
        'computation_seconds':final['computation_seconds'],'elapsed_seconds':time.perf_counter()-began}
    item = {**row,'replay':race.save_replay(),'human_record':False}; public_only(item)
    stream.write(json.dumps(item,separators=(',',':'))+'\n'); stream.flush()
    return row, examples


def collect(truth):
    examples = {}; outcomes = {}
    for split in ('train','validation'):
        rows = []; games = []
        with gzip.open(DATA/f'{split}_public_replays.jsonl.gz','wt',encoding='utf-8') as stream:
            for index, seed in enumerate(truth['splits'][split]):
                row,samples = play(seed,['R0','R1'],index%2,f'{split}-m{index:02d}',f'classic-{split}-m{index:02d}',stream,True)
                rows.extend(samples); games.append(row)
        with gzip.open(DATA/f'{split}_examples.jsonl.gz','wt',encoding='utf-8') as stream:
            for row in rows: stream.write(json.dumps(row,separators=(',',':'))+'\n')
        examples[split] = rows; outcomes[split] = games
        print(f'Classic {split}: {len(games)} races, {len(rows)} selected-action examples',flush=True)
    write(DATA/'collection_outcomes.json',outcomes)
    return examples


def train():
    if (DATA/'training_summary.json').exists(): raise FileExistsError('Preserve existing run; set a new CLASSIC_OUTPUT_DIR')
    truth,protocol = freeze(); began = time.perf_counter(); rows = collect(truth)
    arrays = {}
    for split,values in rows.items():
        arrays[split] = (np.array([r['features'] for r in values]),np.array([r['target'] for r in values]))
    X,y = arrays['train']; V,vy = arrays['validation']; rng = np.random.default_rng(TRAINING_SEED)
    model = {'mean':X.mean(axis=0),'scale':np.maximum(X.std(axis=0),.05),
             'W1':rng.normal(0,.2,(10,16)),'b1':np.zeros(16),'W2':rng.normal(0,.2,(16,1)),'b2':np.zeros(1)}
    MODELS.mkdir(parents=True,exist_ok=True); np.savez_compressed(MODELS/'classic_v3_initial.npz',**model)
    normalized = (X-model['mean'])/model['scale']; history = []; best = None; best_loss = float('inf'); best_epoch = None
    m = {k:np.zeros_like(model[k]) for k in ('W1','b1','W2','b2')}; v = {k:value.copy() for k,value in m.items()}
    for epoch in range(protocol['epochs']+1):
        if epoch%10 == 0:
            loss = float(np.mean((forward(X,model)-y)**2)); validation = float(np.mean((forward(V,model)-vy)**2))
            history.append({'epoch':epoch,'training_mse':loss,'validation_mse':validation})
            if validation < best_loss: best_loss=validation; best_epoch=epoch; best={k:a.copy() for k,a in model.items()}
        if epoch == protocol['epochs']: break
        hidden = np.tanh(normalized@model['W1']+model['b1']); error = (hidden@model['W2']+model['b2']).reshape(-1)-y
        output_gradient = (2*error/len(X))[:,None]; hidden_gradient = (output_gradient@model['W2'].T)*(1-hidden**2)
        gradients = {'W2':hidden.T@output_gradient,'b2':output_gradient.sum(axis=0),
                     'W1':normalized.T@hidden_gradient,'b1':hidden_gradient.sum(axis=0)}
        for key, gradient in gradients.items():
            m[key]=.9*m[key]+.1*gradient; v[key]=.999*v[key]+.001*gradient**2
            model[key] -= .005*(m[key]/(1-.9**(epoch+1)))/(np.sqrt(v[key]/(1-.999**(epoch+1)))+1e-8)
    np.savez_compressed(MODELS/'classic_v3_final.npz',**best)
    np.savez_compressed(DATA/'selected_action_features.npz',train_X=X,train_y=y,validation_X=V,validation_y=vy)
    summary = {'architecture':[10,16,1],'method':'supervised regression of actually observed local reveal utility',
        'teacher_action_labels_used':False,'probability_labels_from_single_outcomes':False,
        'training_examples':len(X),'validation_examples':len(V),'independent_maps':COUNTS,
        'training_seed':TRAINING_SEED,'epochs_completed':protocol['epochs'],'selected_epoch':best_epoch,
        'selected_validation_mse':best_loss,'history':history,'elapsed_seconds':time.perf_counter()-began,
        'model_sha256':digest(MODELS/'classic_v3_final.npz'),'initial_sha256':digest(MODELS/'classic_v3_initial.npz'),
        'protocol_sha256':digest(DATA/'protocol.json'),'test_used_for_selection':False}
    write(DATA/'training_summary.json',summary);write(MODELS/'classic_v3_training.json',summary)
    print(f'Classic C1 trained: selected epoch {best_epoch}, validation MSE={best_loss:.6f}',flush=True)


def evaluate():
    if (DATA/'test_summary.json').exists(): raise FileExistsError('Preserve existing test; set a new CLASSIC_OUTPUT_DIR')
    truth,protocol=freeze(); before=digest(MODELS/'classic_v3_final.npz'); write(DATA/'test_frozen_checkpoint.json',{'model_sha256':before,'before_test':True})
    rows=[]; began=time.perf_counter()
    with gzip.open(DATA/'test_public_replays.jsonl.gz','wt',encoding='utf-8') as stream:
        for index,seed in enumerate(truth['splits']['test']):
            for pair in (('R0','R1'),('R0','C1'),('R1','C1')):
                for swapped in (False,True):
                    policies=list(reversed(pair)) if swapped else list(pair)
                    for first in (0,1):
                        identifier=f'test-m{index:02d}-{pair[0]}-{pair[1]}-s{int(swapped)}-f{first}'
                        row,_=play(seed,policies,first,identifier,f'classic-test-m{index:02d}',stream);rows.append(row)
            print(f'Classic held-out {len(rows)}/144 races',flush=True)
    assert before==digest(MODELS/'classic_v3_final.npz')
    groups={}
    for policy in ('R0','R1','C1'):
        items=[(row,row['policies'].index(policy)) for row in rows if policy in row['policies']]
        groups[policy]={'board_runs':len(items),'clears':sum(row['results'][actor]['cleared'] for row,actor in items),
            'mine_failures':sum(row['results'][actor]['reason']=='mine' for row,actor in items),
            'race_wins':sum(row['winner']==actor for row,actor in items),'race_draws':sum(row['winner']=='draw' for row,actor in items),
            'race_losses':sum(row['winner'] not in (actor,'draw') for row,actor in items),
            'mean_safe_revealed':float(np.mean([row['results'][actor]['safe_revealed'] for row,actor in items])),
            'median_active_reveals':float(np.median([row['active_reveals'][actor] for row,actor in items]))}
    summary={'completed_races':len(rows),'planned_races':144,'independent_test_maps':12,'groups':groups,
        'model_sha256':before,'frozen_checkpoint_unchanged':True,'elapsed_seconds':time.perf_counter()-began,
        'all_failures_retained':True,'limitation':'Only 12 independent beginner maps; seat/first repetitions are not independent samples',
        'human_records':0}
    write(DATA/'test_games.json',rows);write(DATA/'test_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


def report():
    train=json.loads((DATA/'training_summary.json').read_text(encoding='utf-8'));test=json.loads((DATA/'test_summary.json').read_text(encoding='utf-8'))
    text='# 经典扫雷同图独立竞速与 C1 小模型\n\n这是辅助基准，主项目仍是共享棋盘资源争夺。两位 AI 各有独立信息，在相同固定雷图上从同一个受保护中心首击开始。安全展开沿用经典规则；每轮只能主动揭一格未知格。踩雷只结束自己的棋盘，另一方继续。双方结束后依次比较通关、已揭安全格数、较少主动揭格步数；首击不计主动步，推理耗时只报告，不判胜负。\n\n'
    text+='R0 随机选择未揭格；R1 优先选择最低公开后验风险，再用公开邻域信息作为同风险时的偏好。C1 是 10→16→1、tanh 隐层的本地小网络。它回归实际执行动作的局部效用：安全结果指标 + 0.25×min(新增安全格,10)/10。只标注实际点击反馈，不读取其他隐藏格，不把一次成败当成精确雷概率，也不用 R1 动作作为教师标签。特征仍包括人工推断的雷风险，不是从像素端到端学习。\n\n'
    text+=f'基础地图按 24 训练、6 验证、12 测试划分，并检查 D4 镜像/旋转不重。真实采集 {train["training_examples"]} 个训练、{train["validation_examples"]} 个验证动作反馈，Adam 训练 100 轮，按验证 MSE 选择第 {train["selected_epoch"]} 轮，MSE {train["selected_validation_mse"]:.6f}。测试完全不参与选模。\n\n'
    text+='|策略|棋盘运行数|通关/踩雷|竞速胜/平/负|平均安全格|主动揭格中位数|\n|---|---:|---|---|---:|---:|\n'
    for policy,row in test['groups'].items():
        text+=f'|{policy}|{row["board_runs"]}|{row["clears"]}/{row["mine_failures"]}|{row["race_wins"]}/{row["race_draws"]}/{row["race_losses"]}|{row["mean_safe_revealed"]:.2f}|{row["median_active_reveals"]:.1f}|\n'
    text+='\n本次固定留出中，C1 通关数低于 R1；C1 的部分竞速胜出来自通关后较少的主动揭格步数，不能将混合对手总胜场当成通关能力或普遍优越性。每策略的 96 次运行仍只覆盖相同 12 张基础图。C1 在中级/专家棋盘可以运行，但明确标为训练分布之外，尚未作该尺寸的正式训练验证。\n'
    text+=f'\n真实完成 {test["completed_races"]} 场竞速，全部失败保留。只有 12 张独立测试地图，换座与先后手不能当作更多独立样本；不能预设 C1 超过 R1。终局经典模式自动展示的全部雷位在竞速公开观察和数据中被清洗，仅保留实际踩中的雷。决策器在终局被拒绝调用，且只接收自己的棋盘。\n\n最终模型 SHA-256：`{train["model_sha256"]}`。独立新输出目录复现：设置 `CLASSIC_OUTPUT_DIR`，私有地图目录使用仓库外的 `CLASSIC_PRIVATE_DIR`，执行 `python -m arena.classic_learning all`。这批数据全部是自动仿真，真人记录为 0。\n'
    path=OUT/'docs'/'classic_race.md';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')


def audit():
    truth=json.loads((PRIVATE/'classic_manifest.json').read_text(encoding='utf-8'))
    protocol=json.loads((DATA/'protocol.json').read_text(encoding='utf-8'))
    assert digest(PRIVATE/'classic_manifest.json')==protocol['manifest_sha256']
    seen=set()
    for split,values in truth['splits'].items():
        for seed in values:
            value=canonical(ClassicRace(seed));assert value not in seen;seen.add(value)
    totals=Counter();outcomes=Counter();labels=Counter();status=Counter()
    for split in ('train','validation','test'):
        samples={}
        if split!='test':
            with gzip.open(DATA/f'{split}_examples.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    sample=json.loads(line);key=(sample['episode_id'],sample['step'])
                    assert key not in samples;samples[key]=sample
        used=set()
        with gzip.open(DATA/f'{split}_public_replays.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line);public_only(item);replay=item['replay']
                index=int(item['map_id'].rsplit('-m',1)[1]);race=ClassicRace(truth['splits'][split][index],first=item['first'])
                initial=replay['initial_observation'];race.game_id=initial['game_id']
                for actor in (0,1):race._boards[actor].game_id=initial['boards'][actor]['game_id']
                assert race.observe()==initial
                for record in replay['records']:
                    assert race.observe()==record['observation']
                    assert race.current_observation()==record['decision_observation']
                    assert not record['decision_observation']['done']
                    if split!='test':
                        key=(item['episode_id'],record['step']);sample=samples[key];used.add(key)
                        X,cells,_,computation_status,_=prepare(record['decision_observation'])
                        np.testing.assert_allclose(sample['features'],X[cells.index(record['action'])],rtol=0,atol=0)
                        assert sample['target']==observed_utility(record['feedback'])
                        assert sample['feedback']==record['feedback']
                        assert sample['label_source']=='actual_selected_action_feedback'
                        labels[split]+=1;status[computation_status]+=1
                    actual=race.step(record['action'],record['metadata'],record['computation_seconds'])
                    assert actual==record['next_observation']
                    assert race._history[-1]['feedback']==record['feedback']
                    for board in actual['boards']:
                        assert sum(n<0 for r,c,n in board['revealed'])<=1,'Terminal mine map leaked'
                    totals['steps']+=1
                assert race.done and race.observe()==replay['final_observation']
                assert race.observe()['winner']==item['winner']
                assert race.observe()['results']==item['results']
                totals[split]+=1
                for result in item['results']:outcomes[result['reason']]+=1
        assert used==set(samples)
    with np.load(DATA/'selected_action_features.npz',allow_pickle=False) as data:
        for split in ('train','validation'):
            with gzip.open(DATA/f'{split}_examples.jsonl.gz','rt',encoding='utf-8') as stream:rows=[json.loads(line) for line in stream]
            np.testing.assert_array_equal(data[split+'_X'],[r['features'] for r in rows])
            np.testing.assert_array_equal(data[split+'_y'],[r['target'] for r in rows])
    training=json.loads((DATA/'training_summary.json').read_text(encoding='utf-8'))
    best=min(training['history'],key=lambda row:row['validation_mse'])
    assert best['epoch']==training['selected_epoch']
    with np.load(MODELS/'classic_v3_final.npz',allow_pickle=False) as model:
        with np.load(DATA/'selected_action_features.npz',allow_pickle=False) as data:
            mse=float(np.mean((forward(data['validation_X'],model)-data['validation_y'])**2))
    assert abs(mse-training['selected_validation_mse'])<1e-12
    assert digest(MODELS/'classic_v3_final.npz')==training['model_sha256']
    frozen=json.loads((DATA/'test_frozen_checkpoint.json').read_text(encoding='utf-8'))
    assert frozen['model_sha256']==training['model_sha256']
    result={'passed':True,'reexecuted_races':sum(totals[k] for k in ('train','validation','test')),
        'reexecuted_active_reveals':totals['steps'],'races_by_split':{k:totals[k] for k in ('train','validation','test')},
        'feedback_labels_verified':dict(labels),'risk_statuses_of_collected_feedback':dict(status),
        'all_board_outcomes_retained':dict(outcomes),'geometrically_distinct_maps':len(seen),
        'terminal_full_mine_map_scrubbing_checked':True,'each_decision_only_current_board':True,
        'validation_selected_epoch_recomputed':True,'frozen_weights_unchanged':True,
        'limitation':'Checks data provenance and replay semantics, not strategic optimality or general superiority'}
    write(DATA/'audit.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=('all','train','evaluate','report','audit'));args=parser.parse_args()
    if args.stage in ('all','train'):train()
    if args.stage in ('all','evaluate'):evaluate()
    if args.stage in ('all','report'):report()
    if args.stage in ('all','audit'):audit()


if __name__=='__main__':main()
