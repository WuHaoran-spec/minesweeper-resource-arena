"""Independent survival-v1 CPU imitation experiment; no old outputs overwritten.

Private seeds are read only by this trusted runner. B3/L3 see public observations.
python -m arena.survival_training all
"""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import random
import secrets
import time
import numpy as np
from .env import Arena
from .policies import ACTIONS, MODEL_DIR, choose, logits
from .survival import FEATURES, HIDDEN, PARAMETERS, prepare_survival
from .challenge_training import sha, write, read, canonical, public_only, quantiles

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('ARENA_OUTPUT_DIR', str(ROOT)))
DATA = OUT / 'data' / 'survival_v3'
PRIVATE = Path(os.environ.get('ARENA_PRIVATE_DIR', str(ROOT.parent / 'private_delivery' / 'revision3' / 'experiment_truth')))
COUNTS = {'train': 12, 'validation': 4, 'test': 6}
PRESETS = ('intermediate', 'expert')
OPTIMIZER_SEED = 20261007
PAIRS = [('B2', 'B3'), ('B2', 'L3'), ('L_v2', 'L3'), ('B3', 'L3'), ('B2', 'L3_initial')]


def manifest():
    if PRIVATE.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Private manifest must stay outside public repository')
    path = PRIVATE / 'survival_manifest.json'
    if path.exists():
        obj = read(path)
        assert all(len(obj['splits'][p][s]) == n for p in PRESETS for s, n in COUNTS.items())
        return obj
    seen, used, splits, fingerprints = set(), set(), {}, {}
    for preset in PRESETS:
        splits[preset], fingerprints[preset] = {}, {}
        for split, count in COUNTS.items():
            values, prints = [], []
            while len(values) < count:
                value = secrets.randbits(63)
                env = Arena(value, preset=preset, scoring='survival-v1')
                fingerprint = canonical(env)
                if value in used or fingerprint in seen:
                    continue
                used.add(value); seen.add(fingerprint); values.append(value); prints.append(fingerprint)
            splits[preset][split] = values
            fingerprints[preset][split] = prints
    obj = {'private': True, 'rule_version': 'arena-v3.0', 'scoring': 'survival-v1',
           'splits': splits, 'canonical_fingerprints': fingerprints}
    write(path, obj)
    return obj


def freeze():
    truth = manifest()
    protocol = {'protocol_version': 'survival-v3-cpu-pilot-1', 'rule_version': 'arena-v3.0',
        'scoring': 'survival-v1', 'utility': '100*gems + 10*lives - own_actions/max_steps',
        'score_order': 'gems first; then lives; then own actions; not a guarantee of risk aversion',
        'base_maps_per_preset': COUNTS, 'presets': list(PRESETS),
        'private_manifest_sha256': sha(PRIVATE/'survival_manifest.json'),
        'split_unit': 'base map including swapped seats, first turn, rotations and mirrors',
        'collection_games_per_map': 2, 'formal_pairs': [list(p) for p in PAIRS],
        'formal_variants': 'swap false/true x first 0/1', 'formal_games': 240,
        'method': 'B3 public-state heuristic imitation, one train-only correction round',
        'teacher_is_optimal': False, 'features': FEATURES, 'hidden': HIDDEN, 'parameters': PARAMETERS,
        'optimizer': 'Adam', 'optimizer_seed': OPTIMIZER_SEED, 'optimizer_runs': 1,
        'batch_size': 256, 'learning_rate': .002, 'initial_epochs': 35, 'correction_epochs': 18,
        'checkpoint_selection': 'lowest validation cross entropy within fixed epoch budget',
        'correction_selection': 'training-map learner losses or teacher-action disagreements',
        'state_selection': 'deduplicate public observation except game_id/revision; <=192 evenly spaced eligible states per episode',
        'risk_weights_by_lives': {'3': 8., '2': 18., '1': 72.},
        'history': 'each player last 24 public post-action positions; no hidden map',
        'timeouts_deaths_losses': 'retain every full raw trajectory; never filter test maps by outcome',
        'wall_clock': 'reported only; never reward or feature', 'human_records': 0,
        'source_sha256': {name: sha(ROOT/name) for name in
             ['arena/env.py', 'arena/risk.py', 'arena/survival.py', 'arena/policies.py']}}
    path = DATA/'protocol.json'
    if path.exists() and read(path) != protocol:
        raise RuntimeError('Frozen survival protocol changed; create an explicit new experiment')
    if not path.exists():
        write(path, protocol)
    return truth, protocol


def comparable(obs):
    return {k: v for k, v in obs.items() if k != 'game_id'}


def verify(seed, preset, first, swap, replay):
    env = Arena(seed, first=first, swap=swap, preset=preset, scoring='survival-v1')
    assert comparable(env.observe()) == comparable(replay['initial_observation'])
    for record in replay['records']:
        assert comparable(env.observe()) == comparable(record['observation'])
        assert comparable(env.step(record['action'])) == comparable(record['next_observation'])
    assert comparable(env.observe()) == comparable(replay['final_observation'])


def model_hash(policy):
    name = {'L3': 'survival_v3_final.npz', 'L3_initial': 'survival_v3_initial.npz',
            'L_v2': 'challenge_final.npz'}.get(policy)
    return sha(MODEL_DIR/name) if name else None


def collect(split='train', correction=False, *, frozen=None, data_dir=None, balance_first=False):
    truth, protocol = freeze() if frozen is None else frozen
    destination = DATA if data_dir is None else Path(data_dir)
    destination.mkdir(parents=True, exist_ok=True)
    name = 'corrections' if correction else split
    arrays = {k: [] for k in ('X', 'y', 'mask')}
    counts, risk_counts, seen, lengths = Counter(), Counter(), set(), []
    began = time.perf_counter()
    with gzip.open(destination/(name+'_public_replays.jsonl.gz'), 'wt', encoding='utf-8') as replay_file, \
         gzip.open(destination/(name+'_transitions.jsonl.gz'), 'wt', encoding='utf-8') as transition_file:
        for preset in PRESETS:
            for mi, seed in enumerate(truth['splits'][preset][split]):
                for variant in (0, 1):
                    first, swap, learner = variant, bool(mi % 2), variant
                    if balance_first:
                        first, swap = variant ^ (mi % 2), bool((mi // 2) % 2)
                    env = Arena(seed, first=first, swap=swap, preset=preset, scoring='survival-v1')
                    opponent = ('B2', 'L_v2', 'B3')[mi % 3]
                    policies = [opponent, opponent]
                    policies[learner] = 'L3_initial' if correction else 'B3'
                    rng, pending = random.Random(9817+mi*17+variant), []
                    while not env.done:
                        obs = env.observe(); actor = obs['turn']
                        teacher = choose(obs, 'B3')
                        decision = teacher if policies[actor] == 'B3' else choose(obs, policies[actor], rng)
                        X, mask, p, status, *_ = prepare_survival(obs)
                        y = ACTIONS.index(teacher['action'])
                        assert X.shape == (5, FEATURES) and np.isfinite(X).all() and mask[y]
                        assert len(p) == obs['width']*obs['height'] and all(0 <= v <= 1 for v in p)
                        pending.append((X, mask, y, teacher['target'], teacher['action'] != decision['action']))
                        env.step(decision['action'], decision)
                    replay = env.save_replay(); verify(seed, preset, first, swap, replay)
                    failed = correction and env.winner not in ('draw', learner)
                    eligible = []
                    for ix, (record, details) in enumerate(zip(replay['records'], pending)):
                        counts['raw_transitions'] += 1
                        counts['teacher_disagreements'] += int(details[-1])
                        risk_counts[record['metadata']['risk_status']] += 1
                        if correction and not (failed or details[-1]):
                            continue
                        key = json.dumps({k: v for k, v in record['observation'].items() if k not in ('game_id', 'revision')}, sort_keys=True)
                        if key in seen:
                            counts['duplicate_states'] += 1
                            continue
                        seen.add(key); eligible.append(ix)
                    selected = set(eligible if len(eligible) <= 192 else [eligible[i] for i in np.linspace(0, len(eligible)-1, 192, dtype=int)])
                    eid = f'{name}-{preset}-m{mi:03d}-v{variant}'
                    map_id = f'{preset}-{split}-m{mi:03d}'
                    for ix, (record, details) in enumerate(zip(replay['records'], pending)):
                        X, mask, y, target, disagree = details
                        if ix in selected:
                            arrays['X'].append(X); arrays['mask'].append(mask); arrays['y'].append(y)
                        item = {'schema': 'survival-public-transition-v1', 'rule_version': 'arena-v3.0',
                            'scoring': 'survival-v1', 'episode_id': eid, 'map_id': map_id, 'split': split, 'preset': preset,
                            'transition': record, 'teacher_action': ACTIONS[y], 'teacher_target': target,
                            'teacher_label_source': 'B3 life-weighted public-risk heuristic; not optimal action',
                            'teacher_disagreement': disagree, 'selected_for_features': ix in selected,
                            'selected_for_training': ix in selected and split == 'train',
                            'feature_use': 'validation_only' if split == 'validation' else 'gradient_training',
                            'policy_source_sha256': protocol['source_sha256']['arena/survival.py'],
                            'model_sha256': model_hash(policies[record['actor']]),
                            'terminal_label_separate': replay['result'], 'human_record': False}
                        public_only(item); transition_file.write(json.dumps(item, separators=(',', ':'))+'\n')
                    item = {'schema': 'survival-public-episode-v1', 'episode_id': eid, 'map_id': map_id,
                        'preset': preset, 'split': split, 'first': first, 'swap': swap, 'policies': policies,
                        'learner_seat': learner, 'replay': replay}
                    public_only(item); replay_file.write(json.dumps(item, separators=(',', ':'))+'\n')
                    counts['games'] += 1; counts['learner_losses'] += int(failed)
                    counts['draws'] += int(env.winner == 'draw'); counts['timeouts'] += int(env.reason == 'max_steps')
                    counts['mine_hits'] += sum(3-life for life in env.lives)
                    lengths.append(env.steps)
                    replay_file.flush(); transition_file.flush()
                    print(f'{name}/{preset} map {mi+1} variant {variant}: {env.steps} actions, {len(arrays["y"])} features, {time.perf_counter()-began:.1f}s', flush=True)
    saved = {'X': np.asarray(arrays['X'], np.float32), 'mask': np.asarray(arrays['mask'], bool), 'y': np.asarray(arrays['y'], np.int64)}
    np.savez_compressed(destination/(name+'_features.npz'), **saved)
    result = dict(counts, selected_states=len(arrays['y']), risk_status_counts=dict(risk_counts),
                  game_step_quantiles=quantiles(lengths), elapsed_seconds=time.perf_counter()-began,
                  all_replays_verified=counts['games'], features_sha256=sha(destination/(name+'_features.npz')), human_records=0,
                  balanced_teacher_first=balance_first)
    write(destination/(name+'_collection.json'), result)
    return saved


def arrays(name):
    with np.load(DATA/(name+'_features.npz'), allow_pickle=False) as file:
        return {k: file[k].copy() for k in file.files}


def measure(model, data):
    X, y, mask = data['X'], data['y'], data['mask']
    z = logits(X.reshape(-1, FEATURES), model).reshape(-1, 5)
    z[~mask] = -1e9; z -= z.max(1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(1, keepdims=True)
    return float(-np.log(probs[np.arange(len(y)), y]+1e-12).mean()), float((probs.argmax(1) == y).mean())


def fit(train, valid, filename, epochs, initial=None, *, optimizer_seed=None, model_dir=None):
    optimizer_seed = OPTIMIZER_SEED if optimizer_seed is None else optimizer_seed
    model_dir = MODEL_DIR if model_dir is None else Path(model_dir)
    rng = np.random.default_rng(optimizer_seed)
    model = {k: v.copy() for k, v in initial.items()} if initial else {
        'W1': rng.normal(0, .12, (FEATURES, HIDDEN)).astype(np.float32), 'b1': np.zeros(HIDDEN, np.float32),
        'W2': rng.normal(0, .12, (HIDDEN, 1)).astype(np.float32), 'b2': np.zeros(1, np.float32)}
    m = {k: np.zeros_like(v) for k, v in model.items()}; v = {k: np.zeros_like(x) for k, x in model.items()}
    X, y, mask = train['X'], train['y'], train['mask']
    history, steps, best, began = [], 0, float('inf'), time.perf_counter()
    for epoch in range(epochs):
        for idx in np.array_split(rng.permutation(len(y)), int(np.ceil(len(y)/256))):
            xb = X[idx].reshape(-1, FEATURES); h = np.tanh(xb@model['W1']+model['b1'])
            z = (h@model['W2']+model['b2']).reshape(-1, 5)
            z[~mask[idx]] = -1e9; z -= z.max(1, keepdims=True)
            probs = np.exp(z); probs /= probs.sum(1, keepdims=True); probs[np.arange(len(idx)), y[idx]] -= 1
            dz = (probs/len(idx)).reshape(-1, 1); dh = (dz@model['W2'].T)*(1-h*h)
            grads = {'W2': h.T@dz, 'b2': dz.sum(0), 'W1': xb.T@dh, 'b1': dh.sum(0)}; steps += 1
            for key, gradient in grads.items():
                g = np.clip(gradient, -5, 5); m[key] = .9*m[key]+.1*g; v[key] = .999*v[key]+.001*g*g
                model[key] -= .002*(m[key]/(1-.9**steps))/(np.sqrt(v[key]/(1-.999**steps))+1e-8)
        tl, ta = measure(model, train); vl, va = measure(model, valid)
        row = {'epoch': epoch+1, 'train_loss': tl, 'train_accuracy': ta, 'validation_loss': vl,
               'validation_accuracy': va, 'optimizer_steps': steps}
        history.append(row)
        if vl < best:
            best, selected, best_model = vl, row.copy(), {k: x.copy() for k, x in model.items()}
        if epoch == 0 or epoch % 5 == 4:
            print(f'{filename}: epoch {epoch+1}/{epochs}, train loss {tl:.4f}, validation accuracy {va:.4f}', flush=True)
    path = model_dir/filename; path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **best_model)
    info = {'file': filename, 'sha256': sha(path), 'rule_version': 'arena-v3.0', 'scoring': 'survival-v1',
        'features': FEATURES, 'hidden': HIDDEN, 'parameters': PARAMETERS, 'method': 'supervised imitation of B3',
        'optimizer_seed': optimizer_seed, 'optimizer_steps': steps, 'epochs': epochs,
        'train_examples': len(y), 'validation_examples': len(valid['y']), 'history': history,
        'selected': selected, 'selection': 'lowest validation cross entropy',
        'elapsed_optimization_seconds': time.perf_counter()-began, 'human_records': 0}
    write(model_dir/(path.stem+'_training.json'), info)
    return best_model


def initial():
    freeze()
    return fit(arrays('train'), arrays('validation'), 'survival_v3_initial.npz', 35)


def refine():
    extra = collect('train', True); train = arrays('train')
    combined = {k: np.concatenate([train[k], extra[k]]) for k in train}
    with np.load(MODEL_DIR/'survival_v3_initial.npz', allow_pickle=False) as file:
        model = {k: file[k].copy() for k in file.files}
    fit(combined, arrays('validation'), 'survival_v3_final.npz', 18, model)
    write(DATA/'refinement.json', {'rounds': 1, 'test_used_for_training': False,
        'combined_examples': len(combined['y']), 'correction_examples': len(extra['y']),
        'two_rounds_may_repeat_states': True})


def summaries(rows):
    groups = defaultdict(list); result = []; rng = np.random.default_rng(881)
    for row in rows:
        groups[(row['preset'], row['policy0'], row['policy1'])].append(row)
    for (preset, p0, p1), group in groups.items():
        maps = defaultdict(list)
        for row in group:
            maps[row['map_id']].append(row)
        values = np.array([np.mean([1 if r['winner'] == 1 else .5 if r['winner'] == 'draw' else 0 for r in g]) for g in maps.values()])
        sampled = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
        summary = {'preset': preset, 'opponent': p0, 'policy': p1, 'games': len(group), 'base_maps': len(maps),
            'wins': sum(r['winner'] == 1 for r in group), 'draws': sum(r['winner'] == 'draw' for r in group),
            'losses': sum(r['winner'] == 0 for r in group), 'win_plus_half_draw': float(values.mean()),
            'map_bootstrap_95ci': np.quantile(sampled, [.025, .975]).tolist(),
            'timeouts': sum(r['reason'] == 'max_steps' for r in group),
            'winner_differs_from_diamond_only': sum(r['winner'] != r['diamond_winner'] for r in group),
            'game_step_quantiles': quantiles([r['steps'] for r in group])}
        for metric in ['gems', 'lives', 'mine_hits', 'actions', 'utility', 'normalized_utility']:
            summary['policy_mean_'+metric] = float(np.mean([r[metric+'1'] for r in group]))
            summary['opponent_mean_'+metric] = float(np.mean([r[metric+'0'] for r in group]))
        summary['policy_mean_decision_ms'] = 1000*sum(r['decision_seconds1'] for r in group)/max(1, sum(r['actions1'] for r in group))
        result.append(summary)
    return result


def evaluate():
    truth, protocol = freeze(); output = DATA/'evaluation'; output.mkdir(exist_ok=True)
    rows, risk, began = [], Counter(), time.perf_counter()
    hashes = {p: model_hash(p) for p in ['L3', 'L3_initial', 'L_v2']}
    write(output/'run_manifest.json', {'source': 'automated synthetic policy games', 'rule_version': 'arena-v3.0',
        'scoring': 'survival-v1', 'human_records': 0, 'model_sha256': hashes, 'protocol_sha256': sha(DATA/'protocol.json')})
    with gzip.open(output/'public_replays.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for preset in PRESETS:
            for mi, seed in enumerate(truth['splits'][preset]['test']):
                for pi, (p0, p1) in enumerate(PAIRS):
                    for swap in (False, True):
                        for first in (0, 1):
                            env = Arena(seed, preset=preset, scoring='survival-v1', first=first, swap=swap)
                            times, rng = [0., 0.], random.Random(13717+mi*7+first+2*int(swap))
                            while not env.done:
                                obs = env.observe(); actor = obs['turn']; t = time.perf_counter()
                                decision = choose(obs, (p0, p1)[actor], rng)
                                times[actor] += time.perf_counter()-t; risk[decision['risk_status']] += 1
                                env.step(decision['action'], decision)
                            replay = env.save_replay(); verify(seed, preset, first, swap, replay)
                            eid = f'{preset}-m{mi:03d}-p{pi}-s{int(swap)}-f{first}'; map_id = f'{preset}-test-m{mi:03d}'
                            item = {'schema': 'survival-public-episode-v1', 'episode_id': eid, 'map_id': map_id,
                                'preset': preset, 'split': 'test', 'first': first, 'swap': swap,
                                'policies': [p0, p1], 'model_sha256': hashes, 'replay': replay}
                            public_only(item); stream.write(json.dumps(item, separators=(',', ':'))+'\n')
                            final = env.observe()
                            row = {'episode_id': eid, 'map_id': map_id, 'preset': preset, 'policy0': p0, 'policy1': p1,
                                'swap': int(swap), 'first': first, 'winner': env.winner,
                                'diamond_winner': 'draw' if env.scores[0] == env.scores[1] else int(env.scores[1] > env.scores[0]),
                                'steps': env.steps, 'reason': env.reason}
                            for seat in (0, 1):
                                row.update({f'gems{seat}': env.scores[seat], f'lives{seat}': env.lives[seat],
                                    f'mine_hits{seat}': 3-env.lives[seat], f'actions{seat}': final['action_counts'][seat],
                                    f'utility{seat}': final['utility_scores'][seat],
                                    f'normalized_utility{seat}': final['utility_scores'][seat]/(100*env.resource_count+30),
                                    f'decision_seconds{seat}': times[seat]})
                            rows.append(row)
                stream.flush()
                print(f'evaluation/{preset} map {mi+1}: {len(rows)} games, {time.perf_counter()-began:.1f}s', flush=True)
    with (output/'games.csv').open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {'games': len(rows), 'independent_test_maps': 12, 'all_replays_verified': len(rows),
        'pairs': summaries(rows), 'model_sha256': hashes, 'risk_status_counts': dict(risk),
        'game_step_quantiles': quantiles([r['steps'] for r in rows]),
        'draws': sum(r['winner'] == 'draw' for r in rows), 'timeouts': sum(r['reason'] == 'max_steps' for r in rows),
        'winner_differs_from_diamond_only': sum(r['winner'] != r['diamond_winner'] for r in rows),
        'elapsed_seconds': time.perf_counter()-began, 'human_records': 0}
    effects, rng = [], np.random.default_rng(55271)
    for preset in PRESETS:
        paired = defaultdict(dict)
        for row in rows:
            if row['preset'] == preset and row['policy0'] == 'B2':
                paired[(row['map_id'], row['swap'], row['first'])][row['policy1']] = row
        for left, right in [('L3', 'L3_initial'), ('L3', 'B3')]:
            for metric in ['win_plus_half_draw', 'gems', 'lives', 'mine_hits', 'actions', 'normalized_utility']:
                def value(row):
                    return (1 if row['winner'] == 1 else .5 if row['winner'] == 'draw' else 0) if metric == 'win_plus_half_draw' else row[metric+'1']
                maps = defaultdict(list)
                for key, group in paired.items():
                    maps[key[0]].append(value(group[left])-value(group[right]))
                values = np.array([np.mean(v) for v in maps.values()])
                boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
                effects.append({'preset': preset, 'opponent': 'B2', 'left': left, 'right': right,
                    'metric': metric, 'difference': float(values.mean()), 'base_maps': len(values),
                    'paired_map_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()})
    write(output/'paired_effects.json', effects); write(output/'summary.json', result)
    return result


def audit():
    truth, _ = freeze(); seen, replays = set(), 0
    for preset in PRESETS:
        for split, seeds in truth['splits'][preset].items():
            for seed in seeds:
                fingerprint = canonical(Arena(seed, preset=preset, scoring='survival-v1'))
                assert fingerprint not in seen; seen.add(fingerprint)
    model_results = {}
    for stem in ['survival_v3_initial', 'survival_v3_final']:
        info = read(MODEL_DIR/(stem+'_training.json'))
        assert info['sha256'] == sha(MODEL_DIR/(stem+'.npz'))
        assert info['selected']['validation_loss'] == min(h['validation_loss'] for h in info['history'])
        with np.load(MODEL_DIR/(stem+'.npz'), allow_pickle=False) as data:
            assert data['W1'].shape == (FEATURES, HIDDEN) and sum(data[k].size for k in data.files) == PARAMETERS
        model_results[stem] = info['sha256']
    for stage in ['train', 'validation', 'corrections', 'evaluation']:
        path = DATA/stage/'public_replays.jsonl.gz' if stage == 'evaluation' else DATA/(stage+'_public_replays.jsonl.gz')
        with gzip.open(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line); public_only(item)
                preset = item['preset']; split = 'test' if stage == 'evaluation' else 'train' if stage == 'corrections' else stage
                mi = int(item['map_id'].rsplit('m', 1)[1]); seed = truth['splits'][preset][split][mi]
                verify(seed, preset, item['first'], item['swap'], item['replay']); replays += 1
    reconstructed = {}
    for stage in ['train', 'validation', 'corrections']:
        expected = arrays(stage); index, raw = 0, 0
        with gzip.open(DATA/(stage+'_transitions.jsonl.gz'), 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line); public_only(item); raw += 1
                if not item['selected_for_features']:
                    continue
                X, mask, *_ = prepare_survival(item['transition']['observation'])
                teacher = choose(item['transition']['observation'], 'B3')
                np.testing.assert_array_equal(X, expected['X'][index]); np.testing.assert_array_equal(mask, expected['mask'][index])
                assert ACTIONS.index(teacher['action']) == expected['y'][index] and teacher['action'] == item['teacher_action']
                index += 1
        assert index == len(expected['y'])
        reconstructed[stage] = {'raw_transitions': raw, 'selected_features_rebuilt': index}
    result = {'base_maps': len(seen), 'dihedral_map_overlap': 0, 'all_public_replays_verified_again': replays,
        'public_truth_fields_absent': True, 'model_hashes': model_results,
        'feature_reconstruction': reconstructed, 'human_records': 0}
    write(DATA/'integrity_audit.json', result)
    print(json.dumps(result), flush=True)


def analyze_behavior():
    """Deterministic first-matching public examples; not hand-picked success clips."""
    output = DATA/'evaluation'
    groups, cases, case_counts = defaultdict(Counter), [], Counter()
    deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1), 'wait': (0, 0)}
    with gzip.open(output/'public_replays.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            item = json.loads(line); replay = item['replay']; final = replay['final_observation']
            for seat, policy in enumerate(item['policies']):
                group = groups[(item['preset'], policy)]
                group['games'] += 1; group['eliminations'] += int(final['lives'][seat] == 0)
                group['mine_hits'] += 3-final['lives'][seat]
                group['gems'] += final['scores'][seat]; group['actions'] += final['action_counts'][seat]
            for record in replay['records']:
                obs = record['observation']; actor = record['actor']; policy = item['policies'][actor]
                group = groups[(item['preset'], policy)]; action = record['action']
                group['wait_actions'] += int(action == 'wait')
                pos = obs['positions'][actor]; dr, dc = deltas[action]; next_cell = (pos[0]+dr, pos[1]+dc)
                history = obs['recent_positions'][actor]
                group['immediate_backtracks'] += int(len(history) > 1 and tuple(history[-2]) == next_cell)
                if policy not in ('B3', 'L3'):
                    continue
                kinds = []
                if record['next_observation']['lives'][actor] < obs['lives'][actor]:
                    kinds.append(policy+'_mine_hit')
                if policy == 'L3' and record['next_observation']['lives'][actor] == 0 and obs['lives'][actor] > 0:
                    kinds.append('L3_elimination')
                comparison = None
                candidate = (item['preset'], policy+'_lower_risk_than_B2')
                if case_counts[candidate] < 1:
                    other = choose(obs, 'B2')
                    odr, odc = deltas[other['action']]; other_cell = (pos[0]+odr)*obs['width']+pos[1]+odc
                    cell = next_cell[0]*obs['width']+next_cell[1]
                    if other['action'] != action and record['metadata']['risk'][cell]+1e-9 < other['risk'][other_cell]:
                        kinds.append(policy+'_lower_risk_than_B2')
                        comparison = {'B2_action': other['action'], 'B2_immediate_risk': other['risk'][other_cell],
                                      'selected_action': action, 'selected_immediate_risk': record['metadata']['risk'][cell],
                                      'meaning': 'one public-state action comparison, not an optimality or causal outcome claim'}
                for kind in kinds:
                    key = (item['preset'], kind)
                    if case_counts[key] >= 1:
                        continue
                    filename = f'cases/{item["preset"]}_{kind}.json'
                    value = {'kind': kind, 'episode_id': item['episode_id'], 'step': record['step'],
                        'selection': 'first matching event in fixed evaluation order', 'comparison': comparison,
                        'terminal_result': replay['result'], 'replay': replay}
                    public_only(value)
                    path = output/filename; path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
                    cases.append({'kind': kind, 'preset': item['preset'], 'episode_id': item['episode_id'],
                                  'step': record['step'], 'file': filename})
                    case_counts[key] += 1
    result = {'source': 'all 240 frozen formal replays; descriptive post-hoc analysis only',
        'not_used_for_training_or_checkpoint_selection': True,
        'groups': [dict(preset=key[0], policy=key[1], **values) for key, values in groups.items()],
        'cases': cases, 'case_selection': 'first matching event per preset and category; all games remain in full log',
        'aggregate_caution': 'policy group mixes opponents and is not an independently paired policy comparison'}
    write(output/'behavior_audit.json', result)
    print(f'behavior audit saved {len(cases)} traceable cases', flush=True)


def refit_public():
    """Reproduce both weight files using published arrays, no private seeds needed."""
    if OUT.resolve() == ROOT.resolve():
        raise ValueError('Set ARENA_OUTPUT_DIR to a separate reproduction directory; published weights are protected')
    def source_arrays(name):
        with np.load(ROOT/'data'/'survival_v3'/(name+'_features.npz'), allow_pickle=False) as file:
            return {k: file[k].copy() for k in file.files}
    train, valid, extra = [source_arrays(name) for name in ['train', 'validation', 'corrections']]
    initial_model = fit(train, valid, 'survival_v3_initial.npz', 35)
    combined = {k: np.concatenate([train[k], extra[k]]) for k in train}
    fit(combined, valid, 'survival_v3_final.npz', 18, initial_model)
    result = {'source': 'published public feature arrays; no private map seeds read',
        'model_hashes_match_published': {stem: sha(MODEL_DIR/(stem+'.npz')) == sha(ROOT/'models'/(stem+'.npz'))
             for stem in ['survival_v3_initial', 'survival_v3_final']},
        'platform_note': 'bitwise equality verified here; different numerical libraries may round differently'}
    write(OUT/'reproduction_result.json', result)
    print(json.dumps(result), flush=True)


def report():
    protocol = read(DATA/'protocol.json'); summary = read(DATA/'evaluation/summary.json')
    integrity = read(DATA/'integrity_audit.json')
    tr, va, co = [read(DATA/(name+'_collection.json')) for name in ['train', 'validation', 'corrections']]
    initial, final = [read(MODEL_DIR/(stem+'_training.json')) for stem in ['survival_v3_initial', 'survival_v3_final']]
    text = '# 生存与效率修订 v3：真实训练、留出评测和局限\n\n'
    text += '本轮只对应 arena-v3.0 / survival-v1。终局分为 `100×钻石+10×剩余生命−自己的行动数/全局行动上限`。一钻优先于最多3命及行动差；同钻先比较生命再比较行动数。墙钟时间只报告、不参与奖励。该公式不保证期望效用最优策略绝不冒险，也可能允许领先方等待。旧v1/v2依钻石判胜的实验不可与本表串用。\n\n'
    text += 'B3以公共约束后验构造风险路径，3/2/1命的风险权重为8/18/72，增加24次公开位置的重复、立即折返和等待代价。每条边成本为 `1+αp/max(0.03,1−p)`，已知雷视为极高代价；这是加和启发式，不是联合路径生存概率。对手落后惩罚系数0.25；所有资源参与教师选择，网络只取按当前公开路径/竞争成本排序前三资源。没有按未知真雷修改合法动作；低血是软风险偏好而非绝对安全承诺。\n\n'
    text += 'L3是40→48→1、2017参数的NumPy监督模仿网络，5动作共享打分器，仅使用公开信息。它不是无教师强化学习或涌现策略；路径和历史特征是人工设计。40维依次为：下一格雷概率、已揭示、等待、分差、总步进度、剩余资源比例、对手存活、候选行列；3个资源各6维（对数路线代价、当前位置对数代价、前进量、对手对数代价、代价差、是否抵达）；宽高、雷密度、双方命；双方行动比例、候选位置近24次频率、立即折返、生命风险权重、重复惩罚、精确推断标志、survival-v1标志。\n\n'
    text += f"每档12训练/4验证/6测试，共{integrity['base_maps']}独立基础图。先固定分区、协议和源码哈希，再收集数据。训练{tr['games']}局/{tr['selected_states']}特征；验证{va['games']}局/{va['selected_states']}特征；纠偏{co['games']}局/{co['selected_states']}特征，仅选训练图输局或教师分歧，每局至多均匀取192条，完整轨迹均保留。初训35轮/{initial['optimizer_steps']}更新，选择第{initial['selected']['epoch']}轮；纠偏18轮/{final['optimizer_steps']}更新，按验证交叉熵选择第{final['selected']['epoch']}轮。最终验证模仿准确率{final['selected']['validation_accuracy']:.2%}，不能代替胜率。训练只用一个优化器随机种子。\n\n"
    text += f"最终权重SHA256：`{final['sha256']}`。\n\n"
    text += '|难度|策略/对手|胜/平/负|胜+半平|自己/对手钻石|自己/对手剩余命|自己/对手行动|超时|\n|---|---|---:|---:|---:|---:|---:|---:|\n'
    for p in summary['pairs']:
        text += f"|{p['preset']}|{p['policy']}/{p['opponent']}|{p['wins']}/{p['draws']}/{p['losses']}|{p['win_plus_half_draw']:.2%}|{p['policy_mean_gems']:.2f}/{p['opponent_mean_gems']:.2f}|{p['policy_mean_lives']:.2f}/{p['opponent_mean_lives']:.2f}|{p['policy_mean_actions']:.1f}/{p['opponent_mean_actions']:.1f}|{p['timeouts']}|\n"
    text += f"\n共{summary['games']}正式局、12独立留出图，换座×先手4条件全部保留。{summary['draws']}平局、{summary['timeouts']}超时，{summary['winner_differs_from_diamond_only']}局按新计分得到与只数钻石不同的胜负。全局行动分位数：`{summary['game_step_quantiles']}`；风险推断：`{summary['risk_status_counts']}`。每档仅6图，结论是小规模验证；完整均值、地图分组bootstrap95%区间、死亡/行动/归一效用和计时在evaluation/summary.json，禁止只挑胜率。\n\n"
    lookup = {(p['preset'], p['policy'], p['opponent']): p for p in summary['pairs']}
    text += '本轮实测取舍：'
    for preset in PRESETS:
        b3, l3 = lookup[(preset, 'B3', 'B2')], lookup[(preset, 'L3', 'B2')]
        text += f"{preset}的B3对B2为{b3['wins']}胜{b3['losses']}负，平均命{b3['policy_mean_lives']:.3f}/{b3['opponent_mean_lives']:.3f}；L3对B2为{l3['wins']}胜{l3['losses']}负，平均命{l3['policy_mean_lives']:.3f}/{l3['opponent_mean_lives']:.3f}、钻石{l3['policy_mean_gems']:.3f}/{l3['opponent_mean_gems']:.3f}。"
    text += '不能用保命均值掩盖资源与赢面损失，也不能仅凭个别难度称L3全面更强。保留验证交叉熵选择的最终权重，不能事后按测试成绩换回初始模型。\n\n'
    effects = read(DATA/'evaluation/paired_effects.json')
    text += '补训减初训、均对B2的实际变化：'
    for preset in PRESETS:
        effect = {e['metric']: e['difference'] for e in effects if e['preset'] == preset and e['right'] == 'L3_initial'}
        text += f"{preset}胜率{100*effect['win_plus_half_draw']:+.3f}个百分点、钻石{effect['gems']:+.3f}、生命{effect['lives']:+.3f}、行动{effect['actions']:+.3f}。"
    if all(e['difference'] <= 0 for e in effects if e['right'] == 'L3_initial' and e['metric'] == 'win_plus_half_draw'):
        text += '本轮没有观察到补训提高对B2的留出赢面，验证模仿准确率提升不能写成对战提升。'
    if summary['winner_differs_from_diamond_only'] == 0:
        text += '新次级计分的规则功能已有单元测试，但正式样本没有观察到它改变只数钻石得到的最终赢家；不将其写成实验收益。'
    text += '\n\n'
    text += '|难度|比较（相同B2对手）|指标差（左减右）|均值|地图配对95%区间|\n|---|---|---|---:|---|\n'
    for effect in read(DATA/'evaluation/paired_effects.json'):
        lo, hi = effect['paired_map_bootstrap_95ci']
        text += f"|{effect['preset']}|{effect['left']}−{effect['right']}|{effect['metric']}|{effect['difference']:+.4f}|[{lo:+.4f}, {hi:+.4f}]|\n"
    text += '\n行动数较少也可能来自早死，必须连同钻石、命和死亡阅读。上述均值差及区间是实测，不等于稳定因果提升；补训可能增加/降低赢面或命，不能预设优于B3/B2/L2。归一效用分母为本档`100×初始钻石数+30`，不是训练标签。\n\n'
    text += f"审计再次逐步重放{integrity['all_public_replays_verified_again']}局；{integrity['base_maps']}图及D4等价形式无交叉；重新从公开转移构造的全部选中特征/合法掩码/教师标签逐元素一致：`{integrity['feature_reconstruction']}`。原始jsonl.gz保留失败、超时、实际动作、合法掩码、推断类型及终局标签；终局标签独立字段不输入模型。没有真实玩家记录、工业现场数据或精确最优动作标签。\n\n"
    text += '行为审计见 `evaluation/behavior_audit.json`：淘汰、踩雷、等待、立即折返都从实际公开轨迹统计；同目录cases按固定评测顺序取每档每类首次匹配事件，包含保命分歧和学习策略死亡，完整原局可导入游戏回放。例子只解释具体发生过的行为，不证明最优性或因果。\n\n'
    text += '完整复现：`python -m arena.survival_training all`，也可依次collect/initial/refine/evaluate/audit/analyze/report。新实验使用ARENA_OUTPUT_DIR与仓库外ARENA_PRIVATE_DIR；禁止覆盖旧协议或公开地图清单。仅靠公开特征复训不需要私有种子：先把ARENA_OUTPUT_DIR设为独立目录，再运行 `python -m arena.survival_training refit-public`；它重训初始和最终权重并报告哈希是否一致。\n'
    (ROOT/'docs/survival_v3.md').write_text(text, encoding='utf-8')
    write(DATA/'dataset_card.json', {'schema': 'survival-dataset-card-v1', 'source': 'self-generated arena-v3 simulation; MIT',
        'human_records': 0, 'teacher': 'B3 heuristic actions, not optimal ground truth',
        'allowed_observation': 'public Arena.observe including 24 public recent positions only',
        'protocol': protocol, 'train': tr, 'validation': va, 'corrections': co,
        'final_model_sha256': final['sha256'], 'integrity_audit': integrity,
        'limitations': ['one optimization seed', '12 held-out maps only', 'engineered paths and selected resource slots',
                       'new score not comparable to old diamond-only wins', 'fewer actions can mean early death']})
    print('survival report and dataset card written', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['collect', 'initial', 'refine', 'evaluate', 'audit', 'analyze', 'report', 'refit-public', 'all'])
    args = parser.parse_args()
    if args.stage in ('collect', 'all'):
        collect('train'); collect('validation')
    if args.stage in ('initial', 'all'):
        initial()
    if args.stage in ('refine', 'all'):
        refine()
    if args.stage in ('evaluate', 'all'):
        evaluate()
    if args.stage in ('audit', 'all'):
        audit()
    if args.stage in ('analyze', 'all'):
        analyze_behavior()
    if args.stage in ('report', 'all'):
        report()
    if args.stage == 'refit-public':
        refit_public()
