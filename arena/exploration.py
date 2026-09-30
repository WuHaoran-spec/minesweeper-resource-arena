"""Teacher-free CEM search over eight public-state action-scoring parameters.

This is derivative-free policy search, not PPO, a neural network, or discovery
from raw pixels. Only a trusted runner sees map seeds; policies see observations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import time
import numpy as np

from .env import Arena, ACTIONS

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get('ARENA_OUTPUT_DIR', str(ROOT)))
DATA = OUT / 'data' / 'exploration_v2'
MODELS = OUT / 'models'
PRIVATE = Path(os.environ.get('ARENA_PRIVATE_DIR', str(ROOT.parent / 'private_delivery' / 'experiment_truth')))
FEATURE_NAMES = ['destination_mine_risk', 'destination_revealed', 'wait', 'nearest_resource_manhattan',
                 'nearest_resource_risk_route', 'best_resource_distance_lead', 'arrives_at_resource',
                 'distance_to_live_opponent']
ACTION_NAMES = tuple(ACTIONS)
SEARCH_SEED = 20261002
COUNTS = {'train': 6, 'validation': 2, 'test': 4}
_MODEL_CACHE = {}


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights_id(weights):
    return hashlib.sha256(np.asarray(weights, dtype='<f8').tobytes()).hexdigest()


def save_model(name, weights):
    MODELS.mkdir(parents=True, exist_ok=True)
    path = MODELS / name
    np.savez_compressed(path, weights=np.asarray(weights, dtype=np.float64))
    return sha(path)


def prepare_exploration(obs):
    from .policies import prepare_v2
    _, legal, risk, status, detail, targets, base, _ = prepare_v2(obs)
    width, height = obs.get('width', obs['size']), obs.get('height', obs['size'])
    actor = obs['turn']; pos = obs['positions'][actor]; other = obs['positions'][1-actor]
    live_other = obs['alive'][1-actor]
    diameter = width + height - 2
    revealed = {(r, c) for r, c, n in obs['revealed']}
    features = np.zeros((5, 8), dtype=np.float64)
    for index, (dr, dc) in enumerate(ACTIONS.values()):
        r, c = pos[0] + dr, pos[1] + dc
        if not (0 <= r < height and 0 <= c < width):
            continue
        own_dist = [abs(r-tr) + abs(c-tc) for tr, tc in targets]
        opp_dist = [abs(other[0]-tr) + abs(other[1]-tc) for tr, tc in targets]
        lead = max((o-d for o, d in zip(opp_dist, own_dist)), default=0) if live_other else 0
        features[index] = [risk[r*width+c], float((r, c) in revealed), float(index == 4),
                           min(own_dist, default=0)/diameter,
                           min(float(base[index].min())/diameter, 4.), lead/diameter,
                           float([r, c] in targets),
                           (abs(r-other[0])+abs(c-other[1]))/diameter if live_other else 0.]
    return features, legal, risk, status, detail, targets, base


def decide(obs, weights, rng=None, epsilon=0., parameter_version='frozen', model_sha=None):
    if obs['done'] or not obs['legal_actions']:
        raise ValueError('Cannot act from a terminal observation')
    X, legal, risk, status, detail, targets, base = prepare_exploration(obs)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (8,) or not np.isfinite(weights).all():
        raise ValueError('Exploration policy requires eight finite coefficients')
    scores = X @ weights
    scores[~legal] = -np.inf
    rng = rng or random
    exploratory = epsilon > 0 and rng.random() < epsilon
    index = rng.choice(list(np.flatnonzero(legal))) if exploratory else int(np.argmax(scores))
    target = targets[int(np.argmin(base[index]))] if targets else None
    diag = {'version': 'E-linear8-cem-v1', 'parameter_version': parameter_version,
            'parameter_sha256': weights_id(weights), 'model_sha256': model_sha,
            'target_kind': 'inferred_from_selected_action_not_learned_goal',
            'action_scores': {action: float(scores[i]) for i, action in enumerate(ACTION_NAMES) if legal[i]},
            'coefficients': weights.tolist(), 'epsilon': epsilon, 'sampled_exploration_action': exploratory,
            'risk': detail}
    if obs.get('preset') != 'intermediate' or obs.get('rule_version') != 'arena-v2.0':
        diag['out_of_distribution'] = 'E was searched only on intermediate arena-v2; this preset is outside its training distribution'
    return {'action': ACTION_NAMES[index], 'target': target, 'policy': 'E',
            'risk_status': status, 'risk': risk, 'diagnostics': diag}


def choose_exploration(obs, rng=None):
    path = MODELS / 'exploration_final.npz'
    key = (str(path), path.stat().st_mtime_ns)
    if key not in _MODEL_CACHE:
        with np.load(path, allow_pickle=False) as data:
            weights = data['weights'].copy()
        _MODEL_CACHE[key] = (weights, sha(path))
    weights, digest = _MODEL_CACHE[key]
    return decide(obs, weights, rng=rng, parameter_version='validation-selected-frozen', model_sha=digest)


def public_only(value):
    forbidden = {'seed', 'map_seed', 'mines', 'mine_map', 'hidden_mines', 'private_snapshot', 'future_results'}
    if isinstance(value, dict):
        if forbidden & value.keys():
            raise AssertionError('Private truth in public exploration record')
        for item in value.values(): public_only(item)
    elif isinstance(value, (list, tuple)):
        for item in value: public_only(item)


def manifest():
    if PRIVATE.resolve().is_relative_to(ROOT.resolve()) or PRIVATE.resolve().is_relative_to(OUT.resolve()):
        raise ValueError('Exploration truth must be outside public output')
    path = PRIVATE / 'exploration_manifest.json'
    if path.exists():
        value = json.loads(path.read_text(encoding='utf-8'))
        if {k: len(v) for k, v in value['splits'].items()} != COUNTS:
            raise ValueError('Existing exploration split counts do not match frozen protocol')
        return value
    from .challenge_training import canonical
    used, fingerprints, splits = set(), set(), {}
    # Do not reuse previously generated training or test boards from other runs.
    for other in PRIVATE.glob('*manifest.json'):
        obj = json.loads(other.read_text(encoding='utf-8'))
        def walk(node):
            if isinstance(node, dict):
                for item in node.values(): walk(item)
            elif isinstance(node, list):
                used.update(v for v in node if type(v) is int)
        walk(obj.get('splits', {}))
    for split, count in COUNTS.items():
        values = []
        while len(values) < count:
            candidate = secrets.randbits(63)
            if candidate in used: continue
            fingerprint = canonical(Arena(candidate, preset='intermediate'))
            if fingerprint in fingerprints: continue
            used.add(candidate); fingerprints.add(fingerprint); values.append(candidate)
        splits[split] = values
    value = {'private': True, 'preset': 'intermediate', 'splits': splits}
    write(path, value)
    return value


def freeze():
    truth = manifest()
    protocol = {'protocol_version': 'E-linear8-cem-pilot-v1', 'rule_version': 'arena-v2.0',
        'preset': 'intermediate', 'max_steps': 800, 'independent_maps': COUNTS,
        'manifest_sha256': sha(PRIVATE / 'exploration_manifest.json'),
        'method': 'cross-entropy-method derivative-free search of eight linear action-scoring coefficients',
        'feature_names': FEATURE_NAMES, 'teacher_action_labels_used': False,
        'objective': 'terminal win=1, draw=0.5, loss=0 + 0.25*(own_diamonds-opponent_diamonds)/7',
        'step_rewards': None, 'generations': 4, 'population': 8, 'elite_count': 2,
        'initial_parameter_std': .5, 'initial_search_std': 1., 'minimum_search_std': .1,
        'distribution_update': '0.3 old mean/std + 0.7 elite mean/std',
        'training_seed': SEARCH_SEED, 'action_exploration_epsilon': [.15, .10, .05, .05],
        'training_map_indices_by_generation': [[0, 1], [2, 3], [4, 5], [0, 1]],
        'training_conditions': 'two listed maps x swap false/true x first 0/1; candidate is player 1',
        'opponent_schedule': 'condition index modulo 3: B1, B2, frozen previous completed generation training-best',
        'validation_selection': 'initial plus best training candidate from each completed generation; highest validation objective, earlier on ties',
        'validation_conditions': 'two maps x swap false/true x first 0/1; opponent alternates B1/B2 by condition index',
        'test_conditions': 'four independent maps x B1/B2 x swap false/true x first 0/1 x initial/final frozen policies',
        'planned_training_games': 256, 'planned_validation_games': 40, 'planned_test_games': 64,
        'search_wall_budget_seconds': 720, 'test_wall_budget_seconds': 180,
        'budget_boundary': 'finish and retain the current game; never rank an incomplete candidate or generation',
        'behavior_predicates': {'wait': 'action=wait', 'risk': 'selected destination mine marginal >=0.2',
            'detour': 'selected action increases Manhattan distance to its diagnostic target',
            'switch': 'same actor diagnostic target changes while both old and new targets remain available'},
        'behavior_limit': 'Diagnostic targets are path attributions, not learned goal heads; events do not prove causality or emergence',
        'human_records': 0, 'source': 'automated self-generated simulation'}
    path = DATA / 'protocol.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8')) != protocol:
        raise ValueError('Existing exploration protocol differs; use a new output directory')
    write(path, protocol)
    return truth, protocol


def utility(winner, scores, actor=1):
    return (0.5 if winner == 'draw' else float(winner == actor)) + .25 * (scores[actor]-scores[1-actor])/7.


def events(replay):
    found = []; previous = {}
    for index, record in enumerate(replay['records']):
        if record['actor'] != 1: continue
        obs = record['observation']; decision = record['metadata']; target = decision.get('target')
        dr, dc = ACTIONS[record['action']]; r, c = obs['positions'][1]
        risk = decision['risk'][(r+dr)*obs['width'] + c+dc]
        kinds = []
        if record['action'] == 'wait': kinds.append('wait')
        if risk >= .2: kinds.append('risk')
        if target and abs(r+dr-target[0])+abs(c+dc-target[1]) > abs(r-target[0])+abs(c-target[1]): kinds.append('detour')
        old = previous.get(1)
        if old and target and old != target and old in obs['diamonds'] and target in obs['diamonds']: kinds.append('switch')
        previous[1] = target
        for kind in kinds:
            found.append({'kind': kind, 'record_index': index, 'step': record['step'], 'destination_risk': risk,
                          'feedback': record['feedback'], 'diagnostic_target': target})
    return found


def episode(seed, map_id, stage, identifier, weights, opponent, frozen, swap, first, epsilon, action_rng, stream):
    from .policies import choose
    env = Arena(seed, preset='intermediate', swap=swap, first=first)
    rng = random.Random(action_rng)
    began = time.perf_counter()
    while not env.done:
        obs = env.observe()
        if obs['turn'] == 1:
            decision = decide(obs, weights, rng, epsilon, identifier)
        elif opponent == 'frozen_E':
            decision = decide(obs, frozen, rng, 0., 'frozen-previous-generation')
        else:
            decision = choose(obs, opponent, rng)
        env.step(decision['action'], decision)
    replay = env.save_replay()
    row = {'episode_id': identifier, 'map_id': map_id, 'stage': stage, 'opponent': opponent,
           'swap': swap, 'first': first, 'parameter_sha256': weights_id(weights), 'winner': env.winner,
           'scores': env.scores.copy(), 'lives': env.lives.copy(), 'steps': env.steps, 'reason': env.reason,
           'objective': utility(env.winner, env.scores), 'elapsed_seconds': time.perf_counter()-began}
    item = {**row, 'source': 'teacher_free_parameter_search' if stage == 'search' else 'frozen_policy_evaluation',
            'human_record': False, 'replay': replay}
    public_only(item); stream.write(json.dumps(item, separators=(',', ':'))+'\n'); stream.flush()
    return row, replay


def conditions(truth, split, indices, training=False):
    result = []
    for index in indices:
        for swap in (False, True):
            for first in (0, 1):
                k = len(result)
                opponent = ('B1', 'B2', 'frozen_E')[k % 3] if training else ('B1', 'B2')[k % 2]
                result.append((truth['splits'][split][index], f'E-{split}-m{index:02d}', swap, first, opponent))
    return result


def assess(truth, split, indices, identifier, weights, frozen, epsilon, stream, deadline, training=False):
    rows = []; schedule = conditions(truth, split, indices, training)
    for index, (seed, map_id, swap, first, opponent) in enumerate(schedule):
        if rows and time.perf_counter() >= deadline: break
        row, _ = episode(seed, map_id, 'search' if training else 'validation', f'{identifier}-e{index}',
                         weights, opponent, frozen, swap, first, epsilon, 46001+index, stream)
        rows.append(row)
    return rows, len(rows) == len(schedule)


def train():
    if (DATA/'training_summary.json').exists():
        raise FileExistsError('Completed search exists; use a new ARENA_OUTPUT_DIR to preserve original evidence')
    truth, protocol = freeze(); rng = np.random.default_rng(SEARCH_SEED)
    initial = rng.normal(0., .5, 8); mean = initial.copy(); std = np.ones(8)
    save_model('exploration_initial.npz', initial)
    began = time.perf_counter(); deadline = began + protocol['search_wall_budget_seconds']
    best = initial.copy(); best_value = -np.inf; best_id = 'initial'; frozen = initial.copy()
    records, checkpoints = [], []; completed = 0
    with gzip.open(DATA/'search_public_replays.jsonl.gz', 'wt', encoding='utf-8') as train_stream, \
         gzip.open(DATA/'validation_public_replays.jsonl.gz', 'wt', encoding='utf-8') as val_stream:
        rows, complete = assess(truth, 'validation', [0, 1], 'initial-validation', initial, frozen, 0., val_stream, deadline)
        initial_value = float(np.mean([r['objective'] for r in rows]))
        if complete: best_value = initial_value
        checkpoints.append({'candidate': 'initial', 'parameters': initial.tolist(), 'validation_complete': complete,
                            'validation_mean_objective': initial_value, 'games': rows})
        print(f'E initial validation: {len(rows)} games, objective={initial_value:.4f}', flush=True)
        for generation in range(protocol['generations']):
            population = rng.normal(mean, std, size=(protocol['population'], 8)); generation_records = []
            for candidate, weights in enumerate(population):
                if time.perf_counter() >= deadline: break
                identifier = f'g{generation}-c{candidate}'
                rows, complete = assess(truth, 'train', protocol['training_map_indices_by_generation'][generation],
                    identifier, weights, frozen, protocol['action_exploration_epsilon'][generation], train_stream, deadline, True)
                record = {'generation': generation, 'candidate': candidate, 'parameter_id': identifier,
                          'parameters': weights.tolist(), 'parameter_sha256': weights_id(weights),
                          'complete': complete, 'mean_terminal_objective': float(np.mean([r['objective'] for r in rows])), 'games': rows}
                records.append(record); generation_records.append(record)
                write(DATA/'candidate_results.json', records)
                print(f'E {identifier}: {len(rows)}/8 games, objective={record["mean_terminal_objective"]:.4f}, elapsed={time.perf_counter()-began:.1f}s', flush=True)
                if not complete: break
            if len(generation_records) != protocol['population'] or not all(r['complete'] for r in generation_records): break
            ranked = sorted(generation_records, key=lambda r: (-r['mean_terminal_objective'], r['candidate']))
            elite = np.asarray([r['parameters'] for r in ranked[:protocol['elite_count']]])
            mean = .3*mean + .7*elite.mean(axis=0)
            std = np.maximum(.1, .3*std + .7*elite.std(axis=0))
            frozen = np.asarray(ranked[0]['parameters']); completed += 1
            save_model(f'exploration_g{generation}.npz', frozen)
            rows, complete = assess(truth, 'validation', [0, 1], ranked[0]['parameter_id']+'-validation',
                                   frozen, frozen, 0., val_stream, deadline)
            value = float(np.mean([r['objective'] for r in rows]))
            checkpoint = {'candidate': ranked[0]['parameter_id'], 'parameters': frozen.tolist(),
                          'validation_complete': complete, 'validation_mean_objective': value, 'games': rows}
            checkpoints.append(checkpoint); write(DATA/'validation_checkpoints.json', checkpoints)
            if complete and value > best_value: best, best_value, best_id = frozen.copy(), value, ranked[0]['parameter_id']
            if time.perf_counter() >= deadline: break
    write(DATA/'validation_checkpoints.json', checkpoints)
    digest = save_model('exploration_final.npz', best)
    info = {'method': protocol['method'], 'parameter_count': 8, 'feature_names': FEATURE_NAMES,
            'training_seed': SEARCH_SEED, 'teacher_action_labels_used': False, 'reward': protocol['objective'],
            'completed_generations': completed, 'planned_generations': protocol['generations'],
            'training_games': sum(len(r['games']) for r in records), 'validation_games': sum(len(c['games']) for c in checkpoints),
            'selected_candidate': best_id, 'selected_validation_mean_objective': best_value,
            'selected_parameters': best.tolist(), 'initial_parameters': initial.tolist(),
            'elapsed_seconds': time.perf_counter()-began, 'budget_reached': time.perf_counter() >= deadline,
            'final_sha256': digest, 'initial_sha256': sha(MODELS/'exploration_initial.npz'),
            'protocol_sha256': sha(DATA/'protocol.json'), 'selection_uses_test': False}
    write(DATA/'training_summary.json', info); write(MODELS/'exploration_training.json', info)
    return info


def evaluate():
    if (DATA/'test_summary.json').exists():
        raise FileExistsError('Completed evaluation exists; use a new ARENA_OUTPUT_DIR to preserve original evidence')
    truth, protocol = freeze(); weights = {}
    for label in ('initial', 'final'):
        with np.load(MODELS/f'exploration_{label}.npz', allow_pickle=False) as value: weights[label] = value['weights'].copy()
    frozen = {'models': {label: sha(MODELS/f'exploration_{label}.npz') for label in weights},
              'before_test': True, 'no_test_selection': True}
    write(DATA/'test_frozen_checkpoints.json', frozen)
    began = time.perf_counter(); deadline = began + protocol['test_wall_budget_seconds']; rows = []
    event_counts = {label: Counter() for label in weights}; event_maps = {label: defaultdict(set) for label in weights}; cases = Counter()
    stopped = False
    with gzip.open(DATA/'test_public_replays.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for index, seed in enumerate(truth['splits']['test']):
            for opponent in ('B1', 'B2'):
                for swap in (False, True):
                    for first in (0, 1):
                        for label, params in weights.items():
                            if rows and time.perf_counter() >= deadline: stopped = True; break
                            identifier = f'test-m{index:02d}-{opponent}-s{int(swap)}-f{first}-{label}'
                            row, replay = episode(seed, f'E-test-m{index:02d}', 'test', identifier, params,
                                opponent, weights['initial'], swap, first, 0., 76001+index*8+int(swap)*2+first, stream)
                            row['model'] = label; rows.append(row)
                            for event in events(replay):
                                kind = event['kind']; event_counts[label][kind] += 1; event_maps[label][kind].add(index)
                                if label == 'final' and cases[kind] < 2:
                                    cases[kind] += 1
                                    write(DATA/'cases'/f'{kind}_{cases[kind]}.json', {'episode': row, 'event': event,
                                        'review': 'Automatically selected by frozen public predicate; not evidence of intention, causality or stable emergence',
                                        'human_review': False, 'replay': replay})
                            if len(rows) % 8 == 0: print(f'E held-out: {len(rows)}/64 games, elapsed={time.perf_counter()-began:.1f}s', flush=True)
                        if stopped: break
                    if stopped: break
                if stopped: break
            if stopped: break
    write(DATA/'test_games.json', rows)
    groups = {}
    for label in weights:
        chosen = [r for r in rows if r['model'] == label]
        groups[label] = {'games': len(chosen), 'independent_maps': len({r['map_id'] for r in chosen}),
            'wins': sum(r['winner'] == 1 for r in chosen), 'draws': sum(r['winner'] == 'draw' for r in chosen),
            'losses': sum(r['winner'] == 0 for r in chosen), 'timeouts': sum(r['reason'] == 'max_steps' for r in chosen),
            'mean_terminal_objective': float(np.mean([r['objective'] for r in chosen])) if chosen else None,
            'mean_diamond_difference': float(np.mean([r['scores'][1]-r['scores'][0] for r in chosen])) if chosen else None,
            'step_quantiles': {str(q): float(np.quantile([r['steps'] for r in chosen], q)) for q in (0, .5, .9, 1)} if chosen else {},
            'behavior_events': dict(event_counts[label]), 'behavior_distinct_maps': {k: len(v) for k, v in event_maps[label].items()}}
    paired = defaultdict(dict)
    for row in rows: paired[(row['map_id'], row['opponent'], row['swap'], row['first'])][row['model']] = row['objective']
    differences = [v['final']-v['initial'] for v in paired.values() if set(v) == {'initial', 'final'}]
    summary = {'groups': groups, 'completed_games': len(rows), 'planned_games': 64, 'complete': len(rows) == 64,
        'paired_conditions': len(differences), 'paired_mean_objective_change': float(np.mean(differences)) if differences else None,
        'elapsed_seconds': time.perf_counter()-began, 'budget_reached': stopped, 'frozen_models': frozen['models'],
        'behavior_conclusion': '未发现有充分独立证据支持的稳定新策略；固定判据事件只作为可复核候选。',
        'human_records': 0, 'small_sample_limit': 'Only four independent held-out maps; no reliable superiority or emergence claim'}
    write(DATA/'test_summary.json', summary)
    compact_cases()
    return summary


def compact_cases():
    """Deduplicate showcase payloads while retaining every selected event."""
    directory = DATA/'cases'; index_path = directory/'index.json'
    if index_path.exists() and not any(directory.glob('*.json')):
        return
    selected = []
    for path in sorted(directory.glob('*.json')):
        if path.name == 'index.json': continue
        case = json.loads(path.read_text(encoding='utf-8'))
        if 'replay' not in case: continue
        episode_id = case['episode']['episode_id']
        target = directory/(episode_id+'.json.gz')
        if target.resolve().parent != directory.resolve():
            raise ValueError('Case filename must stay inside its directory')
        if not target.exists():
            with gzip.open(target, 'wt', encoding='utf-8') as stream:
                json.dump(case['replay'], stream, separators=(',', ':'))
        with gzip.open(target, 'rt', encoding='utf-8') as stream:
            assert json.load(stream) == case['replay']
        selected.append({k: v for k, v in case.items() if k != 'replay'} |
                        {'original_case': path.stem, 'full_public_replay': target.name})
    if selected:
        write(index_path, {'cases': selected,
                          'all_games': '../test_public_replays.jsonl.gz',
                          'format': 'One full public replay per unique episode; every selected event retained'})
        # Only remove verified duplicate showcase payloads; complete experiment logs stay intact.
        for case in selected:
            duplicate = directory/(case['original_case']+'.json')
            assert duplicate.resolve().parent == directory.resolve()
            duplicate.unlink()


def analyze_case():
    """Post hoc single-state explanation; never trains or alters a checkpoint."""
    index = json.loads((DATA/'cases'/'index.json').read_text(encoding='utf-8'))
    selected = next(case for case in index['cases'] if case['original_case'] == 'detour_1')
    path = DATA/'cases'/selected['full_public_replay']
    with gzip.open(path, 'rt', encoding='utf-8') as stream: replay = json.load(stream)
    record = replay['records'][selected['event']['record_index']]
    observation = record['observation']
    with np.load(MODELS/'exploration_final.npz', allow_pickle=False) as model: weights = model['weights'].copy()
    features, legal, risk, status, detail, targets, base = prepare_exploration(observation)
    actual = features @ weights
    ablated = weights.copy(); ablated[1] = 0.
    counterfactual = features @ ablated
    actual[~legal] = -np.inf; counterfactual[~legal] = -np.inf
    assert ACTION_NAMES[int(np.argmax(actual))] == record['action']
    for action, value in record['metadata']['diagnostics']['action_scores'].items():
        assert np.isclose(actual[ACTION_NAMES.index(action)], value, rtol=1e-12, atol=1e-12)
    states = Counter(); own_positions = []; action_counts = Counter()
    for item in replay['records']:
        if item['actor'] != 1: continue
        obs = item['observation']; own_positions.append(tuple(obs['positions'][1])); action_counts[item['action']] += 1
        state = {k: v for k, v in obs.items() if k not in ('game_id', 'revision', 'steps')}
        states[json.dumps(state, sort_keys=True)] += 1
    result = {'analysis_type': 'post_hoc_public_single_state_ablation',
        'source_full_replay': 'cases/'+selected['full_public_replay'], 'replay_sha256': sha(path),
        'model_sha256': sha(MODELS/'exploration_final.npz'), 'episode_id': selected['episode']['episode_id'],
        'record_index': selected['event']['record_index'], 'step': record['step'], 'observation': observation,
        'feature_names': FEATURE_NAMES, 'weights': weights.tolist(), 'risk_status': status,
        'actions': {action: {'features': features[i].tolist(), 'weighted_contributions': (features[i]*weights).tolist(),
                            'actual_score': float(actual[i]), 'score_with_revealed_coefficient_zero': float(counterfactual[i])}
                    for i, action in enumerate(ACTION_NAMES) if legal[i]},
        'recorded_action': record['action'], 'recomputed_action': ACTION_NAMES[int(np.argmax(actual))],
        'single_coefficient_counterfactual_action': ACTION_NAMES[int(np.argmax(counterfactual))],
        'whole_episode_observations': {'final_scores': replay['result']['scores'], 'reason': replay['result']['reason'],
            'steps': replay['final_observation']['steps'], 'candidate_actions': dict(action_counts),
            'candidate_unique_pre_action_positions': len(set(own_positions)),
            'repeated_identical_public_states_excluding_clock': sum(n-1 for n in states.values()),
            'two_action_return_patterns': sum(own_positions[i] == own_positions[i-2] and own_positions[i] != own_positions[i-1]
                                             for i in range(2, len(own_positions)))},
        'limits': 'Single-state score intervention only. Does not execute the altered policy, estimate outcome change, prove optimality, or change final weights. Chosen after held-out evaluation for failure analysis.',
        'selection_for_training': False, 'checkpoint_modified': False, 'human_records': 0}
    write(DATA/'failure_analysis.json', result)
    print(json.dumps({k: result[k] for k in ('episode_id', 'step', 'recorded_action',
        'single_coefficient_counterfactual_action', 'whole_episode_observations')}, ensure_ascii=False), flush=True)
    return result


def report():
    training = json.loads((DATA/'training_summary.json').read_text(encoding='utf-8'))
    evaluation = json.loads((DATA/'test_summary.json').read_text(encoding='utf-8'))
    text = '# 无教师动作标签的自由参数探索 E\n\n'
    text += 'E 是在预设公开特征空间中真实执行的交叉熵方法（CEM）参数搜索。它是 8 参数线性动作评分器，不是神经网络、PPO、梯度强化学习或通用大模型，也没有从零发现算法。风险计数和路径特征仍由人工编写；没有使用 B2 动作作为训练标签。\n\n'
    text += '输入为目的格风险、已揭状态、等待标志、最近资源曼哈顿距离、最近资源风险路径代价、资源距离优势、是否到达资源、与存活对手距离。每个候选动作分数为这些公开特征与 8 个待搜索系数的点积。训练时参数和动作均有随机探索；演示与冻结评测取合法动作中最高分。\n\n'
    text += '唯一优化目标是终局胜记 1、平记 0.5、负记 0，再加 0.25×（己方钻石−对手钻石）/7。没有中间步奖励、目标切换奖励或教师行动标签。只在 16×16 中级规则上搜索；在专家或旧版棋盘运行会显示分布外诊断。\n\n'
    text += f'冻结计划为 6 训练图、2 验证图、4 独立测试图。实际完成 {training["completed_generations"]}/4 代、{training["training_games"]} 个搜索对局和 {training["validation_games"]} 个验证对局，耗时 {training["elapsed_seconds"]:.2f} 秒。最终选择 {training["selected_candidate"]}，只依据验证终局目标，未使用测试集挑参数。搜索预算是否耗尽：{training["budget_reached"]}。候选参数、全部胜负/超时和公开重放均保留。\n\n'
    text += '|冻结策略|局数|独立测试图|胜/平/负|超时|终局目标均值|步数中位数/90分位|\n|---|---:|---:|---|---:|---:|---|\n'
    for label, item in evaluation['groups'].items():
        q = item['step_quantiles']; objective = item['mean_terminal_objective']
        objective_display = f'{objective:.4f}' if objective is not None else '未运行'
        median = f'{q["0.5"]:.1f}' if '0.5' in q else '—'
        tail = f'{q["0.9"]:.1f}' if '0.9' in q else '—'
        text += f'|{label}|{item["games"]}|{item["independent_maps"]}|{item["wins"]}/{item["draws"]}/{item["losses"]}|{item["timeouts"]}|{objective_display}|{median}/{tail}|\n'
    change = evaluation['paired_mean_objective_change']
    text += f'\n独立测试完成 {evaluation["completed_games"]}/64 局；初始与最终参数在相同条件上的完整配对数为 {evaluation["paired_conditions"]}，平均目标差为 {change:.4f}。仅有 4 张基础图，换座和先手重复不是独立地图，不能据此得出可靠优越性结论。\n\n'
    text += '等待、绕行、目标归因切换与风险动作的识别判据在搜索前写入协议。案例索引 `data/exploration_v2/cases/index.json` 保留每个选中事件的步号，并引用该局唯一的完整公开重放压缩文件；全部测试局仍在原始日志中。路径归因目标不是模型的独立目标头。获胜局里的等待也不能证明等待导致胜利。'+evaluation['behavior_conclusion']+'\n\n'
    analysis_path = DATA/'failure_analysis.json'
    if analysis_path.exists():
        analysis = json.loads(analysis_path.read_text(encoding='utf-8'))
        down, left = analysis['actions']['down'], analysis['actions']['left']
        pattern = analysis['whole_episode_observations']
        text += f'失败分析采用测试完成后的单状态消融，未再次训练或选择参数。`{analysis["episode_id"]}` 第 {analysis["step"]} 步，向下得分 {down["actual_score"]:.6f}，向左 {left["actual_score"]:.6f}，两格风险均为 0。向下的“已揭示”特征贡献 +{down["weighted_contributions"][1]:.6f}；把这一项系数临时置零，当前状态的最高分动作变为向左。最终权重没有改动，也没有执行这个替代策略，不能推断胜率改善。\n\n'
        text += f'该失败局最终 0:0、800 步超时，候选策略只出现在 {pattern["candidate_unique_pre_action_positions"]} 个不同的行动前位置，记录到 {pattern["two_action_return_patterns"]} 次两次行动返回原格的模式，排除时间字段后重复公开状态 {pattern["repeated_identical_public_states_excluding_clock"]} 次。当前 8 特征没有访问历史，不能表达“刚从这里来”的循环惩罚；已揭示偏好在这个状态中压过了较低的资源路线代价。这里是可复算的失败原因线索，不是成功策略或普遍因果结论。完整分数分解在 `data/exploration_v2/failure_analysis.json`。\n\n'
    text += f'最终权重 SHA-256：`{training["final_sha256"]}`。原始证据位于 `data/exploration_v2/`，私有地图清单位于公开仓库之外。全部为自动化合成对局，真人记录为 0。重新运行应使用新的 `ARENA_OUTPUT_DIR` 与私人目录：`python -m arena.exploration all`。\n'
    document = OUT/'docs'/'exploration.md'
    document.parent.mkdir(parents=True, exist_ok=True)
    document.write_text(text, encoding='utf-8')


def audit():
    """Trusted replay audit. No map truth is copied into the public report."""
    from .challenge_training import canonical
    truth = json.loads((PRIVATE/'exploration_manifest.json').read_text(encoding='utf-8'))
    protocol = json.loads((DATA/'protocol.json').read_text(encoding='utf-8'))
    assert sha(PRIVATE/'exploration_manifest.json') == protocol['manifest_sha256']
    fingerprints = {}
    for split, values in truth['splits'].items():
        for index, seed in enumerate(values):
            fingerprint = canonical(Arena(seed, preset='intermediate'))
            assert fingerprint not in fingerprints, 'Mirrored or rotated map crosses exploration splits'
            fingerprints[fingerprint] = (split, index)
    # Independent check against all earlier experiment maps, including unused prefixes.
    earlier_boards = 0
    for name in ('map_manifest.json', 'challenge_manifest.json'):
        path = PRIVATE/name
        if not path.exists(): continue
        other = json.loads(path.read_text(encoding='utf-8'))
        splits = other.get('splits', {})
        batches = splits.items() if name == 'challenge_manifest.json' else [('legacy', splits)]
        for preset, by_split in batches:
            for values in by_split.values():
                for seed in values:
                    assert canonical(Arena(seed, preset=preset)) not in fingerprints, 'Map reused from earlier experiment'
                    earlier_boards += 1
    totals, outcomes, risk_statuses = Counter(), Counter(), Counter()
    sampled = 0
    for filename in ('search_public_replays.jsonl.gz', 'validation_public_replays.jsonl.gz', 'test_public_replays.jsonl.gz'):
        with gzip.open(DATA/filename, 'rt', encoding='utf-8') as source:
            for line in source:
                item = json.loads(line); public_only(item)
                split, number = item['map_id'].removeprefix('E-').rsplit('-m', 1)
                seed = truth['splits'][split][int(number)]
                env = Arena(seed, preset='intermediate', first=item['first'], swap=item['swap'])
                replay = item['replay']; env.game_id = replay['initial_observation']['game_id']
                assert env.observe() == replay['initial_observation']
                for record in replay['records']:
                    assert env.observe() == record['observation']
                    decision = record['metadata']; assert decision['action'] == record['action']
                    if record['actor'] == 1:
                        assert decision['policy'] == 'E'
                        assert weights_id(decision['diagnostics']['coefficients']) == item['parameter_sha256']
                        values = decision['diagnostics']['action_scores']
                        assert set(values) == set(env.legal_actions())
                        if not decision['diagnostics']['sampled_exploration_action']:
                            assert values[decision['action']] == max(values.values())
                        else: sampled += 1
                        risk_statuses[decision['risk_status']] += 1
                    assert env.step(record['action'], decision) == record['next_observation']
                    assert env._history[-1]['feedback'] == record['feedback']
                    totals['steps'] += 1
                assert env.done and env.observe() == replay['final_observation']
                assert env.scores == item['scores'] and env.lives == item['lives']
                assert env.winner == item['winner'] and env.reason == item['reason']
                assert utility(env.winner, env.scores) == item['objective']
                totals[item['stage']] += 1; outcomes[str(item['winner'])] += 1
                totals['timeouts'] += int(env.reason == 'max_steps')
    checkpoints = json.loads((DATA/'validation_checkpoints.json').read_text(encoding='utf-8'))
    eligible = [v for v in checkpoints if v['validation_complete']]
    selected = max(eligible, key=lambda v: v['validation_mean_objective'])
    with np.load(MODELS/'exploration_final.npz', allow_pickle=False) as model:
        assert np.array_equal(model['weights'], selected['parameters'])
    summary = json.loads((DATA/'training_summary.json').read_text(encoding='utf-8'))
    assert summary['selected_candidate'] == selected['candidate']
    assert summary['final_sha256'] == sha(MODELS/'exploration_final.npz')
    frozen = json.loads((DATA/'test_frozen_checkpoints.json').read_text(encoding='utf-8'))
    for label, digest in frozen['models'].items():
        assert sha(MODELS/f'exploration_{label}.npz') == digest
    result = {'passed': True, 'reexecuted_games': sum(totals[k] for k in ('search', 'validation', 'test')),
        'reexecuted_steps': totals['steps'], 'stage_games': {k: totals[k] for k in ('search', 'validation', 'test')},
        'outcomes_player_1': dict(outcomes), 'retained_timeouts': totals['timeouts'],
        'sampled_exploration_actions': sampled, 'candidate_risk_statuses': dict(risk_statuses),
        'independent_new_maps': len(fingerprints), 'earlier_boards_checked_for_geometric_duplicates': earlier_boards,
        'selection_recomputed_from_validation_only': True, 'frozen_test_hashes_match': True,
        'public_records_private_truth_check': 'passed',
        'scope': 'Full state/action/feedback/outcome replay, public metadata and selection checks; no strategic optimality claim'}
    write(DATA/'audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['all', 'train', 'evaluate', 'report', 'audit', 'compact-cases', 'analyze-case'])
    args = parser.parse_args()
    if args.stage in ('all', 'train'): train()
    if args.stage in ('all', 'evaluate'): evaluate()
    if args.stage in ('all', 'report'): report()
    if args.stage in ('all', 'audit'): audit()
    if args.stage == 'compact-cases': compact_cases()
    if args.stage == 'analyze-case': analyze_case()


if __name__ == '__main__':
    main()
