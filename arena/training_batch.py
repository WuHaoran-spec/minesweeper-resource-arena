"""Versioned, multi-initialization L3 experiments with per-run evidence.

Run with explicit --output and --private directories. Existing models are read
only. The trusted runner owns hidden map seeds; decisions receive public states.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import random
import secrets
import time

import numpy as np
from . import survival_training as training
from .challenge_training import canonical, public_only, read, sha, write
from .env import Arena
from .policies import ACTIONS, MODEL_DIR, choose, logits
from .survival import FEATURES, HIDDEN, PARAMETERS, prepare_survival

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ('intermediate', 'expert')
COUNTS = {'train': 36, 'validation': 12, 'test': 20}
RUN_SEEDS = (11001, 22002, 33003)
EPOCHS = 45
OPPONENTS = ('B2', 'B3')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_paths(output, private):
    output, private = Path(output).resolve(), Path(private).resolve()
    if output == ROOT or ROOT.is_relative_to(output):
        raise ValueError('Use a dedicated new experiment output directory')
    if private.is_relative_to(ROOT) or private.is_relative_to(output) or output.is_relative_to(private):
        raise ValueError('Private truth must be outside repository and public output')
    return output, private


def fingerprint_values(value):
    if isinstance(value, dict):
        return set().union(*(fingerprint_values(v) for v in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(fingerprint_values(v) for v in value)) if value else set()
    return {value} if isinstance(value, str) and len(value) == 64 else set()


def create_protocol(output, private, exclusions):
    output, private = validate_paths(output, private)
    if (output/'protocol.json').exists() or (private/'manifest.json').exists():
        raise FileExistsError('Batch already exists; use stage commands to resume, not all')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('New batch output directory must be empty')
    forbidden, excluded_files = set(), []
    for path in exclusions:
        obj = read(path)
        fingerprints = fingerprint_values(obj.get('canonical_fingerprints', {}))
        if not fingerprints and obj.get('preset') in PRESETS:
            fingerprints = {canonical(Arena(seed, preset=obj['preset'], scoring='survival-v1'))
                            for seeds in obj['splits'].values() for seed in seeds}
        if not fingerprints:
            raise ValueError('Exclusion manifest lacks canonical fingerprints')
        forbidden.update(fingerprints)
        excluded_files.append({'sha256': sha(path), 'fingerprints': len(fingerprints)})
    seen, used, splits, prints = set(forbidden), set(), {}, {}
    for preset in PRESETS:
        splits[preset], prints[preset] = {}, {}
        for split, count in COUNTS.items():
            values, fps = [], []
            while len(values) < count:
                seed = secrets.randbits(63)
                fingerprint = canonical(Arena(seed, preset=preset, scoring='survival-v1'))
                if seed in used or fingerprint in seen:
                    continue
                used.add(seed); seen.add(fingerprint)
                values.append(seed); fps.append(fingerprint)
            splits[preset][split], prints[preset][split] = values, fps
    truth = {'private': True, 'splits': splits, 'canonical_fingerprints': prints,
             'excluded_fingerprints': sorted(forbidden)}
    write(private/'manifest.json', truth)
    sources = ['arena/training_batch.py', 'arena/survival_training.py', 'arena/env.py',
               'arena/risk.py', 'arena/policies.py', 'arena/survival.py']
    protocol = {'schema': 'l3-expanded-batch-v1', 'created_utc': utc_now(),
        'rule_version': 'arena-v3.0', 'scoring': 'survival-v1',
        'base_maps_per_preset': COUNTS, 'presets': list(PRESETS),
        'split_unit': 'base map including all rotations, mirrors, seat and first-turn variants',
        'private_manifest_sha256': sha(private/'manifest.json'),
        'excluded_prior_manifest_hashes': excluded_files,
        'prior_fingerprints_excluded': len(forbidden),
        'run_seeds': list(RUN_SEEDS), 'runs': 3, 'epochs_per_run': EPOCHS,
        'initialization': 'independent random initialization; same expanded data for all runs',
        'method': 'supervised B3 imitation; no reinforcement learning or extra correction stage',
        'features': FEATURES, 'hidden': HIDDEN, 'parameters': PARAMETERS,
        'optimizer': 'Adam', 'learning_rate': .002, 'batch_size': 256,
        'collection_games_per_map': 2, 'state_cap_per_episode': 192,
        'collection_balance': 'learner seat varies per map; teacher first/second and corner assignment balanced across maps',
        'checkpoint_selection': 'lowest validation cross entropy within 45 epochs',
        'candidate_nomination': 'lowest validation cross entropy across three runs, frozen before test',
        'test_opponents': list(OPPONENTS), 'test_variants': 'swap false/true x first 0/1',
        'games_per_candidate': 320, 'baseline_games_shared_once': 320,
        'total_test_games': 1280, 'independent_test_maps': 40,
        'baseline': 'existing L3 survival_v3_final.npz',
        'baseline_sha256': sha(MODEL_DIR/'survival_v3_final.npz'),
        'collection_opponent_L2_sha256': sha(MODEL_DIR/'challenge_final.npz'),
        'promote_to_game': False, 'all_losses_draws_timeouts_retained': True,
        'human_records': 0, 'source_sha256': {name: sha(ROOT/name) for name in sources}}
    write(output/'protocol.json', protocol)
    return truth, protocol


def load_batch(output, private):
    output, private = validate_paths(output, private)
    protocol, truth = read(output/'protocol.json'), read(private/'manifest.json')
    if sha(private/'manifest.json') != protocol['private_manifest_sha256']:
        raise ValueError('Private manifest changed')
    for name, digest in protocol['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('Frozen implementation changed: '+name)
    if sha(MODEL_DIR/'survival_v3_final.npz') != protocol['baseline_sha256']:
        raise ValueError('Baseline changed')
    if sha(MODEL_DIR/'challenge_final.npz') != protocol['collection_opponent_L2_sha256']:
        raise ValueError('Collection opponent changed')
    return output, private, truth, protocol


def collect(output, private):
    output, _, truth, protocol = load_batch(output, private)
    for split in ('train', 'validation'):
        if (output/'data'/(split+'_collection.json')).exists():
            info = read(output/'data'/(split+'_collection.json'))
            assert info['features_sha256'] == sha(output/'data'/(split+'_features.npz'))
            assert info['games'] == 2*sum(len(truth['splits'][p][split]) for p in PRESETS)
            assert info['balanced_teacher_first'] is True
            print('REUSE completed '+split+' collection', flush=True)
            continue
        archive_partial(output, list((output/'data').glob(split+'_*')), split)
        training.collect(split, frozen=(truth, protocol), data_dir=output/'data', balance_first=True)


def load_arrays(output, split):
    with np.load(output/'data'/(split+'_features.npz'), allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def archive_partial(output, paths, stage):
    existing = [p for p in paths if p.is_file()]
    if not existing:
        return
    folder = output/'incomplete_attempts'/(stage+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    folder.mkdir(parents=True)
    for path in existing:
        if not path.resolve().is_relative_to(output.resolve()):
            raise ValueError('Cannot archive outside this batch')
        path.replace(folder/path.name)
    write(folder/'status.json', {'status': 'incomplete attempt retained', 'stage': stage,
          'counted_as_completed_training_or_test': False, 'files': [p.name for p in existing]})


def train_all(output, private):
    output, _, _, protocol = load_batch(output, private)
    train, valid = load_arrays(output, 'train'), load_arrays(output, 'validation')
    results = []
    for index, seed in enumerate(protocol['run_seeds'], 1):
        name = f'L3_run{index:02d}'
        path = output/'models'/(name+'_training.json')
        if path.exists():
            info = read(path)
            assert info['optimizer_seed'] == seed and info['epochs'] == protocol['epochs_per_run']
            assert info['sha256'] == sha(output/'models'/(name+'.npz'))
            assert info['train_features_sha256'] == sha(output/'data/train_features.npz')
            assert info['validation_features_sha256'] == sha(output/'data/validation_features.npz')
            results.append(info)
            print('REUSE completed '+name, flush=True)
            continue
        archive_partial(output, [output/'models'/(name+'.npz')], name)
        started = utc_now()
        training.fit(train, valid, name+'.npz', protocol['epochs_per_run'],
                     optimizer_seed=seed, model_dir=output/'models')
        info = read(path)
        info.update(run_id=name, started_utc=started, completed_utc=utc_now(),
                    independent_initialization=True, test_used_for_training=False,
                    train_features_sha256=sha(output/'data/train_features.npz'),
                    validation_features_sha256=sha(output/'data/validation_features.npz'))
        write(path, info); results.append(info)
        print('TRAINING_COMPLETE '+name+' '+info['sha256'], flush=True)
    nominated = min(results, key=lambda r: r['selected']['validation_loss'])
    selection_path = output/'selection_before_test.json'
    if selection_path.exists():
        assert read(selection_path)['sha256'] == nominated['sha256']
        return
    write(selection_path, {'created_utc': utc_now(),
          'policy': nominated['run_id'], 'sha256': nominated['sha256'],
          'selection': protocol['candidate_nomination'], 'test_read': False,
          'game_default_changed': False})


def candidate_decision(obs, model, name, digest):
    if obs['done'] or not obs['legal_actions']:
        raise ValueError('Cannot choose action in terminal state')
    X, mask, p, status, detail, targets, _, costs = prepare_survival(obs)
    values = logits(X, model); values[~mask] = -1e9
    ai = int(np.argmax(values))
    target = targets[int(np.argmin(costs[ai]))] if targets else None
    detail.update(model=name, model_sha256=digest,
                  target_kind='inferred_from_selected_action_not_network_head',
                  action_logits={a: round(float(values[i]), 6) for i, a in enumerate(ACTIONS) if mask[i]})
    return {'action': ACTIONS[ai], 'target': target, 'policy': name,
            'risk_status': status, 'risk': p, 'diagnostics': detail}


def paired_effects(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row['preset'], row['policy0'], row['map_id'], row['swap'], row['first'])
        grouped[key][row['policy1']] = row
    result, rng = [], np.random.default_rng(51001)
    for preset in PRESETS:
        for opponent in OPPONENTS:
            for candidate in ('L3_run01', 'L3_run02', 'L3_run03'):
                for metric in ('win_plus_half_draw', 'gems', 'lives', 'actions', 'normalized_utility'):
                    maps = defaultdict(list)
                    def val(row):
                        if metric == 'win_plus_half_draw':
                            return 1 if row['winner'] == 1 else .5 if row['winner'] == 'draw' else 0
                        return row[metric+'1']
                    for key, group in grouped.items():
                        if key[:2] == (preset, opponent):
                            maps[key[2]].append(val(group[candidate])-val(group['L3_previous']))
                    values = np.array([np.mean(v) for v in maps.values()])
                    boot = values[rng.integers(0, len(values), (2000, len(values)))].mean(1)
                    result.append({'preset': preset, 'opponent': opponent, 'policy': candidate,
                        'reference': 'L3_previous', 'metric': metric, 'base_maps': len(values),
                        'difference': float(values.mean()),
                        'paired_map_bootstrap_95ci': np.quantile(boot, [.025, .975]).tolist()})
    return result


def evaluate(output, private):
    output, _, truth, protocol = load_batch(output, private)
    if not (output/'selection_before_test.json').exists():
        raise ValueError('Freeze validation nomination before test')
    models, hashes = {}, {'L3_previous': protocol['baseline_sha256']}
    training_records = []
    for index in range(1, 4):
        name = f'L3_run{index:02d}'; path = output/'models'/(name+'.npz')
        hashes[name] = sha(path)
        info = read(output/'models'/(name+'_training.json'))
        assert info['sha256'] == hashes[name]
        assert info['selected']['validation_loss'] == min(row['validation_loss'] for row in info['history'])
        training_records.append(info)
        with np.load(path, allow_pickle=False) as data:
            models[name] = {key: data[key].copy() for key in data.files}
    nominated = min(training_records, key=lambda info: info['selected']['validation_loss'])
    nomination = read(output/'selection_before_test.json')
    assert nomination['policy'] == nominated['run_id'] and nomination['sha256'] == nominated['sha256']
    directory = output/'evaluation'; directory.mkdir(exist_ok=True)
    if (directory/'summary.json').exists():
        raise FileExistsError('Evaluation already complete')
    archive_partial(output, list(directory.glob('*')), 'evaluation')
    write(directory/'run_manifest.json', {'started_utc': utc_now(), 'model_hashes': hashes,
        'protocol_sha256': sha(output/'protocol.json'),
        'selection_sha256': sha(output/'selection_before_test.json'), 'human_records': 0})
    rows, risk_counts, began = [], Counter(), time.perf_counter()
    with gzip.open(directory/'public_replays.jsonl.gz', 'wt', encoding='utf-8') as stream:
        for preset in PRESETS:
            for mi, seed in enumerate(truth['splits'][preset]['test']):
                for opponent in OPPONENTS:
                    for name in ('L3_previous', 'L3_run01', 'L3_run02', 'L3_run03'):
                        for swap in (False, True):
                            for first in (0, 1):
                                env = Arena(seed, preset=preset, scoring='survival-v1', first=first, swap=swap)
                                times, rng = [0., 0.], random.Random(51017+mi*7+first+2*int(swap))
                                while not env.done:
                                    obs = env.observe(); actor = obs['turn']; started = time.perf_counter()
                                    decision = (choose(obs, opponent, rng) if actor == 0 else
                                                choose(obs, 'L3', rng) if name == 'L3_previous' else
                                                candidate_decision(obs, models[name], name, hashes[name]))
                                    times[actor] += time.perf_counter()-started
                                    risk_counts[decision['risk_status']] += 1
                                    env.step(decision['action'], decision)
                                replay = env.save_replay(); training.verify(seed, preset, first, swap, replay)
                                map_id = f'{preset}-test-m{mi:03d}'
                                eid = f'{map_id}-{opponent}-{name}-s{int(swap)}-f{first}'
                                item = {'episode_id': eid, 'map_id': map_id, 'preset': preset, 'split': 'test',
                                        'first': first, 'swap': swap, 'policies': [opponent, name], 'replay': replay}
                                public_only(item); stream.write(json.dumps(item, separators=(',', ':'))+'\n')
                                final = env.observe()
                                row = {'episode_id': eid, 'map_id': map_id, 'preset': preset, 'policy0': opponent,
                                       'policy1': name, 'swap': int(swap), 'first': first, 'winner': env.winner,
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
                print(f'TEST {preset} {mi+1}/{COUNTS["test"]}: {len(rows)} games {time.perf_counter()-began:.1f}s', flush=True)
    with (directory/'games.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    effects = paired_effects(rows)
    result = {'games': len(rows), 'independent_test_maps': protocol['independent_test_maps'],
        'all_replays_verified': len(rows), 'pairs': training.summaries(rows),
        'draws': sum(r['winner'] == 'draw' for r in rows),
        'timeouts': sum(r['reason'] == 'max_steps' for r in rows),
        'risk_status_counts': dict(risk_counts), 'elapsed_seconds': time.perf_counter()-began,
        'completed_utc': utc_now(), 'human_records': 0}
    write(directory/'paired_effects.json', effects); write(directory/'summary.json', result)
    for index in range(1, 4):
        name = f'L3_run{index:02d}'
        record = {'run_id': name, 'protocol': protocol,
                  'training': read(output/'models'/(name+'_training.json')),
                  'train_collection': read(output/'data/train_collection.json'),
                  'validation_collection': read(output/'data/validation_collection.json'),
                  'evaluation': [r for r in result['pairs'] if r['policy'] == name],
                  'reference_evaluation': [r for r in result['pairs'] if r['policy'] == 'L3_previous'],
                  'paired_effects': [r for r in effects if r['policy'] == name],
                  'validation_nomination': read(output/'selection_before_test.json'),
                  'game_default_changed': False}
        write(output/'reports'/(name+'_result.json'), record)
    return result


def audit(output, private):
    output, _, truth, protocol = load_batch(output, private)
    forbidden = set(truth['excluded_fingerprints']); seen = set(); split_counts = Counter()
    for preset in PRESETS:
        for split, seeds in truth['splits'][preset].items():
            for seed in seeds:
                fp = canonical(Arena(seed, preset=preset, scoring='survival-v1'))
                assert fp not in seen and fp not in forbidden
                seen.add(fp); split_counts[split] += 1
    reconstructed, count = {}, 0
    for split in ('train', 'validation'):
        expected = load_arrays(output, split); index = 0
        with gzip.open(output/'data'/(split+'_transitions.jsonl.gz'), 'rt', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line); public_only(item)
                if not item['selected_for_features']:
                    continue
                obs = item['transition']['observation']; X, mask, *_ = prepare_survival(obs)
                teacher = choose(obs, 'B3')
                np.testing.assert_array_equal(X, expected['X'][index])
                np.testing.assert_array_equal(mask, expected['mask'][index])
                assert ACTIONS.index(teacher['action']) == expected['y'][index]
                index += 1
        assert index == len(expected['y']); reconstructed[split] = index
    paths = [('train', output/'data/train_public_replays.jsonl.gz'),
             ('validation', output/'data/validation_public_replays.jsonl.gz'),
             ('test', output/'evaluation/public_replays.jsonl.gz')]
    for split, path in paths:
        with gzip.open(path, 'rt', encoding='utf-8') as file:
            for line in file:
                item = json.loads(line); public_only(item)
                mi = int(item['map_id'].rsplit('m', 1)[1])
                seed = truth['splits'][item['preset']][split][mi]
                training.verify(seed, item['preset'], item['first'], item['swap'], item['replay']); count += 1
    for index in range(1, 4):
        name = f'L3_run{index:02d}'; info = read(output/'models'/(name+'_training.json'))
        assert sha(output/'models'/(name+'.npz')) == info['sha256']
        assert info['selected']['validation_loss'] == min(x['validation_loss'] for x in info['history'])
        model = dict(np.load(output/'models'/(name+'.npz'), allow_pickle=False))
        measured = training.measure(model, load_arrays(output, 'validation'))
        assert abs(measured[0]-info['selected']['validation_loss']) < 1e-7
        assert sum(x.size for x in model.values()) == PARAMETERS
    result = {'completed_utc': utc_now(), 'independent_maps': len(seen), 'split_counts': dict(split_counts),
              'overlap_with_prior_maps': 0, 'overlap_between_splits_including_dihedral': 0,
              'public_replays_independently_verified': count, 'features_reconstructed': reconstructed,
              'all_checkpoint_hashes_and_validation_losses_verified': True,
              'public_truth_fields_absent': True, 'baseline_unchanged': True, 'human_records': 0}
    write(output/'integrity_audit.json', result)
    print('AUDIT_COMPLETE '+json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('all', 'freeze', 'collect', 'train', 'evaluate', 'audit'))
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--private', required=True, type=Path)
    parser.add_argument('--exclude-manifest', action='append', default=[], type=Path)
    args = parser.parse_args()
    out, private = validate_paths(args.output, args.private)
    if args.stage in ('all', 'freeze'): create_protocol(out, private, args.exclude_manifest)
    if args.stage in ('all', 'collect'): collect(out, private)
    if args.stage in ('all', 'train'): train_all(out, private)
    if args.stage in ('all', 'evaluate'): evaluate(out, private)
    if args.stage in ('all', 'audit'): audit(out, private)


if __name__ == '__main__':
    main()
