"""Post-hoc descriptive paired comparison; no training or model selection.

Reads the completed, frozen batch CSV. Writes only the explicitly requested
evaluation/posthoc_overall_comparison.json. It does not rerun any policy.

Usage: python analyze_overall.py path/to/batch
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

BASELINE = 'L3_previous'
CANDIDATES = ('L3_run01', 'L3_run02', 'L3_run03')
PRESETS = ('intermediate', 'expert')
METRICS = ('win_half_draw', 'gems', 'lives', 'actions', 'normalized_utility')
RNG_SEED = 62001
RESAMPLES = 10000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value(row, metric):
    if metric == 'win_half_draw':
        return 1.0 if row['winner'] == '1' else .5 if row['winner'] == 'draw' else 0.0
    return float(row[metric+'1'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('batch_dir', type=Path,
                        help='Batch directory containing evaluation/ and selection_before_test.json')
    args = parser.parse_args()
    run = args.batch_dir.resolve()
    source = run/'evaluation/games.csv'
    with source.open(encoding='utf8', newline='') as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1280:
        raise ValueError('Expected complete 1280-game batch')
    names = (BASELINE, *CANDIDATES)
    matched = defaultdict(dict)
    for row in rows:
        key = (row['preset'], row['map_id'], row['policy0'], row['swap'], row['first'])
        if row['policy1'] in matched[key]:
            raise ValueError('Duplicate matched policy/condition')
        matched[key][row['policy1']] = row
    if len(matched) != 320 or any(set(group) != set(names) for group in matched.values()):
        raise ValueError('Baseline/candidate matched conditions are incomplete')
    for preset in PRESETS:
        map_counts = Counter(key[1] for key in matched if key[0] == preset)
        if len(map_counts) != 20 or set(map_counts.values()) != {8}:
            raise ValueError('Expected 20 base maps per difficulty, each with 2 opponents x 4 variants')
    selection_path = run/'selection_before_test.json'
    selection = json.loads(selection_path.read_text(encoding='utf8'))
    run_manifest = json.loads((run/'evaluation/run_manifest.json').read_text(encoding='utf8'))
    selection_unchanged = sha(selection_path) == run_manifest['selection_sha256']
    if not selection_unchanged or selection['policy'] != 'L3_run02':
        raise ValueError('Original validation nomination changed')
    totals = []
    for name in names:
        group = [row for row in rows if row['policy1'] == name]
        total = {'policy': name, 'games': len(group),
                 'wins': sum(row['winner'] == '1' for row in group),
                 'draws': sum(row['winner'] == 'draw' for row in group),
                 'losses': sum(row['winner'] == '0' for row in group),
                 'timeouts': sum(row['reason'] == 'max_steps' for row in group)}
        total.update({metric: float(np.mean([value(row, metric) for row in group])) for metric in METRICS})
        totals.append(total)
    rng = np.random.default_rng(RNG_SEED)
    comparisons = []
    # RNG is continuous in this exact candidate/metric order, not reseeded per row.
    for candidate in CANDIDATES:
        for metric in METRICS:
            by_preset = defaultdict(lambda: defaultdict(list))
            for key, group in matched.items():
                by_preset[key[0]][key[1]].append(value(group[candidate], metric)-value(group[BASELINE], metric))
            strata = [np.asarray([np.mean(values) for values in by_preset[preset].values()]) for preset in PRESETS]
            samples = np.mean([v[rng.integers(0, len(v), (RESAMPLES, len(v)))].mean(1) for v in strata], axis=0)
            interval = np.quantile(samples, [.025, .975]).tolist()
            comparisons.append({'policy': candidate, 'reference': BASELINE, 'metric': metric,
                'difference': float(np.mean([v.mean() for v in strata])),
                'stratified_paired_map_bootstrap_95ci': interval,
                'independent_maps': 40, 'includes_zero': interval[0] <= 0 <= interval[1]})
    report = {'schema': 'l3-expanded-posthoc-overall-comparison-v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'analysis_kind': 'post-hoc descriptive comparison; not a preregistered aggregate endpoint',
        'used_for_training': False, 'used_for_checkpoint_selection': False,
        'used_for_candidate_nomination': False, 'used_to_change_game_default': False,
        'source_games_csv': 'evaluation/games.csv', 'source_games_csv_sha256': sha(source),
        'analysis_script_sha256': sha(Path(__file__).resolve()),
        'validation_nominee_before_test': selection['policy'],
        'validation_nominee_sha256': selection['sha256'],
        'selection_file_sha256': sha(selection_path), 'selection_hash_matches_test_manifest': selection_unchanged,
        'resampling_rng': 'numpy.random.default_rng / PCG64', 'resampling_rng_seed': RNG_SEED,
        'resamples': RESAMPLES, 'interval_method': 'percentile 2.5% and 97.5%',
        'resampling_order': {'candidates': list(CANDIDATES), 'metrics': list(METRICS), 'presets': list(PRESETS),
                             'map_order': 'first appearance in the frozen CSV', 'rng_reseed_per_comparison': False},
        'sampling_unit': 'base map; all 2 opponents x 4 seat/first-turn conditions remain together',
        'stratification': 'sample 20 base maps with replacement inside each difficulty; average two difficulty means equally',
        'independent_maps': 40, 'maps_per_difficulty': 20, 'opponents_per_map': 2, 'variants_per_opponent': 4,
        'policies_share_same_maps': True, 'three_initializations_are_not_120_independent_test_maps': True,
        'totals': totals, 'paired_comparisons': comparisons,
        'conclusion': 'All three aggregate win-rate-difference intervals include zero; no reliable improvement established.',
        'efficiency_caution': 'Candidate mean action counts increased; all corresponding difference intervals include zero. Fewer actions can also reflect early death, so no efficiency improvement is claimed.',
        'limitations': ['40 base maps and two fixed opponents only',
                       'same maps reused for all three independent initializations',
                       'descriptive multiple comparisons; no familywise multiplicity correction',
                       'test-best candidate must not replace the validation-nominated run02',
                       'larger data, fresh initialization and omission of prior correction differ together; not a controlled data-size-only ablation']}
    if not all(row['includes_zero'] for row in comparisons if row['metric'] == 'win_half_draw'):
        raise ValueError('Unexpected source change: review conclusions before writing')
    destination = run/'evaluation/posthoc_overall_comparison.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print(json.dumps({'output': str(destination), 'source_sha256': report['source_games_csv_sha256'],
                      'totals': totals, 'win_rate_comparisons': [r for r in comparisons if r['metric'] == 'win_half_draw']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
