"""Public-state survival-sensitive B3 and genuinely fitted L3 imitation policy.

The risk/path costs are heuristics, not independent-path survival probabilities
or an optimal solver for survival-v1 utility. Legal masks remain environment
geometry/public-known-mine masks; unknown mines are never consulted.
"""
from collections import Counter
from functools import lru_cache
import hashlib
import heapq
import numpy as np
from .policies import ACTIONS, DELTAS, MODEL_DIR, logits
from .risk import infer

FEATURES = 40
HIDDEN = 48
PARAMETERS = 2017


def risk_weight(lives):
    return 8.0 * (3.0 / max(1, lives)) ** 2


@lru_cache(maxsize=512)
def paths(target, probabilities, width, height, weight):
    distance = [float('inf')] * (width * height)
    distance[target] = 0.0
    queue = [(0.0, target)]
    while queue:
        cost, cell = heapq.heappop(queue)
        if cost != distance[cell]:
            continue
        r, c = divmod(cell, width)
        p = probabilities[cell]
        edge = 100000.0 if p >= 1 - 1e-12 else 1 + weight * p / max(.03, 1-p)
        for dr, dc in DELTAS[:4]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < height and 0 <= cc < width:
                j = rr * width + cc
                candidate = cost + edge
                if candidate < distance[j]:
                    distance[j] = candidate
                    heapq.heappush(queue, (candidate, j))
    return tuple(distance)


def prepare_survival(obs):
    width, height = obs.get('width', obs['size']), obs.get('height', obs['size'])
    me, other = obs['turn'], 1-obs['turn']
    pos, opponent = obs['positions'][me], obs['positions'][other]
    lives = obs.get('lives', [int(x) for x in obs['alive']])
    weight = risk_weight(lives[me])
    p, status, risk_detail = infer(obs)
    probabilities = tuple(p)
    cell = lambda rc: rc[0] * width + rc[1]
    here, opp = cell(pos), cell(opponent)
    targets = sorted(obs['diamonds'])
    ds = [paths(cell(t), probabilities, width, height, weight) for t in targets]
    live_opp = bool(obs['alive'][other])
    # Public distance for the opponent uses the SAME risk map, its own life count.
    ods = [paths(cell(t), probabilities, width, height, risk_weight(lives[other]))[opp]
           if live_opp else 0.0 for t in targets]
    def contest(d, od):
        return d + .25 * max(0., d-od+.5) if live_opp else d
    selected = sorted(range(len(targets)), key=lambda j: (contest(ds[j][here], ods[j]), targets[j]))[:3]
    history = obs.get('recent_positions', [[list(pos)], [list(opponent)]])[me][-24:]
    visits = Counter(map(tuple, history))
    previous = tuple(history[-2]) if len(history) > 1 else None
    total, scale = obs.get('resource_count', 3), float(max(width, height))
    revealed = {cell(x[:2]) for x in obs['revealed']}
    actions = obs.get('action_counts', [0, 0])
    X = np.zeros((5, FEATURES), np.float32)
    base = np.full((5, max(1, len(targets))), 1e6)
    costs = base.copy()
    repeat_costs = np.zeros(5)
    legal = np.array([a in obs['legal_actions'] for a in ACTIONS], dtype=bool)
    for a, (dr, dc) in enumerate(DELTAS):
        rr, cc = pos[0]+dr, pos[1]+dc
        if not (0 <= rr < height and 0 <= cc < width):
            continue
        candidate = (rr, cc)
        i = cell(candidate)
        edge = 100000. if p[i] >= 1-1e-12 else 1 + weight*p[i]/max(.03, 1-p[i])
        repeat = min(8., .75*visits[candidate]) + (1.5 if candidate == previous else 0.)
        repeat += 5. if a == 4 else 0.
        repeat_costs[a] = repeat
        X[a, :9] = [p[i], float(i in revealed), float(a == 4),
                     (obs['scores'][me]-obs['scores'][other])/total,
                     obs['steps']/obs['max_steps'], len(targets)/total, float(live_opp),
                     rr/max(1, height-1), cc/max(1, width-1)]
        for j, dist in enumerate(ds):
            d = edge+dist[i]
            base[a, j] = d
            costs[a, j] = contest(d, ods[j])+repeat
        for slot, j in enumerate(selected):
            d, od, now = base[a, j], ods[j], ds[j][here]
            # Log distance avoids clipping all long/high-risk routes to one value.
            X[a, 9+slot*6:15+slot*6] = [np.log1p(d)/np.log1p(4*scale),
                np.log1p(now)/np.log1p(4*scale), np.clip(now-d, -scale, scale)/(scale/4),
                np.log1p(od)/np.log1p(4*scale) if live_opp else 0.,
                np.clip(d-od, -scale, scale)/(scale/2) if live_opp else 0., float(i == cell(targets[j]))]
        for slot in range(len(selected), 3):
            X[a, 9+slot*6] = 4.
        X[a, 27:32] = [width/30, height/30, obs['mine_count']/(width*height), lives[me]/3, lives[other]/3]
        X[a, 32:] = [actions[me]/obs['max_steps'], actions[other]/obs['max_steps'],
                     visits[candidate]/24, float(candidate == previous), weight/72,
                     repeat/14.5, float(status == 'exact_global_model_count'),
                     float(obs.get('scoring') == 'survival-v1')]
    detail = {'risk': risk_detail, 'risk_weight': weight, 'history_window': len(history),
              'repeat_costs': {a: round(float(repeat_costs[i]), 5) for i, a in enumerate(ACTIONS) if legal[i]},
              'network_candidate_resources': [targets[j] for j in selected],
              'candidate_selection': 'three lowest public life-weighted route/contest costs',
              'risk_cost_semantics': 'additive heuristic; not joint path survival or optimal utility'}
    return X, legal, p, status, detail, targets, base, costs


@lru_cache(maxsize=8)
def _load(path, stamp):
    with np.load(path, allow_pickle=False) as file:
        model = {k: file[k].copy() for k in file.files}
    if model['W1'].shape != (FEATURES, HIDDEN):
        raise ValueError('Incorrect L3 model architecture')
    return model, hashlib.sha256(path.read_bytes()).hexdigest()


def load_survival(policy='L3'):
    filename = 'survival_v3_initial.npz' if policy == 'L3_initial' else 'survival_v3_final.npz'
    path = MODEL_DIR / filename
    return _load(path, path.stat().st_mtime_ns)


def choose_survival(obs, policy='B3', rng=None):
    if obs['done'] or not obs['legal_actions']:
        raise ValueError('Cannot choose action in terminal state')
    X, legal, p, status, detail, targets, base, costs = prepare_survival(obs)
    diag = dict(detail, version=policy+'-survival-v1', target_kind='explicit_heuristic_target')
    if policy == 'B3':
        values = costs.min(1)
        values[~legal] = 1e12
        ai = int(np.argmin(values))
        diag['action_costs'] = {a: round(float(values[i]), 5) for i, a in enumerate(ACTIONS) if legal[i]}
    elif policy in ('L3', 'L3_initial'):
        model, model_hash = load_survival(policy)
        values = logits(X, model)
        values[~legal] = -1e9
        ai = int(np.argmax(values))
        diag.update(model=policy, model_sha256=model_hash,
                    target_kind='inferred_from_selected_action_not_network_head',
                    action_logits={a: round(float(values[i]), 6) for i, a in enumerate(ACTIONS) if legal[i]})
    else:
        raise ValueError('Unknown survival policy: '+policy)
    if obs.get('scoring') != 'survival-v1' or 'recent_positions' not in obs:
        diag['out_of_distribution'] = 'L3 trained on arena-v3 survival-v1 with public recent positions'
    tj = int(np.argmin(costs[ai])) if targets else None
    return {'action': ACTIONS[ai], 'target': targets[tj] if tj is not None else None,
            'policy': policy, 'risk_status': status, 'risk': p, 'diagnostics': diag}
