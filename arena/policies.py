"""Policies consume a public observation only. No environment or seed input."""
import heapq
import random
import os
from pathlib import Path
import numpy as np
from .risk import infer

ACTIONS = ('up', 'down', 'left', 'right', 'wait')
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
MODEL_DIR = Path(os.environ.get('ARENA_OUTPUT_DIR', str(Path(__file__).resolve().parents[1]))) / 'models'
_MODELS = {}
FEATURES = 27


def _cell(pos, size):
    return pos[0]*size + pos[1]


def _paths(target, p, size):
    """Additive risk penalty is a heuristic cost, not joint path survival."""
    dst = [float('inf')] * (size*size)
    dst[target] = 0.
    queue = [(0., target)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != dst[i]:
            continue
        r, c = divmod(i, size)
        edge = 1. + 5. * p[i] / max(.03, 1. - p[i])
        if p[i] >= 1 - 1e-12:
            edge = 10000.
        for dr, dc in DELTAS[:4]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < size and 0 <= cc < size:
                j = rr*size+cc
                alt = cost+edge
                if alt < dst[j]:
                    dst[j] = alt
                    heapq.heappush(queue, (alt, j))
    return dst


def prepare(obs, no_opponent=False):
    size = obs['size']
    p, status, detail = infer(obs)
    me = obs['turn']
    pos = obs['positions'][me]
    other = obs['positions'][1-me]
    targets = sorted(obs['diamonds'])
    distances = [_paths(_cell(t, size), p, size) for t in targets]
    here = _cell(pos, size)
    opp = _cell(other, size)
    live_opp = bool(obs['alive'][1-me]) and not no_opponent
    revealed = {_cell(x[:2], size) for x in obs['revealed']}
    X = np.zeros((5, FEATURES), dtype=np.float32)
    costs = np.full((5, max(1, len(targets))), 1e6, dtype=float)
    base = np.full_like(costs, 1e6)
    for a, (dr, dc) in enumerate(DELTAS):
        rr, cc = pos[0]+dr, pos[1]+dc
        if not (0 <= rr < size and 0 <= cc < size):
            continue
        i = rr*size+cc
        edge = 1. + 5.*p[i]/max(.03, 1.-p[i])
        if p[i] >= 1-1e-12:
            edge = 10000.
        X[a, :9] = [p[i], float(i in revealed), float(a == 4),
                     (obs['scores'][me]-(0 if no_opponent else obs['scores'][1-me]))/3., obs['steps']/obs['max_steps'],
                     len(targets)/3., float(live_opp), rr/(size-1), cc/(size-1)]
        for j, (target, dist) in enumerate(zip(targets, distances)):
            d = edge+dist[i] + (2 if a == 4 else 0)
            od = dist[opp] if live_opp else 0.
            contest = max(0., d-od+.5) if live_opp else 0.
            costs[a, j] = d+.7*contest-(1./(1.+abs(d-od)) if live_opp else 0.)
            base[a, j] = d
            X[a, 9+j*6:15+j*6] = [min(d, 80)/20., min(dist[here], 80)/20.,
                                      np.clip(dist[here]-d, -20, 20)/5.,
                                      min(od, 80)/20. if live_opp else 0.,
                                      np.clip(d-od, -20, 20)/10. if live_opp else 0.,
                                      float(i == _cell(target, size))]
        # Missing resource slots are distinguished from zero-distance resources.
        for j in range(len(targets), 3):
            X[a, 9+j*6] = 4.
    legal = np.array([a in obs['legal_actions'] for a in ACTIONS], dtype=bool)
    return X, legal, p, status, detail, targets, base, costs


def load_model(policy='L'):
    filename = {'L': 'L_final.npz', 'L_initial': 'L_initial.npz',
                'L_no_opponent': 'L_no_opponent.npz'}.get(policy, policy)
    path = MODEL_DIR / filename
    stamp = path.stat().st_mtime_ns  # Missing weights are an explicit error.
    key = (str(path), stamp)
    if key not in _MODELS:
        with np.load(path, allow_pickle=False) as data:
            _MODELS[key] = {k: data[k].copy() for k in data.files}
    return _MODELS[key]


def logits(X, model):
    hidden = np.tanh(X @ model['W1'] + model['b1'])
    return (hidden @ model['W2'] + model['b2']).reshape(-1)


def choose(obs, policy='B2', rng=None):
    if obs['done'] or not obs['legal_actions']:
        raise ValueError('Cannot choose action from a terminal observation')
    X, legal, p, status, detail, targets, base, costs = prepare(obs, policy == 'L_no_opponent')
    diag = {'risk': detail, 'version': policy+'-v1', 'target_kind': 'explicit_heuristic_target'}
    if policy == 'B0':
        action = (rng or random).choice(obs['legal_actions'])
        ai = ACTIONS.index(action)
        tj = None
    elif policy in ('B1', 'B2'):
        values = (base if policy == 'B1' else costs).min(axis=1)
        values[~legal] = 1e12
        ai = int(np.argmin(values))
        tj = int(np.argmin((base if policy == 'B1' else costs)[ai])) if targets else None
        action = ACTIONS[ai]
        diag['action_costs'] = {a: round(float(values[i]), 5) for i, a in enumerate(ACTIONS) if legal[i]}
    elif policy in ('L', 'L_initial', 'L_no_opponent'):
        values = logits(X, load_model(policy))
        values[~legal] = -1e9
        ai = int(np.argmax(values))
        action = ACTIONS[ai]
        tj = int(np.argmin(base[ai])) if targets else None
        diag.update({'action_logits': {a: round(float(values[i]), 6) for i, a in enumerate(ACTIONS) if legal[i]},
                     'target_kind': 'inferred_from_selected_action_not_network_head', 'model': policy})
    else:
        raise ValueError('Unknown policy: '+policy)
    return {'action': action, 'target': targets[tj] if tj is not None else None,
            'policy': policy, 'risk_status': status, 'risk': p, 'diagnostics': diag}
