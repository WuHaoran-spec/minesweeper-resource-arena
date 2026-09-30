"""Policies consume a public observation only. No environment or seed input."""
import heapq
import random
import os
from pathlib import Path
from functools import lru_cache
import numpy as np
from .risk import infer

ACTIONS = ('up', 'down', 'left', 'right', 'wait')
DELTAS = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))
MODEL_DIR = Path(os.environ.get('ARENA_OUTPUT_DIR', str(Path(__file__).resolve().parents[1]))) / 'models'
_MODELS = {}
FEATURES = 27
CHALLENGE_FEATURES = 32


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
    # Legacy feature numerics stay unchanged for the published v1 experiments.
    if obs.get('rule_version', '').startswith(('arena-v2', 'arena-v3')):
        return prepare_v2(obs, no_opponent)
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


@lru_cache(maxsize=512)
def _rect_paths(target, probabilities, width, height):
    """Reverse Dijkstra on rectangular public risk; no true map access."""
    dst = [float('inf')] * (width*height)
    dst[target] = 0.
    queue = [(0., target)]
    while queue:
        cost, i = heapq.heappop(queue)
        if cost != dst[i]:
            continue
        r, c = divmod(i, width)
        p = probabilities[i]
        edge = 10000. if p >= 1-1e-12 else 1.+5.*p/max(.03,1.-p)
        for dr, dc in DELTAS[:4]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < height and 0 <= cc < width:
                j = rr*width+cc
                alt = cost+edge
                if alt < dst[j]:
                    dst[j] = alt
                    heapq.heappush(queue, (alt, j))
    return tuple(dst)


def prepare_v2(obs, no_opponent=False):
    """32 public features; all resources considered by rules, top three by network.

    Resource selection uses current additive risk-route/contest score and stable
    coordinate tie breaks. It is engineered feature selection, not a learned
    target head. Losing lives or returning to spawn never reveals unknown mines.
    """
    width, height = obs.get('width',obs['size']), obs.get('height',obs['size'])
    p, status, detail = infer(obs)
    if len(p) != width*height:
        raise ValueError('Risk inference returned a non-rectangular probability map')
    me=obs['turn']; other=1-me; pos=obs['positions'][me]
    targets=sorted(obs['diamonds']); here=_cell(pos,width); opp=_cell(obs['positions'][other],width)
    probability_tuple=tuple(p)
    distances=[_rect_paths(_cell(t,width),probability_tuple,width,height) for t in targets]
    live_opp=bool(obs['alive'][other]) and not no_opponent
    def target_priority(j):
        d=distances[j][here]; od=distances[j][opp]
        score=d+.7*max(0.,d-od+.5)-1./(1.+abs(d-od)) if live_opp else d
        return score, targets[j][0], targets[j][1]
    selected=sorted(range(len(targets)),key=target_priority)[:3]
    revealed={_cell(x[:2],width) for x in obs['revealed']}
    total_resources=obs.get('resource_count',3)
    lives=obs.get('lives',[int(x) for x in obs['alive']])
    scale=float(max(width,height)); X=np.zeros((5,CHALLENGE_FEATURES),dtype=np.float32)
    base=np.full((5,max(1,len(targets))),1e6,dtype=float); costs=base.copy()
    for a,(dr,dc) in enumerate(DELTAS):
        rr,cc=pos[0]+dr,pos[1]+dc
        if not (0<=rr<height and 0<=cc<width): continue
        i=rr*width+cc; edge=10000. if p[i]>=1-1e-12 else 1.+5.*p[i]/max(.03,1.-p[i])
        X[a,:9]=[p[i],float(i in revealed),float(a==4),
                  (obs['scores'][me]-(0 if no_opponent else obs['scores'][other]))/total_resources,
                  obs['steps']/obs['max_steps'],len(targets)/total_resources,float(live_opp),
                  rr/max(1,height-1),cc/max(1,width-1)]
        for j,dist in enumerate(distances):
            d=edge+dist[i]+(2 if a==4 else 0); od=dist[opp] if live_opp else 0.
            base[a,j]=d
            costs[a,j]=d+.7*max(0.,d-od+.5)-(1./(1.+abs(d-od))) if live_opp else d
        for slot,j in enumerate(selected):
            dist=distances[j];d=base[a,j];od=dist[opp] if live_opp else 0.
            X[a,9+slot*6:15+slot*6]=[min(d,4*scale)/scale,min(dist[here],4*scale)/scale,
                np.clip(dist[here]-d,-scale,scale)/(scale/4),min(od,4*scale)/scale if live_opp else 0.,
                np.clip(d-od,-scale,scale)/(scale/2) if live_opp else 0.,float(i==_cell(targets[j],width))]
        for slot in range(len(selected),3): X[a,9+slot*6]=4.
        X[a,27:]=[width/30.,height/30.,obs['mine_count']/(width*height),lives[me]/3.,
                   lives[other]/3. if not no_opponent else 0.]
    legal=np.asarray([a in obs['legal_actions'] for a in ACTIONS],dtype=bool)
    detail=dict(detail)
    detail['network_candidate_resources']=[targets[j] for j in selected]
    detail['candidate_selection']='three lowest public current route/contest costs; stable coordinate ties'
    return X,legal,p,status,detail,targets,base,costs


def load_model(policy='L'):
    filename = {'L': 'L_final.npz', 'L_initial': 'L_initial.npz',
                'L_no_opponent': 'L_no_opponent.npz', 'L_v2': 'challenge_final.npz',
                'L_v2_initial':'challenge_initial.npz'}.get(policy, policy)
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
    if policy == 'E':
        from .exploration import choose_exploration
        return choose_exploration(obs, rng=rng)
    if policy in ('B3', 'L3', 'L3_initial'):
        from .survival import choose_survival
        return choose_survival(obs, policy=policy, rng=rng)
    v2=obs.get('rule_version','').startswith(('arena-v2','arena-v3'))
    prepare_fn=prepare_v2 if v2 or policy in ('L_v2','L_v2_initial') else prepare
    X, legal, p, status, detail, targets, base, costs = prepare_fn(obs, policy == 'L_no_opponent')
    diag = {'risk': detail, 'version': policy+('-v2' if v2 else '-v1'), 'target_kind': 'explicit_heuristic_target'}
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
    elif policy in ('L', 'L_initial', 'L_no_opponent','L_v2','L_v2_initial'):
        if v2 and policy in ('L','L_initial','L_no_opponent'):
            X=X[:,:FEATURES]
            diag['out_of_distribution']='Legacy 9x9 model on v2 board; this is not the v2 trained model'
        elif not v2 and policy in ('L_v2','L_v2_initial'):
            diag['out_of_distribution']='Challenge model on legacy rules; outside training presets'
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
