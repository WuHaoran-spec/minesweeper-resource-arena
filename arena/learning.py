"""Real NumPy behavior cloning plus one train-only failure correction round.

Run: python -m arena.learning initial | refine | all
No mine layout/outcome is used as a risk-probability label. Teacher actions are
heuristic labels; imitation is not reinforcement learning or an optimality proof.
"""
import argparse
import gzip
import hashlib
import json
import os
import secrets
import time
from pathlib import Path
import numpy as np
from .env import Arena, RULE_VERSION
from .policies import ACTIONS, FEATURES, MODEL_DIR, prepare, choose, logits

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get('ARENA_OUTPUT_DIR', str(ROOT))) / 'data'
SPLIT_SIZES = {'train': 200, 'validation': 30, 'test_initial': 100,
               'test_retest': 100, 'test_random': 100}
PRIVATE = Path(os.environ.get('ARENA_PRIVATE_DIR', str(ROOT.parent / 'private_delivery' / 'experiment_truth')))
if PRIVATE.resolve().is_relative_to(ROOT.resolve()):
    raise ValueError('ARENA_PRIVATE_DIR must be outside the public repository')
SPLITS = None  # Loaded only by the trusted experiment runner, never by policies.
SEED = 20260930


def get_splits():
    global SPLITS
    if SPLITS is not None:
        return SPLITS
    PRIVATE.mkdir(parents=True, exist_ok=True)
    manifest = PRIVATE / 'map_manifest.json'
    if manifest.exists():
        saved = json.loads(manifest.read_text(encoding='utf-8'))
        SPLITS = saved['splits']
    else:
        used = set()
        SPLITS = {}
        for name, count in SPLIT_SIZES.items():
            values = []
            while len(values) < count:
                value = secrets.randbits(63)
                if value not in used:
                    used.add(value)
                    values.append(value)
            SPLITS[name] = values
        write_json(manifest, {'purpose': 'PRIVATE: reproducibility only; never policy input or public export',
                              'splits': SPLITS})
    assert {k: len(v) for k, v in SPLITS.items()} == SPLIT_SIZES
    flat = [seed for values in SPLITS.values() for seed in values]
    assert len(flat) == len(set(flat)), 'Base-map seed overlaps across partitions'
    return SPLITS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def freeze_protocol():
    DATA.mkdir(parents=True, exist_ok=True)
    get_splits()
    protocol = {'protocol_version': '2026-09-30-private-splits-v2',
                'rule_version': RULE_VERSION, 'map_split_counts': SPLIT_SIZES, 'training_seed': SEED,
                'map_manifest_sha256': sha(PRIVATE / 'map_manifest.json'),
                'map_manifest_location': 'private experiment directory, outside the public repository',
                'split_unit': 'base map; seats/first-player/replays never cross splits',
                'initial_pairs': [['B1','B2'], ['B1','L_initial'], ['B2','L_initial']],
                'retest_pairs': [['B1','B2'], ['B1','L'], ['B2','L'], ['B1','L_initial'], ['B2','L_initial'],
                                 ['B1','L_no_opponent'], ['B2','L_no_opponent']],
                'random_pairs': [['B0','B1'], ['B0','B2'], ['B0','L']],
                'variants_per_map_pair': 'swap false/true x first 0/1',
                'max_steps': 200, 'bootstrap': '2000 resamples of base-map groups',
                'supplemental_selection': 'training maps only; all decision states from L_initial losses plus action disagreements',
                'training': {'method':'behavior cloning; then one DAgger-like supervised correction round',
                             'teacher':'B2-v1', 'epochs_initial':45, 'epochs_refine':20, 'batch_size':256,
                             'hidden':32, 'optimizer':'Adam', 'learning_rate':.002, 'label':'teacher action',
                             'reward_shaping':None, 'training_seeds':1,
                             'checkpoint_selection': 'lowest validation cross entropy within each fixed epoch budget'},
                'human_records': 0, 'policy_input_excludes': ['map_seed','hidden_mines','future_outcomes']}
    path = DATA / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != protocol:
            raise RuntimeError('Frozen protocol differs; version it explicitly instead of silently overwriting')
    else:
        write_json(path, protocol)
    return protocol


def collect(name, seeds, learned=False):
    """Public states are logged; runner seed manifest stays separate from policy input."""
    xs, xn, ys, masks = [], [], [], []
    counts = {'games':0, 'raw_states':0, 'unique_states':0, 'failures':0, 'disagreements':0,
              'source': 'heuristic_teacher_data' if not learned else 'learning_policy_rollout_with_teacher_corrections',
              'human_records':0}
    seen = set()
    with gzip.open(DATA / (name+'_public_replays.jsonl.gz'), 'wt', encoding='utf-8') as stream:
        for ix, seed in enumerate(seeds):
            for variant in range(2):
                env = Arena(seed, first=variant, swap=bool(ix%2))
                learner = variant
                records = []
                while not env.done:
                    obs = env.observe()
                    teacher = choose(obs, 'B2')
                    if learned:
                        actual = choose(obs, 'L_initial' if obs['turn']==learner else ('B1' if ix%2 else 'B2'))
                    else:
                        actual = teacher if obs['turn']==variant else choose(obs, 'B1' if ix%2 else 'B2')
                    X, legal, *_ = prepare(obs)
                    Xn = prepare(obs, True)[0]
                    teacher_y = ACTIONS.index(teacher['action'])
                    disagreement = actual['action'] != teacher['action']
                    records.append((obs, X, Xn, teacher_y, legal, disagreement))
                    env.step(actual['action'], actual)
                failed = learned and env.winner not in ('draw', learner)
                counts['games'] += 1
                counts['failures'] += int(failed)
                for obs, X, Xn, y, mask, disagreement in records:
                    counts['raw_states'] += 1
                    counts['disagreements'] += int(disagreement)
                    if learned and not (failed or disagreement):
                        continue
                    public = {k:v for k,v in obs.items() if k not in ('game_id','revision')}
                    signature = json.dumps(public, sort_keys=True)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    xs.append(X); xn.append(Xn); ys.append(y); masks.append(mask)
                map_group = 'train' if learned else name
                stream.write(json.dumps({'map_id':f'{map_group}-map{ix:03d}', 'source':counts['source'],
                                         'learner_seat': learner if learned else None,
                                         'replay':env.save_replay()}, separators=(',',':'))+'\n')
            if ix%25==24:
                print(f'{name}: {ix+1}/{len(seeds)} maps, {len(xs)} states', flush=True)
    counts['unique_states'] = len(xs)
    arrays = {'X':np.asarray(xs, dtype=np.float32), 'X_no_opponent':np.asarray(xn, dtype=np.float32),
              'y':np.asarray(ys, dtype=np.int64), 'mask':np.asarray(masks, dtype=bool)}
    np.savez_compressed(DATA / (name+'_features.npz'), **arrays)
    write_json(DATA / (name+'_collection.json'), counts)
    return arrays, counts


def _measure(model, X, y, mask):
    z = logits(X.reshape(-1, FEATURES), model).reshape(len(X), 5)
    z[~mask] = -1e9
    z -= z.max(axis=1, keepdims=True)
    probs = np.exp(z); probs /= probs.sum(axis=1, keepdims=True)
    loss = -np.log(probs[np.arange(len(y)), y]+1e-12).mean()
    accuracy = (probs.argmax(axis=1)==y).mean()
    return float(loss), float(accuracy)


def fit(train, valid, filename, epochs, initial=None, no_opponent=False):
    rng = np.random.default_rng(SEED)
    key = 'X_no_opponent' if no_opponent else 'X'
    X, y, mask = train[key], train['y'], train['mask']
    model = {k:v.copy() for k,v in initial.items()} if initial else {
        'W1':rng.normal(0, .14, (FEATURES,32)).astype(np.float32), 'b1':np.zeros(32,dtype=np.float32),
        'W2':rng.normal(0,.14,(32,1)).astype(np.float32), 'b2':np.zeros(1,dtype=np.float32)}
    ms = {k:np.zeros_like(v) for k,v in model.items()}; vs = {k:np.zeros_like(v) for k,v in model.items()}
    history = []; steps = 0; began = time.perf_counter()
    best_loss = float('inf'); best_model = None; best_epoch = None
    for epoch in range(epochs):
        order = rng.permutation(len(X))
        for begin in range(0,len(X),256):
            idx = order[begin:begin+256]
            xb = X[idx].reshape(-1,FEATURES)
            h = np.tanh(xb@model['W1']+model['b1'])
            z = (h@model['W2']+model['b2']).reshape(len(idx),5)
            z[~mask[idx]] = -1e9
            z -= z.max(axis=1,keepdims=True)
            probs = np.exp(z); probs /= probs.sum(axis=1,keepdims=True)
            probs[np.arange(len(idx)),y[idx]] -= 1
            dz = (probs/len(idx)).reshape(-1,1)
            dh = (dz@model['W2'].T)*(1-h*h)
            grads = {'W2':h.T@dz, 'b2':dz.sum(axis=0), 'W1':xb.T@dh, 'b1':dh.sum(axis=0)}
            steps += 1
            for k, gradient in grads.items():
                gradient = np.clip(gradient,-5,5)
                ms[k] = .9*ms[k]+.1*gradient
                vs[k] = .999*vs[k]+.001*gradient*gradient
                model[k] -= .002*(ms[k]/(1-.9**steps))/(np.sqrt(vs[k]/(1-.999**steps))+1e-8)
        train_loss, train_acc = _measure(model,X,y,mask)
        val_loss,val_acc = _measure(model,valid[key],valid['y'],valid['mask'])
        history.append({'epoch':epoch+1,'train_loss':train_loss,'train_accuracy':train_acc,
                        'validation_loss':val_loss,'validation_accuracy':val_acc,'optimizer_steps':steps})
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch+1
            best_model = {k: v.copy() for k, v in model.items()}
        if epoch%5==4 or epoch==0:
            print(f'{filename}: epoch {epoch+1}/{epochs}, loss {train_loss:.4f}, val acc {val_acc:.4f}',flush=True)
    MODEL_DIR.mkdir(exist_ok=True)
    output = MODEL_DIR/filename
    model = best_model
    np.savez_compressed(output,**model)
    elapsed = time.perf_counter()-began
    info = {'file':filename,'sha256':sha(output),'seed':SEED,'method':'supervised behavior cloning',
            'teacher':'B2-v1','train_examples':len(y),'validation_examples':len(valid['y']),
            'epochs':epochs,'optimizer_steps':steps,'elapsed_seconds':elapsed,'no_opponent':no_opponent,
            'initial_weights':initial is not None,'history':history,'final':history[-1],
            'selection': 'minimum validation loss', 'selected_epoch':best_epoch,
            'selected':history[best_epoch-1], 'parameter_count':sum(v.size for v in model.values())}
    write_json(MODEL_DIR/(output.stem+'_training.json'),info)
    print(json.dumps({k:v for k,v in info.items() if k!='history'}),flush=True)
    return model, info


def load_arrays(name):
    with np.load(DATA/(name+'_features.npz'),allow_pickle=False) as data:
        return {k:data[k].copy() for k in data.files}


def initial_train():
    freeze_protocol()
    began = time.perf_counter()
    train,_ = collect('train', SPLITS['train'])
    valid,_ = collect('validation', SPLITS['validation'])
    fit(train,valid,'L_initial.npz',45)
    # It is also the playable checkpoint until the declared refinement completes.
    (MODEL_DIR/'L_final.npz').write_bytes((MODEL_DIR/'L_initial.npz').read_bytes())
    write_json(DATA/'initial_stage_timing.json', {'elapsed_seconds_including_collection': time.perf_counter()-began})


def refine_train():
    freeze_protocol()
    began = time.perf_counter()
    train=load_arrays('train'); valid=load_arrays('validation')
    supplemental,counts = collect('train_corrections',SPLITS['train'],learned=True)
    # One correction round, with no selection using either test partition.
    combined={k:np.concatenate([train[k],supplemental[k]]) for k in train}
    with np.load(MODEL_DIR/'L_initial.npz',allow_pickle=False) as data:
        model={k:data[k].copy() for k in data.files}
    fit(combined,valid,'L_final.npz',20,initial=model)
    # Matched ablation: identical samples, labels, architecture, seed, and epoch budgets.
    no_opp,_=fit(train,valid,'L_no_opponent_initial.npz',45,no_opponent=True)
    fit(combined,valid,'L_no_opponent.npz',20,initial=no_opp,no_opponent=True)
    write_json(DATA/'refinement.json',{'selection':'training maps only; losses or action disagreements',
                                     'collection':counts,'combined_examples':len(combined['y']),
                                     'test_used_for_training':False,'rounds':1})
    write_json(DATA/'refine_stage_timing.json', {'elapsed_seconds_including_collection': time.perf_counter()-began})


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['initial','refine','all'])
    args=parser.parse_args()
    if args.stage in ('initial','all'): initial_train()
    if args.stage in ('refine','all'): refine_train()
