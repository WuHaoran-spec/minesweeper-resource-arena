"""Real CPU imitation training on held-out rectangular, multi-life arena-v2 maps.

Only this trusted runner reads private map seeds. The policy sees public observe().
Run stages in order: collect, initial, refine, evaluate, audit, report (or all).
The published arena-v1 datasets and weights are never overwritten.
"""
import argparse
from collections import Counter, defaultdict
import csv
import datetime
import faulthandler
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import secrets
import time
import numpy as np
from .env import Arena
from .policies import ACTIONS, MODEL_DIR, CHALLENGE_FEATURES, choose, prepare_v2, logits

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('ARENA_OUTPUT_DIR',str(ROOT)))
DATA=OUT/'data'/'challenge_v2'
PRIVATE=Path(os.environ.get('ARENA_PRIVATE_DIR',str(ROOT.parent/'private_delivery'/'experiment_truth')))
PRESETS=('intermediate','expert')
COUNTS={'train':20,'validation':6,'test':8}
TRAIN_SEED=20261001
FORBIDDEN={'seed','map_seed','mines','mine_map','hidden_mines','private_snapshot','future_results'}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(env):
    width,height=env.width,env.height;forms=[]
    for reflect in (False,True):
        for turns in range(4):
            def transform(cell):
                r,c=cell;w,h=width,height
                if reflect:c=w-1-c
                for _ in range(turns):r,c,w,h=c,h-1-r,h,w
                return r,c
            w,h=(height,width) if turns%2 else (width,height)
            forms.append(json.dumps([w,h,sorted(map(transform,env._mines)),
                                    sorted(map(transform,env._initial_diamonds))]))
    return hashlib.sha256(min(forms).encode()).hexdigest()


def manifest():
    if PRIVATE.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Private challenge truth must stay outside the public repository')
    path=PRIVATE/'challenge_manifest.json'
    if path.exists():
        saved=read(path)
        # Prospectively amended CPU pilot uses prefixes; no outcome-based map removal.
        for preset in PRESETS:
            for split,count in COUNTS.items():
                saved['splits'][preset][split]=saved['splits'][preset][split][:count]
                saved['canonical_fingerprints'][preset][split]=saved['canonical_fingerprints'][preset][split][:count]
        return saved
    seen=set();used=set();splits={};fingerprints={}
    for preset in PRESETS:
        splits[preset]={};fingerprints[preset]={}
        for split,count in COUNTS.items():
            seeds=[];prints=[]
            while len(seeds)<count:
                seed=secrets.randbits(63)
                if seed in used:continue
                env=Arena(seed,preset=preset);fingerprint=canonical(env)
                if fingerprint in seen:continue
                seen.add(fingerprint);used.add(seed);seeds.append(seed);prints.append(fingerprint)
            splits[preset][split]=seeds;fingerprints[preset][split]=prints
    obj={'private':True,'rule_version':'arena-v2.0','splits':splits,'canonical_fingerprints':fingerprints}
    write(path,obj);return obj


def freeze():
    truth=manifest();DATA.mkdir(parents=True,exist_ok=True)
    value={'protocol_version':'challenge-v2-20260930-cpu-pilot','rule_version':'arena-v2.0',
           'base_map_counts_per_preset':COUNTS,'presets':list(PRESETS),
           'manifest_sha256':sha(PRIVATE/'challenge_manifest.json'),
           'split_unit':'base map; seat, first-player, mirror, snapshot and replay stay in one split',
           'train_games_per_map':2,'test_variants_per_map':'swap false/true x first 0/1',
           'test_pairs':[['B1','B2'],['B1','L_v2_initial'],['B2','L_v2_initial'],
                         ['B1','L_v2'],['B2','L_v2'],['B0','L_v2']],
           'method':'B2-v2 supervised imitation; one training-map teacher correction round',
           'features':32,'hidden':48,'parameters':1633,'optimizer':'Adam',
           'learning_rate':.002,'batch_size':256,'epochs_initial':35,'epochs_correction':18,
           'training_seed':TRAIN_SEED,'training_seeds_count':1,
           'checkpoint_selection':'lowest held-out validation cross entropy within fixed epoch budget',
           'teacher_label':'heuristic B2 action, not optimality or exact mine-probability ground truth',
           'supplemental_selection':'training games lost by initial learner or any teacher-action disagreement',
           'state_sampling':'deduplicate complete public state excluding opaque ids/revision; at most 192 equally spaced eligible states per game',
           'raw_log_policy':'all actions, failures, draws and timeouts retained; downsampling affects feature training only',
           'risk_module_sha256':sha(ROOT/'arena/risk.py'),'policy_module_sha256':sha(ROOT/'arena/policies.py'),
           'environment_module_sha256':sha(ROOT/'arena/env.py'),
           'human_records':0,'reward_shaping':None}
    path=DATA/'protocol.json'
    if path.exists() and read(path)!=value: raise RuntimeError('Frozen challenge protocol differs; use a new output directory')
    if not path.exists():write(path,value)
    return truth,value


def public_only(value):
    if isinstance(value,dict):
        if set(value)&FORBIDDEN:raise AssertionError('Private truth in public record')
        for item in value.values():public_only(item)
    elif isinstance(value,(list,tuple)):
        for item in value:public_only(item)


def comparable(obs):return {k:v for k,v in obs.items() if k!='game_id'}


def verify(seed,preset,first,swap,replay):
    env=Arena(seed,first=first,swap=swap,preset=preset)
    assert comparable(env.observe())==comparable(replay['initial_observation'])
    for record in replay['records']:
        assert comparable(env.observe())==comparable(record['observation'])
        assert comparable(env.step(record['action']))==comparable(record['next_observation'])
    assert comparable(env.observe())==comparable(replay['final_observation'])


def model_hash(policy):
    name={'L_v2':'challenge_final.npz','L_v2_initial':'challenge_initial.npz'}.get(policy)
    return sha(MODEL_DIR/name) if name else None


def quantiles(values):
    return {str(q):float(np.quantile(values,q)) for q in [0,.1,.25,.5,.75,.9,1]} if values else {}


def collect(split='train',correction=False):
    truth,protocol=freeze();name='corrections' if correction else split
    arrays={k:[] for k in ['X','y','mask']};seen=set();counts=Counter();risk_counts=Counter();lengths=[]
    preset_lengths=defaultdict(list);began=time.perf_counter()
    policy_hash=sha(ROOT/'arena/policies.py')
    hashes={p:model_hash(p) for p in ['B0','B1','B2']+(['L_v2_initial'] if correction else [])}
    with gzip.open(DATA/(name+'_public_replays.jsonl.gz'),'wt',encoding='utf-8') as rp, \
         gzip.open(DATA/(name+'_transitions.jsonl.gz'),'wt',encoding='utf-8') as transitions:
        for preset in PRESETS:
            for mi,seed in enumerate(truth['splits'][preset][split]):
                for variant in (0,1):
                    swap=bool(mi%2);first=variant;learner=variant
                    env=Arena(seed,first=first,swap=swap,preset=preset)
                    opponent=('B1','B2','B0')[mi%3]
                    policies=[opponent,opponent];policies[learner]='L_v2_initial' if correction else 'B2'
                    rng=random.Random(11103+mi*17+variant)
                    pending=[]
                    while not env.done:
                        obs=env.observe();teacher=choose(obs,'B2');actor=obs['turn']
                        actual=teacher if policies[actor]=='B2' else choose(obs,policies[actor],rng)
                        X,mask,p,status,*_=prepare_v2(obs)
                        assert X.shape==(5,32) and len(p)==obs['width']*obs['height']
                        assert np.isfinite(X).all() and all(0<=v<=1 for v in p)
                        y=ACTIONS.index(teacher['action']);assert mask[y]
                        pending.append((X,mask,y,teacher['target'],actual['action']!=teacher['action']))
                        env.step(actual['action'],actual)
                        if env.steps%100==0:
                            print(f'{name}/{preset} map {mi+1} variant {variant}: {env.steps} steps, {len(obs["revealed"])} public clues',flush=True)
                    replay=env.save_replay();verify(seed,preset,first,swap,replay)
                    failed=correction and env.winner not in ('draw',learner)
                    episode_id=f'{name}-{preset}-m{mi:03d}-v{variant}'
                    map_id=f'{preset}-{split}-m{mi:03d}'
                    eligible=[]
                    for ix,(record,details) in enumerate(zip(replay['records'],pending)):
                        obs=record['observation'];disagree=details[-1]
                        counts['raw_transitions']+=1;counts['teacher_disagreements']+=int(disagree)
                        risk_counts[record['metadata']['risk_status']]+=1
                        if correction and not (failed or disagree):continue
                        signature=json.dumps({k:v for k,v in obs.items() if k not in ('game_id','revision')},sort_keys=True)
                        if signature in seen:counts['duplicate_states']+=1;continue
                        seen.add(signature);eligible.append(ix)
                    chosen=set(eligible if len(eligible)<=192 else [eligible[i] for i in np.linspace(0,len(eligible)-1,192,dtype=int)])
                    for ix,(record,details) in enumerate(zip(replay['records'],pending)):
                        X,mask,y,target,disagree=details
                        if ix in chosen:
                            arrays['X'].append(X);arrays['mask'].append(mask);arrays['y'].append(y)
                        item={'schema':'challenge-public-transition-v1','rule_version':'arena-v2.0',
                              'episode_id':episode_id,'map_id':map_id,'split':split,'preset':preset,
                              'source':'learning_policy_rollout_with_teacher_corrections' if correction else 'heuristic_teacher_data',
                              'policy_source_sha256':policy_hash,'model_sha256':hashes[policies[record['actor']]],
                              'transition':record,'teacher_action':ACTIONS[y],'teacher_target':target,
                              'teacher_label_source':'B2-v2 public-state heuristic; not optimal-action ground truth',
                              'teacher_disagreement':disagree,'selected_for_features':ix in chosen,
                              'selected_for_training':ix in chosen and split=='train',
                              'feature_use':'validation_only' if split=='validation' else 'gradient_training',
                              'terminal_label_separate':replay['result'],'human_record':False}
                        public_only(item);transitions.write(json.dumps(item,separators=(',',':'))+'\n')
                    item={'schema':'challenge-public-episode-v1','episode_id':episode_id,'map_id':map_id,
                          'preset':preset,'split':split,'first':first,'swap':swap,'policies':policies,
                          'learner_seat':learner,'replay':replay}
                    public_only(item);rp.write(json.dumps(item,separators=(',',':'))+'\n')
                    counts['games']+=1;counts['learner_losses']+=int(failed)
                    counts['draws']+=int(env.winner=='draw');counts['timeouts']+=int(env.reason=='max_steps')
                    lengths.append(env.steps);preset_lengths[preset].append(env.steps)
                    rp.flush();transitions.flush()
                    print(f'{name}/{preset}: map {mi+1} variant {variant}, steps {env.steps}, {counts["games"]} games, {len(arrays["y"])} selected states, {time.perf_counter()-began:.1f}s',flush=True)
    saved={'X':np.asarray(arrays['X'],dtype=np.float32),'mask':np.asarray(arrays['mask'],dtype=bool),
           'y':np.asarray(arrays['y'],dtype=np.int64)}
    np.savez_compressed(DATA/(name+'_features.npz'),**saved)
    result=dict(counts);result.update(selected_states=len(arrays['y']),risk_status_counts=dict(risk_counts),
        game_step_quantiles=quantiles(lengths),step_quantiles_by_preset={k:quantiles(v) for k,v in preset_lengths.items()},
        elapsed_seconds=time.perf_counter()-began,all_replays_verified=counts['games'],
        features_sha256=sha(DATA/(name+'_features.npz')),human_records=0)
    write(DATA/(name+'_collection.json'),result);print(json.dumps(result),flush=True)
    return saved


def load_arrays(name):
    with np.load(DATA/(name+'_features.npz'),allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def measure(model,X,y,mask):
    z=logits(X.reshape(-1,32),model).reshape(-1,5);z[~mask]=-1e9;z-=z.max(axis=1,keepdims=True)
    probs=np.exp(z);probs/=probs.sum(axis=1,keepdims=True)
    return float(-np.log(probs[np.arange(len(y)),y]+1e-12).mean()),float((probs.argmax(1)==y).mean())


def fit(train,valid,filename,epochs,initial=None):
    rng=np.random.default_rng(TRAIN_SEED)
    model={k:v.copy() for k,v in initial.items()} if initial else {
        'W1':rng.normal(0,.12,(32,48)).astype(np.float32),'b1':np.zeros(48,np.float32),
        'W2':rng.normal(0,.12,(48,1)).astype(np.float32),'b2':np.zeros(1,np.float32)}
    m={k:np.zeros_like(v) for k,v in model.items()};v={k:np.zeros_like(value) for k,value in model.items()}
    X,y,mask=train['X'],train['y'],train['mask'];history=[];steps=0;best=float('inf');selected=None
    began=time.perf_counter()
    for epoch in range(epochs):
        for indexes in np.array_split(rng.permutation(len(y)),int(np.ceil(len(y)/256))):
            xb=X[indexes].reshape(-1,32);h=np.tanh(xb@model['W1']+model['b1'])
            z=(h@model['W2']+model['b2']).reshape(-1,5);z[~mask[indexes]]=-1e9;z-=z.max(1,keepdims=True)
            probs=np.exp(z);probs/=probs.sum(1,keepdims=True);probs[np.arange(len(indexes)),y[indexes]]-=1
            dz=(probs/len(indexes)).reshape(-1,1);dh=(dz@model['W2'].T)*(1-h*h)
            grads={'W2':h.T@dz,'b2':dz.sum(0),'W1':xb.T@dh,'b1':dh.sum(0)};steps+=1
            for key,g in grads.items():
                g=np.clip(g,-5,5);m[key]=.9*m[key]+.1*g;v[key]=.999*v[key]+.001*g*g
                model[key]-=.002*(m[key]/(1-.9**steps))/(np.sqrt(v[key]/(1-.999**steps))+1e-8)
        tl,ta=measure(model,X,y,mask);vl,va=measure(model,valid['X'],valid['y'],valid['mask'])
        row={'epoch':epoch+1,'train_loss':tl,'train_accuracy':ta,'validation_loss':vl,'validation_accuracy':va,'optimizer_steps':steps}
        history.append(row)
        if vl<best:best=vl;selected=row.copy();bestmodel={k:value.copy() for k,value in model.items()}
        if epoch%5==4 or epoch==0:print(f'{filename} epoch {epoch+1}/{epochs} loss={tl:.4f} val_acc={va:.4f}',flush=True)
    MODEL_DIR.mkdir(parents=True,exist_ok=True);path=MODEL_DIR/filename;np.savez_compressed(path,**bestmodel)
    info={'file':filename,'sha256':sha(path),'rule_version':'arena-v2.0','presets':list(PRESETS),
          'features':32,'hidden':48,'parameter_count':1633,'method':'supervised behavior cloning of B2-v2',
          'training_seed':TRAIN_SEED,'epochs':epochs,'optimizer_steps':steps,'train_examples':len(y),
          'validation_examples':len(valid['y']),'history':history,'selected':selected,
          'selection':'lowest validation loss','elapsed_optimization_seconds':time.perf_counter()-began,
          'initial_weights':initial is not None,'human_records':0}
    write(MODEL_DIR/(path.stem+'_training.json'),info);return bestmodel


def initial():
    freeze();model=fit(load_arrays('train'),load_arrays('validation'),'challenge_initial.npz',35)
    (MODEL_DIR/'challenge_final.npz').write_bytes((MODEL_DIR/'challenge_initial.npz').read_bytes())
    return model


def refine():
    extra=collect('train',True);train=load_arrays('train');valid=load_arrays('validation')
    combined={k:np.concatenate([train[k],extra[k]]) for k in train}
    with np.load(MODEL_DIR/'challenge_initial.npz',allow_pickle=False) as data:model={k:data[k].copy() for k in data.files}
    fit(combined,valid,'challenge_final.npz',18,model)
    write(DATA/'refinement.json',{'rounds':1,'test_used_for_training':False,'combined_examples':len(combined['y']),
                               'correction_examples':len(extra['y']),'two_rounds_may_repeat_states':True})


def summarize(rows):
    groups=defaultdict(list);rng=np.random.default_rng(7661);result=[]
    for row in rows:groups[(row['preset'],row['policy0'],row['policy1'])].append(row)
    for (preset,p0,p1),group in groups.items():
        maps=defaultdict(list)
        for row in group:maps[row['map_id']].append(row)
        scores=np.array([np.mean([1 if r['winner']==1 else .5 if r['winner']=='draw' else 0 for r in g]) for g in maps.values()])
        sample=scores[rng.integers(0,len(scores),(2000,len(scores)))].mean(1)
        wins=sum(r['winner']==1 for r in group);draws=sum(r['winner']=='draw' for r in group)
        result.append({'preset':preset,'policy':p1,'opponent':p0,'games':len(group),'base_maps':len(maps),
            'wins':wins,'draws':draws,'losses':len(group)-wins-draws,'win_plus_half_draw':float(scores.mean()),
            'map_bootstrap_95ci':np.quantile(sample,[.025,.975]).tolist(),
            'mean_diamond_difference':float(np.mean([r['score1']-r['score0'] for r in group])),
            'timeouts':sum(r['reason']=='max_steps' for r in group),
            'policy_eliminations':sum(r['lives1']==0 for r in group),'opponent_eliminations':sum(r['lives0']==0 for r in group),
            'game_step_quantiles':quantiles([r['steps'] for r in group]),
            'policy_mean_decision_ms':1000*sum(r['time1'] for r in group)/max(1,sum(r['actions1'] for r in group))})
    return result


def evaluate(limit=None):
    truth,protocol=freeze();label='evaluation' if limit is None else f'pilot{limit}'
    out=DATA/label;out.mkdir(exist_ok=True);rows=[];risk=Counter();began=time.perf_counter()
    case_counts=Counter();cases=[]
    hashes={p:model_hash(p) for p in ['L_v2_initial','L_v2']}
    write(out/'run_manifest.json',{'rule_version':'arena-v2.0','source':'automated_synthetic_policy_matches',
          'human_records':0,'model_sha256':hashes,'protocol_sha256':sha(DATA/'protocol.json'),
          'limited_maps_per_preset':limit,'stage':label})
    with gzip.open(out/'public_replays.jsonl.gz','wt',encoding='utf-8') as output:
        for preset in PRESETS:
            for mi,seed in enumerate(truth['splits'][preset]['test'][:limit]):
                for pi,(p0,p1) in enumerate(protocol['test_pairs']):
                    for swap in (False,True):
                        for first in (0,1):
                            env=Arena(seed,first=first,swap=swap,preset=preset)
                            rng=random.Random(62219+mi*11+int(swap)*2+first);times=[0.,0.];actions=[0,0]
                            while not env.done:
                                obs=env.observe();actor=obs['turn'];t=time.perf_counter()
                                decision=choose(obs,(p0,p1)[actor],rng);times[actor]+=time.perf_counter()-t
                                actions[actor]+=1;risk[decision['risk_status']]+=1;env.step(decision['action'],decision)
                            replay=env.save_replay();verify(seed,preset,first,swap,replay)
                            eid=f'{preset}-m{mi:03d}-p{pi}-s{int(swap)}-f{first}';map_id=f'{preset}-test-m{mi:03d}'
                            item={'schema':'challenge-public-episode-v1','episode_id':eid,'map_id':map_id,'preset':preset,
                                  'first':first,'swap':swap,'policies':[p0,p1],'model_sha256':hashes,'replay':replay}
                            public_only(item);output.write(json.dumps(item,separators=(',',':'))+'\n')
                            previous=[None,None]
                            for rec in replay['records']:
                                obs=rec['observation'];actor=rec['actor'];target=rec['metadata']['target'];kinds=[]
                                explicit=rec['metadata']['diagnostics'].get('target_kind')=='explicit_heuristic_target'
                                if explicit and target and target==previous[1-actor] and all(obs['alive']) and len(obs['diamonds'])>1:
                                    kinds.append('explicit_resource_contest')
                                if explicit and previous[actor] and previous[actor]!=target and previous[actor] in obs['diamonds'] and target in obs['diamonds']:
                                    kinds.append('explicit_target_switch')
                                if (p0,p1)[actor]=='L_v2' and rec['feedback'].get('respawned'):
                                    kinds.append('learned_mine_hit_and_respawn')
                                if rec['metadata']['risk_status']=='approximate_cutoff':kinds.append('budgeted_approximate_risk')
                                for kind in kinds:
                                    if case_counts[kind]>=2:continue
                                    file=f'cases/{kind}_{case_counts[kind]+1}.json'
                                    case_path=out/file;case_path.parent.mkdir(parents=True,exist_ok=True)
                                    case_path.write_text(json.dumps({'kind':kind,'episode_id':eid,'step':rec['step'],
                                          'detection':'public logged predicate; no intent or human review claim','replay':replay},
                                          separators=(',',':')),encoding='utf-8')
                                    cases.append({'kind':kind,'episode_id':eid,'step':rec['step'],'file':file})
                                    case_counts[kind]+=1
                                previous[actor]=target if explicit else None
                            rows.append({'episode_id':eid,'map_id':map_id,'preset':preset,'policy0':p0,'policy1':p1,
                                'swap':int(swap),'first':first,'winner':env.winner,'score0':env.scores[0],'score1':env.scores[1],
                                'lives0':env.lives[0],'lives1':env.lives[1],'steps':env.steps,'reason':env.reason,
                                'actions0':actions[0],'actions1':actions[1],'time0':times[0],'time1':times[1]})
                if (mi+1)%5==0:print(f'{label}/{preset}: {mi+1} maps, {len(rows)} games, {time.perf_counter()-began:.1f}s',flush=True)
    with (out/'games.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'games':len(rows),'independent_test_maps':sum(len(truth['splits'][p]['test'][:limit]) for p in PRESETS),
            'all_replays_verified':len(rows),'risk_status_counts':dict(risk),'elapsed_seconds':time.perf_counter()-began,
            'game_step_quantiles':quantiles([r['steps'] for r in rows]),'pairs':summarize(rows),'model_sha256':hashes,
            'draws':sum(r['winner']=='draw' for r in rows),'timeouts':sum(r['reason']=='max_steps' for r in rows),
            'human_records':0,'verified_case_candidates':cases}
    effects=[];rng=np.random.default_rng(44031)
    for preset in PRESETS:
        for opponent in ['B1','B2']:
            groups=defaultdict(lambda:defaultdict(list))
            for row in rows:
                if row['preset']==preset and row['policy0']==opponent and row['policy1'] in ['L_v2','L_v2_initial']:
                    groups[row['map_id']][row['policy1']].append(1 if row['winner']==1 else .5 if row['winner']=='draw' else 0)
            values=np.asarray([np.mean(x['L_v2'])-np.mean(x['L_v2_initial']) for x in groups.values()])
            bootstrap=values[rng.integers(0,len(values),(2000,len(values)))].mean(1)
            effects.append({'preset':preset,'opponent':opponent,'left':'L_v2','right':'L_v2_initial',
                            'base_maps':len(values),'win_plus_half_draw_difference':float(values.mean()),
                            'paired_map_bootstrap_95ci':np.quantile(bootstrap,[.025,.975]).tolist()})
    write(out/'paired_effects.json',effects)
    write(out/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def audit():
    truth,_=freeze();seen=set();model_results={};count=0
    for preset in PRESETS:
        for split,seeds in truth['splits'][preset].items():
            for seed in seeds:
                fingerprint=canonical(Arena(seed,preset=preset));assert fingerprint not in seen;seen.add(fingerprint)
    for stem in ['challenge_initial','challenge_final']:
        info=read(MODEL_DIR/(stem+'_training.json'));assert info['sha256']==sha(MODEL_DIR/(stem+'.npz'))
        assert info['selected']['validation_loss']==min(h['validation_loss'] for h in info['history'])
        with np.load(MODEL_DIR/(stem+'.npz'),allow_pickle=False) as weights:
            assert weights['W1'].shape==(32,48) and sum(weights[k].size for k in weights.files)==1633
        model_results[stem]=info['sha256']
    for stage in ['train','validation','corrections','evaluation']:
        path=DATA/stage/'public_replays.jsonl.gz' if stage=='evaluation' else DATA/(stage+'_public_replays.jsonl.gz')
        with gzip.open(path,'rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line);public_only(item);preset=item['preset'];split='test' if stage=='evaluation' else ('train' if stage=='corrections' else stage)
                mi=int(item['map_id'].rsplit('m',1)[1]);seed=truth['splits'][preset][split][mi]
                verify(seed,preset,item['first'],item['swap'],item['replay']);count+=1
    rebuilt={}
    for stage in ['train','validation','corrections']:
        with np.load(DATA/(stage+'_features.npz'),allow_pickle=False) as d:arrays={k:d[k].copy() for k in d.files}
        idx=0;raw=0
        with gzip.open(DATA/(stage+'_transitions.jsonl.gz'),'rt',encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line);public_only(item);raw+=1
                if not item['selected_for_features']:continue
                obs=item['transition']['observation'];X,mask,*_=prepare_v2(obs);teacher=choose(obs,'B2')
                assert item['teacher_action']==teacher['action'];np.testing.assert_array_equal(arrays['X'][idx],X)
                np.testing.assert_array_equal(arrays['mask'][idx],mask);assert arrays['y'][idx]==ACTIONS.index(teacher['action']);idx+=1
        assert idx==len(arrays['y']);rebuilt[stage]={'raw_transitions':raw,'selected_features_rebuilt':idx}
    result={'base_maps':len(seen),'dihedral_map_overlap':0,'all_public_replays_verified_again':count,
            'public_truth_fields_absent':True,'model_hashes':model_results,'feature_reconstruction':rebuilt,'human_records':0}
    write(DATA/'integrity_audit.json',result);print(json.dumps(result),flush=True)


def report():
    protocol=read(DATA/'protocol.json');summary=read(DATA/'evaluation/summary.json');integrity=read(DATA/'integrity_audit.json')
    tr,va,co=[read(DATA/(x+'_collection.json')) for x in ['train','validation','corrections']]
    initial,final=[read(MODEL_DIR/(x+'_training.json')) for x in ['challenge_initial','challenge_final']]
    text='# 大棋盘挑战版 v2：真实训练与数据\n\n'
    text+='旧arena-v1.0的9×9模型、5200局日志保持独立，本报告只对应arena-v2.0。中级16×16、40雷、7钻石；高级30列×16行、99雷、11钻石。三生命、角落3×3安全区、踩雷复活和公开资源竞争是本项目变体，局长上限分别800/1600。没有保证钻石全可达。\n\nL_v2属于教师模仿，不是无教师自由探索；独立E策略的黑箱回报搜索使用自己的协议、分区和报告，不能混用本表结果。\n\n'
    text+=f"## 训练与可信边界\n\n本轮是有计算预算的小规模验证，每难度{COUNTS['train']}训练图、{COUNTS['validation']}验证图、{COUNTS['test']}测试图，共{2*sum(COUNTS.values())}张独立基础图。完整训练/验证/测试之前按CPU预算修订规模，使用原冻结随机分区的前缀、包含当前困难图；原提案与修订原因分别保存在protocol_initial_proposal.json和protocol_amendment.json，没有按对战结果筛选地图。模型32个公开动作特征→48个tanh隐藏单元→1动作评分，1633参数；先按公开路线/竞争代价稳定选择3个资源槽，规则教师仍考虑全部资源。这是工程特征选择，不是模型自主学会目标选择。旧L在v2明确标记分布外。\n\n"
    text+=f"初始训练{tr['games']}局、{tr['raw_transitions']}条原始转移，选入{tr['selected_states']}状态；验证{va['games']}局、{va['selected_states']}状态。初训35轮、{initial['optimizer_steps']}优化步，验证最优第{initial['selected']['epoch']}轮，动作准确率{initial['selected']['validation_accuracy']:.2%}。纠正轮仅使用训练图，{co['learner_losses']}个输局、{co['teacher_disagreements']}个动作分歧，选入{co['selected_states']}状态；再训18轮、{final['optimizer_steps']}更新，验证最优第{final['selected']['epoch']}轮、准确率{final['selected']['validation_accuracy']:.2%}。只有一个训练种子，未评估跨种子稳定性。\n\n"
    text+=f"最终模型SHA256：`{final['sha256']}`。学习的是B2启发式动作标签，不是最优动作；没有强化学习、真人数据或将单次雷位当概率标签。准确率不能代替胜率。\n\n"
    text+='## 留出实测\n\n|难度|策略 / 对手|胜 / 平 / 负|胜+半平|95%基础图区间|步数中位数 / 90%分位|超时|\n|---|---|---:|---:|---|---:|---:|\n'
    for p in summary['pairs']:
        lo,hi=p['map_bootstrap_95ci'];q=p['game_step_quantiles']
        text+=f"|{p['preset']}|{p['policy']} / {p['opponent']}|{p['wins']} / {p['draws']} / {p['losses']}|{p['win_plus_half_draw']:.3%}|{lo:.3%}–{hi:.3%}|{q['0.5']:.0f} / {q['0.9']:.0f}|{p['timeouts']}|\n"
    text+=f"\n正式{summary['games']}局、{summary['independent_test_maps']}独立测试基础图，每图对阵交换起点与先手共4条件；保留{summary['draws']}平局及{summary['timeouts']}超时。全部局长分位数：`{summary['game_step_quantiles']}`。风险状态计数：`{summary['risk_status_counts']}`。区间按地图分组bootstrap2000次，不把变体当独立地图。每难度只有{COUNTS['test']}测试图，估计有限；逐表报告，不预设学习优于教师。\n\n"
    if (DATA/'pilot1/summary.json').exists():
        pilot=read(DATA/'pilot1/summary.json')
        text+=f"先行小批量验证{pilot['games']}局、{pilot['independent_test_maps']}基础图，与正式留出图重叠，单列pilot1，不重复计入正式局数或独立图数。\n\n"
    text+='同图补训减初训的配对比较：\n\n|难度|对手|胜+半平差 / 百分点|配对95%区间 / 百分点|\n|---|---|---:|---|\n'
    for e in read(DATA/'evaluation/paired_effects.json'):
        lo,hi=e['paired_map_bootstrap_95ci']
        text+=f"|{e['preset']}|{e['opponent']}|{100*e['win_plus_half_draw_difference']:+.3f}|{100*lo:+.3f} 至 {100*hi:+.3f}|\n"
    text+='\n小样本区间包含0时不确认收益；即使点估计改善也不能宣称稳定优势。退化为[0.5,0.5]或[1,1]的bootstrap区间只说明本次观察地图的分组结果相同，不意味着总体无不确定性。\n\n'
    text+='## 数据质量与公开下载结构\n\n“高质量”指可核验的规则、来源、分区、动作和标签，不意味着教师最优。所有失败/平局/超时原始轨迹均保留。训练去重键为除不透明game_id/revision外的完整公开观察；保留时间预算信息。每局最多均匀取192个符合条件状态，防止超长循环支配梯度；未选状态仍在原始转移日志。纠正筛选是训练图输局或教师动作分歧，未使用测试图标签。两轮合并可能重复。\n\n'
    text+='`data/challenge_v2/*_transitions.jsonl.gz`每行一条公开转移，含ruleset、preset、map_id、split、策略源码哈希、实际模型哈希、合法动作/风险状态/实际动作与反馈、B2教师动作及来源、是否进入训练；终局胜负另存terminal_label_separate，不输入策略。`*_public_replays.jsonl.gz`每行完整对局。NPZ是已选训练特征。私有生成种子只在仓库外challenge_manifest.json，公开协议仅含哈希。\n\n'
    text+=f"审计：{integrity['base_maps']}张图及旋转/镜像等价形式无重复；{integrity['all_public_replays_verified_again']}局完整轨迹再次逐步重放一致。公开转移重新构造全部选中特征、掩码和教师标签，与NPZ逐元素一致：`{integrity['feature_reconstruction']}`。人工玩家记录0。\n\n"
    text+='复现：`python -m arena.challenge_training all`；可依次执行collect、initial、refine、evaluate、audit、report。已有协议不同会报错。建议以ARENA_OUTPUT_DIR指定新输出目录，并以ARENA_PRIVATE_DIR指定仓库外的私有清单目录；旧v1默认权重文件不会被此流程覆盖。\n'
    (ROOT/'docs/challenge_v2_experiments.md').write_text(text,encoding='utf-8')
    card={'schema':'challenge-dataset-card-v1','source':'self-generated arena-v2 simulation; MIT',
          'human_records':0,'teacher':'B2-v2 heuristic labels, not optimal ground truth','allowed_observation':'public Arena.observe fields only',
          'split_counts':protocol['base_map_counts_per_preset'],'presets':list(PRESETS),'training_collection':tr,
          'validation_collection':va,'correction_collection':co,'model_sha256':final['sha256'],
          'quality_gates':['private truth fields excluded','legal teacher labels','finite public features',
                           'rule version verified','base-map disjoint including dihedral transforms',
                           'full replay audit','all chosen features rebuilt from public transitions'],
          'limitations':['one training seed','heuristic imitation, not optimal control','time-outs retained',
                         'network uses only three selected resource slots','no real human records']}
    write(DATA/'dataset_card.json',card);print('challenge report and dataset card written',flush=True)


if __name__=='__main__':
    if os.environ.get('ARENA_TRACE_STACKS') == '1':
        faulthandler.dump_traceback_later(45,repeat=True)
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['collect','initial','refine','evaluate','audit','report','all'])
    parser.add_argument('--limit',type=int);args=parser.parse_args()
    if args.stage in ('collect','all'):collect('train');collect('validation')
    if args.stage in ('initial','all'):initial()
    if args.stage in ('refine','all'):refine()
    if args.stage in ('evaluate','all'):evaluate(args.limit)
    if args.stage in ('audit','all'):audit()
    if args.stage in ('report','all'):report()
