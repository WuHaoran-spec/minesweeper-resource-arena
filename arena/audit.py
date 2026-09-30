"""Read-only experiment integrity audit. Private truth is used by this runner only."""
import csv
import gzip
import hashlib
import json
from collections import Counter
import numpy as np
from .env import Arena
from .experiment import verify_episode
from .learning import DATA, ROOT, PRIVATE, get_splits, sha, write_json
from .policies import MODEL_DIR, ACTIONS, prepare, choose


def canonical_map(env):
    """Canonicalize mines AND resources over eight dihedral transforms."""
    versions=[]
    for reflect in (False,True):
        for turns in range(4):
            def transform(cell):
                r,c=cell
                if reflect: c=8-c
                for _ in range(turns): r,c=c,8-r
                return r,c
            versions.append(json.dumps([sorted(map(transform,env._mines)),
                                        sorted(map(transform,env._initial_diamonds))]))
    return hashlib.sha256(min(versions).encode()).hexdigest()


def audit():
    splits=get_splits(); layouts={}; checks=[]
    for split,seeds in splits.items():
        for seed in seeds:
            signature=canonical_map(Arena(seed))
            if signature in layouts:
                raise AssertionError('Duplicate/mirrored base map across experiment partitions')
            layouts[signature]=split
    model_info={}
    for path in sorted(MODEL_DIR.glob('*_training.json')):
        info=json.loads(path.read_text(encoding='utf-8'))
        model_path=MODEL_DIR/info['file']
        assert sha(model_path)==info['sha256']
        assert info['selected']['validation_loss']==min(x['validation_loss'] for x in info['history'])
        with np.load(model_path,allow_pickle=False) as model:
            assert model['W1'].shape==(27,32) and model['W2'].shape==(32,1)
            assert all(np.isfinite(model[k]).all() for k in model.files)
            assert sum(model[k].size for k in model.files)==929
        model_info[info['file']]={'sha256':info['sha256'],'selected_epoch':info['selected_epoch'],
                                 'optimizer_steps':info['optimizer_steps'],'parameters':929}
    first_wins=last_wins=draws=eliminated_winners=0
    formal_games=0; adjacent=set(); risk=Counter(); stages={}
    forbidden={'seed','map_seed','mines','mine_map','hidden_mines','private_snapshot','future_results'}
    def public_only(value):
        if isinstance(value,dict):
            assert not (set(value)&forbidden)
            for item in value.values(): public_only(item)
        elif isinstance(value,list):
            for item in value: public_only(item)
    for stage,split in [('initial','test_initial'),('retest','test_retest'),('random','test_random')]:
        base=DATA/'evaluation'/stage
        rows={r['episode_id']:r for r in csv.DictReader((base/'games.csv').open(encoding='utf-8'))}
        count=0
        with gzip.open(base/'public_replays.jsonl.gz','rt',encoding='utf-8') as source:
            for line in source:
                episode=json.loads(line); row=rows[episode['episode_id']]
                public_only(episode)
                index=int(episode['map_id'].rsplit('-',1)[1])
                replay=episode['replay']
                verify_episode(splits[split][index],int(row['first']),bool(int(row['swap'])),replay)
                final=replay['final_observation']
                assert str(final['winner'])==row['winner']
                assert final['scores']==[int(row['score0']),int(row['score1'])]
                assert final['steps']==int(row['steps']) and final['reason']==row['reason']
                first=int(row['first']); winner=final['winner']
                if winner=='draw': draws+=1
                elif winner==first: first_wins+=1
                else: last_wins+=1
                if winner!='draw' and not final['alive'][winner]: eliminated_winners+=1
                obs=replay['initial_observation']
                if any(abs(r-pr)+abs(c-pc)==1 for r,c in obs['diamonds'] for pr,pc in obs['positions']):
                    adjacent.add(episode['map_id'])
                risk.update(x['metadata']['risk_status'] for x in replay['records'])
                count+=1
        assert count==len(rows)
        formal_games+=count
        stages[stage]={'replays_verified_again':count,'csv_sha256':sha(base/'games.csv'),
                       'replay_gzip_sha256':sha(base/'public_replays.jsonl.gz')}
    data_provenance={}
    for name in ('train','validation','train_corrections'):
        with np.load(DATA/(name+'_features.npz'),allow_pickle=False) as arrays:
            X,y,mask=arrays['X'],arrays['y'],arrays['mask']
            assert X.shape==(len(y),5,27) and arrays['X_no_opponent'].shape==X.shape
            assert np.isfinite(X).all() and mask[np.arange(len(y)),y].all()
            expected={key:arrays[key].copy() for key in arrays.files}
        seen=set(); rebuilt={key:[] for key in expected}; games=0
        with gzip.open(DATA/(name+'_public_replays.jsonl.gz'),'rt',encoding='utf-8') as source:
            for line in source:
                episode=json.loads(line); public_only(episode); games+=1
                replay=episode['replay']
                learned=name=='train_corrections'
                failed=learned and replay['result']['winner'] not in ('draw',episode['learner_seat'])
                for rec in replay['records']:
                    obs=rec['observation']; teacher=choose(obs,'B2')
                    if learned and not (failed or rec['action']!=teacher['action']): continue
                    public={k:v for k,v in obs.items() if k not in ('game_id','revision')}
                    signature=json.dumps(public,sort_keys=True)
                    if signature in seen: continue
                    seen.add(signature)
                    features,legal,*_=prepare(obs)
                    rebuilt['X'].append(features)
                    rebuilt['X_no_opponent'].append(prepare(obs,True)[0])
                    rebuilt['mask'].append(legal)
                    rebuilt['y'].append(ACTIONS.index(teacher['action']))
        for key in expected:
            np.testing.assert_array_equal(expected[key],np.asarray(rebuilt[key],dtype=expected[key].dtype))
        data_provenance[name]={'games':games,'states':len(seen),
                               'all_features_and_labels_rebuilt_from_public_replays':True,
                               'features_sha256':sha(DATA/(name+'_features.npz'))}
    result={'base_maps_total':len(layouts),'base_map_partitions':{k:len(v) for k,v in splits.items()},
            'no_duplicate_or_dihedral_maps':True,'private_manifest_sha256':sha(PRIVATE/'map_manifest.json'),
            'formal_games':formal_games,'replays_verified_again':formal_games,'stages':stages,
            'test_base_maps_with_adjacent_diamond':len(adjacent),'first_player_wins':first_wins,
            'second_player_wins':last_wins,'draws':draws,'eliminated_final_winners':eliminated_winners,
            'risk_status_counts':dict(risk),'models':model_info,'feature_shape_and_legal_labels':True,
            'observation_and_replay_no_truth_fields':True,'human_records':0,
            'public_data_feature_reconstruction':data_provenance}
    write_json(DATA/'final_integrity_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': audit()
