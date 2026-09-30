"""Frozen grouped-map evaluation, full public replays, and candidate case mining."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import random
from pathlib import Path
import time
import numpy as np
from .env import Arena
from .learning import DATA, ROOT, get_splits, freeze_protocol, write_json, sha
from .policies import choose, MODEL_DIR


def comparable(obs):
    return {k:v for k,v in obs.items() if k!='game_id'}


def verify_episode(seed, first, swap, replay):
    env=Arena(seed,first=first,swap=swap)
    assert comparable(env.observe())==comparable(replay['initial_observation'])
    for record in replay['records']:
        assert comparable(env.observe())==comparable(record['observation'])
        after=env.step(record['action'])
        assert comparable(after)==comparable(record['next_observation'])
    assert comparable(env.observe())==comparable(replay['final_observation'])


def grouped_summary(rows):
    rng=np.random.default_rng(9917)
    pair_groups=defaultdict(list)
    for row in rows:
        pair_groups[(row['policy0'],row['policy1'])].append(row)
    summary=[]
    for (p0,p1),group in pair_groups.items():
        maps=defaultdict(list)
        for row in group:
            maps[row['map_id']].append(row)
        map_scores=[]; map_diffs=[]
        for values in maps.values():
            map_scores.append(np.mean([1. if x['winner']==1 else .5 if x['winner']=='draw' else 0. for x in values]))
            map_diffs.append(np.mean([x['score1']-x['score0'] for x in values]))
        ids=rng.integers(0,len(maps),(2000,len(maps)))
        ci=np.quantile(np.asarray(map_scores)[ids].mean(axis=1),[.025,.975]).tolist()
        dci=np.quantile(np.asarray(map_diffs)[ids].mean(axis=1),[.025,.975]).tolist()
        wins=sum(x['winner']==1 for x in group); losses=sum(x['winner']==0 for x in group)
        summary.append({'policy':p1,'opponent':p0,'games':len(group),'independent_maps':len(maps),
                        'wins':wins,'draws':len(group)-wins-losses,'losses':losses,
                        'win_rate':wins/len(group),'win_plus_half_draw':float(np.mean(map_scores)),
                        'map_bootstrap_95ci_win_plus_half_draw':ci,
                        'mean_diamond_difference':float(np.mean(map_diffs)), 'map_bootstrap_95ci_diamond_difference':dci,
                        'eliminations_policy':sum(x['eliminated1'] for x in group),
                        'eliminations_opponent':sum(x['eliminated0'] for x in group),
                        'timeouts':sum(x['reason']=='max_steps' for x in group),
                        'mean_steps':float(np.mean([x['steps'] for x in group])),
                        'policy_mean_decision_ms':1000*sum(x['time1'] for x in group)/max(1,sum(x['actions1'] for x in group)),
                        'opponent_mean_decision_ms':1000*sum(x['time0'] for x in group)/max(1,sum(x['actions0'] for x in group))})
    return summary


def run(stage, limit=None):
    protocol=freeze_protocol()
    initial=stage=='initial'
    split={'initial':'test_initial', 'retest':'test_retest', 'random':'test_random'}[stage]
    seeds=get_splits()[split][:limit]
    pairs=protocol[{'initial':'initial_pairs', 'retest':'retest_pairs', 'random':'random_pairs'}[stage]]
    label=stage if limit is None else f'{stage}_pilot{limit}'
    output=DATA/'evaluation'/label; output.mkdir(parents=True,exist_ok=True)
    (output/'cases').mkdir(exist_ok=True)
    rows=[]; cases=[]; case_counts=Counter(); events=Counter(); risk_status=Counter()
    began=time.perf_counter()
    loaded_hashes={p:sha(MODEL_DIR/({'L':'L_final','L_initial':'L_initial','L_no_opponent':'L_no_opponent'}[p]+'.npz'))
                   for pair in pairs for p in pair if p.startswith('L')}
    write_json(output/'run_manifest.json',{'stage':label,'base_map_count':len(seeds),'split':split,'pairs':pairs,
               'model_sha256':loaded_hashes,'protocol_sha256':sha(DATA/'protocol.json'),
               'source':'automated_synthetic_policy_matches','human_records':0,
               'timing':'perf_counter wall time; shared warm process/cache, includes risk and features'})
    with gzip.open(output/'public_replays.jsonl.gz','wt',encoding='utf-8') as replay_file:
        for index,seed in enumerate(seeds):
            for pair_index,(p0,p1) in enumerate(pairs):
                for swap in (False,True):
                    for first in (0,1):
                        episode_id=f'{label}-m{index:03d}-p{pair_index}-s{int(swap)}-f{first}'
                        env=Arena(seed,first=first,swap=swap)
                        # Action randomness is public/reproducible and independent of the private map seed.
                        policy_rng=random.Random(81643 + index*100 + pair_index*10 + int(swap)*2 + first)
                        times=[0.,0.]; actions=[0,0]; previous=[None,None]; candidate=[]
                        while not env.done:
                            obs=env.observe(); actor=obs['turn']; policy=(p0,p1)[actor]
                            start=time.perf_counter(); decision=choose(obs,policy,policy_rng)
                            times[actor]+=time.perf_counter()-start; actions[actor]+=1
                            risk_status[decision['risk_status']]+=1
                            target=decision['target']
                            if target is not None and target==previous[1-actor] and all(obs['alive']):
                                candidate.append(('contested_target_candidate',obs['steps']))
                            if previous[actor] is not None and previous[actor]!=target and previous[actor] in obs['diamonds'] and target in obs['diamonds']:
                                candidate.append(('target_switch_candidate',obs['steps']))
                            if obs['steps']%5==0:
                                b1=choose(obs,'B1'); b2=choose(obs,'B2')
                                if b1['action']!=b2['action']:
                                    candidate.append(('B1_B2_disagreement',obs['steps']))
                            previous[actor]=target
                            env.step(decision['action'],decision)
                        replay=env.save_replay()
                        # Every evaluated action is deterministically replayed, not just selected wins.
                        verify_episode(seed,first,swap,replay)
                        replay_file.write(json.dumps({'episode_id':episode_id,'map_id':f'{split}-{index:03d}',
                                                     'policies':[p0,p1],'replay':replay},separators=(',',':'))+'\n')
                        if p1.startswith('L') and env.winner==0:
                            candidate.append(('learning_policy_failure',max(0,env.steps-1)))
                        if env.reason=='max_steps':
                            candidate.append(('timeout',max(0,env.steps-1)))
                        if env.winner=='draw':
                            candidate.append(('draw',max(0,env.steps-1)))
                        for kind,step in candidate:
                            events[kind]+=1
                            if case_counts[kind]<3:
                                filename=f'{kind}_{case_counts[kind]+1}.json'
                                case={'kind':kind,'episode_id':episode_id,'step':step,'policies':[p0,p1],
                                      'detection':'automated candidate; not a claim of intent or human review',
                                      'replay':replay}
                                write_json(output/'cases'/filename,case)
                                cases.append({'kind':kind,'episode_id':episode_id,'step':step,'file':'cases/'+filename})
                                case_counts[kind]+=1
                        rows.append({'episode_id':episode_id,'map_id':f'{split}-{index:03d}',
                                     'policy0':p0,'policy1':p1,'swap':int(swap),'first':first,
                                     'winner':env.winner,'score0':env.scores[0],'score1':env.scores[1],
                                     'eliminated0':int(not env.alive[0]),'eliminated1':int(not env.alive[1]),
                                     'steps':env.steps,'reason':env.reason,'actions0':actions[0],'actions1':actions[1],
                                     'time0':times[0],'time1':times[1]})
            if index%10==9:
                print(f'{label}: {index+1}/{len(seeds)} maps, {len(rows)} games, elapsed {time.perf_counter()-began:.1f}s',flush=True)
    with (output/'games.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result={'stage':label,'games':len(rows),'independent_base_maps':len(seeds),'elapsed_seconds':time.perf_counter()-began,
            'all_replays_verified':len(rows),'risk_status_counts':dict(risk_status),'event_counts':dict(events),
            'behavior_definitions':{
                'contested_target_candidate':'both alive; current emitted/inferred target equals other actor last target and remains available',
                'target_switch_candidate':'same actor changes target while both old and new targets remain available; learned target is inferred',
                'B1_B2_disagreement':'different B1/B2 actions on same public observation, checked every fifth step',
                'learning_policy_failure':'learning policy loses by final diamonds, regardless of survival',
                'target_interpretation':'B1/B2 targets are explicit; L targets are post-hoc path attribution, not a learned target head'},
            'pairs':grouped_summary(rows),'cases':cases,'human_records':0,'model_sha256':loaded_hashes}
    write_json(output/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['initial','retest','random'])
    parser.add_argument('--limit',type=int)
    args=parser.parse_args(); run(args.stage,args.limit)
