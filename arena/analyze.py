"""Regenerate comparison intervals and verified case descriptions from saved logs."""
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import numpy as np
from .learning import DATA, write_json
from .policies import choose


def paired_effect(rows, left, right, opponent):
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        if row['policy0']==opponent and row['policy1'] in (left,right):
            score=1. if row['winner']=='1' else .5 if row['winner']=='draw' else 0.
            groups[row['map_id']][row['policy1']].append(score)
    values=np.array([np.mean(x[left])-np.mean(x[right]) for x in groups.values()])
    rng=np.random.default_rng(8821)
    samples=values[rng.integers(0,len(values),(2000,len(values)))].mean(axis=1)
    return {'left':left,'right':right,'opponent':opponent,'independent_maps':len(values),
            'difference_win_plus_half_draw':float(values.mean()),
            'paired_map_bootstrap_95ci':np.quantile(samples,[.025,.975]).tolist()}


def analyze():
    base=DATA/'evaluation'/'retest'
    rows=list(csv.DictReader((base/'games.csv').open(encoding='utf-8')))
    effects=[paired_effect(rows,*pair,opponent) for pair in [('L','L_initial'),('L','L_no_opponent')]
             for opponent in ['B1','B2']]
    write_json(base/'paired_effects.json',effects)
    summaries=[]
    selected=set()
    out=base/'verified_cases';out.mkdir(exist_ok=True)
    with gzip.open(base/'public_replays.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            episode=json.loads(line);replay=episode['replay'];previous=[None,None]
            for rec in replay['records']:
                obs=rec['observation'];decision=rec['metadata'];actor=rec['actor'];target=decision['target']
                tags=[]
                if episode['policies']==['B1','B2']:
                    if target and target==previous[1-actor] and all(obs['alive']) and len(obs['diamonds'])>1:
                        tags.append(('explicit_contest','当前规则策略目标与另一方上次显式输出目标相同，双方仍存活。'))
                    if previous[actor] and previous[actor]!=target and previous[actor] in obs['diamonds'] and target in obs['diamonds']:
                        tags.append(('explicit_target_switch',f'同一行动者目标从{previous[actor]}转为{target}，两个钻石当时均未被领取。'))
                    b1=choose(obs,'B1');b2=choose(obs,'B2')
                    if b1['action']!=b2['action']:
                        tags.append(('algorithm_disagreement',f'同一公开观察：B1={b1["action"]}，B2={b2["action"]}。'))
                if episode['policies'][1]=='L' and actor==1 and rec['feedback']['exploded'] and replay['result']['winner']==0:
                    tags.append(('learned_failure','最终L输局，且该步踩雷；踩雷前风险只是后验/估计，不是读取真值。'))
                if decision['risk_status']=='approximate_cutoff':
                    tags.append(('approximate_risk','模型计数达到预算，明确标记近似；不声称精确概率。'))
                for kind,description in tags:
                    if kind in selected:
                        continue
                    selected.add(kind)
                    rr,cc=rec['next_observation']['positions'][actor]
                    item={'kind':kind,'episode_id':episode['episode_id'],'step':rec['step'],'actor':actor,
                          'positions':obs['positions'],'scores':obs['scores'],'diamonds':obs['diamonds'],
                          'action':rec['action'],'risk_of_destination':decision['risk'][rr*obs['size']+cc],
                          'risk_status':decision['risk_status'],'target':target,'previous_target':previous[actor],
                          'description':description,'feedback':rec['feedback'],'result':replay['result'],
                          'verification':'logical predicate checked against actual logged public state; full replay verified during evaluation',
                          'human_review':False,'public_replay_file':'verified_cases/'+kind+'.json'}
                    summaries.append(item)
                    write_json(out/(kind+'.json'),replay)
                previous[actor]=target
    write_json(base/'verified_cases.json',summaries)
    print(json.dumps({'paired_effects':effects,'verified_case_count':len(summaries)},ensure_ascii=False,indent=2))


if __name__=='__main__':analyze()
