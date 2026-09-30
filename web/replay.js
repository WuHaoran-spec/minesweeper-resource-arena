'use strict';
// Validate untrusted replay data before changing the displayed session.
function validateReplay(data) {
  const bad=()=>{throw Error('回放状态无效：请选择本程序导出的公开回放。')};
  if(data?.mode==='classic_race'){
    if(data.schema!=='arena-ui-replay-1'||!Array.isArray(data.controllers)||data.controllers.length!==2||!data.controllers.every(x=>typeof x==='string')||typeof data.source!=='string'||!Array.isArray(data.frames)||!data.frames.length||data.frames.length>5000)bad();
    let shape=null;
    for(const f of data.frames){
      if(!f||f.rule_version!=='classic-race-v3'||!Array.isArray(f.boards)||f.boards.length!==2||![0,1].includes(f.turn)||typeof f.done!=='boolean'||!Number.isInteger(f.revision)||f.revision<0||!Array.isArray(f.results)||f.results.length!==2||!Array.isArray(f.active_reveals)||f.active_reveals.length!==2||!f.active_reveals.every(x=>Number.isInteger(x)&&x>=0)||!Array.isArray(f.computation_seconds)||f.computation_seconds.length!==2||!f.computation_seconds.every(x=>Number.isFinite(x)&&x>=0))bad();
      if(f.revision!==f.active_reveals[0]+f.active_reveals[1])bad();
      const ranks=[];
      for(let i=0;i<2;i++){
        const b=f.boards[i],r=f.results[i];
        validateReplay({...data,mode:'classic',frames:[b]});
        if(b.width!==f.width||b.height!==f.height||b.size!==f.size||b.mine_count!==f.boards[0].mine_count||!r||typeof r.done!=='boolean'||typeof r.cleared!=='boolean'||!Number.isInteger(r.safe_revealed)||r.safe_revealed<0||r.safe_revealed>f.width*f.height-b.mine_count||r.active_reveals!==f.active_reveals[i])bad();
        const safe=b.revealed.filter(x=>x[2]>=0).length,mines=b.revealed.length-safe,total=f.width*f.height-b.mine_count;
        if(b.rule_version!=='classic-v2.0'||b.done!==r.done||safe!==r.safe_revealed||
           b.active_reveals!==f.active_reveals[i]||b.revision!==b.active_reveals+1||
           r.reason!==b.reason||r.cleared!==(b.winner==='win'))bad();
        // Only the actually hit mine can be present. Classic's terminal full-mine display is not a race observation.
        if(!b.done&&(mines!==0||b.winner!==null||b.reason!==null||safe>=total))bad();
        if(b.done&&(b.winner==='win'?(mines!==0||safe!==total||b.reason!=='all_safe_revealed'):
                       (b.winner!=='loss'||mines!==1||safe>=total||b.reason!=='mine')))bad();
        ranks.push([Number(r.cleared),safe,-r.active_reveals]);
      }
      if(f.done!==f.boards.every(b=>b.done)||(!f.done&&(f.winner!==null||f.boards[f.turn].done)))bad();
      if(f.done){let expected='draw';for(let k=0;k<3;k++)if(ranks[0][k]!==ranks[1][k]){expected=Number(ranks[1][k]>ranks[0][k]);break}if(f.winner!==expected)bad();}
      const current=[f.width,f.height,f.boards[0].mine_count].join('/');if(shape!==null&&shape!==current)bad();shape=current;
    }
    return true;
  }
  if(!data||data.schema!=='arena-ui-replay-1'||!['arena','classic'].includes(data.mode)||
     !Array.isArray(data.controllers)||data.controllers.length!==2||!data.controllers.every(x=>typeof x==='string')||
     typeof data.source!=='string'||!Array.isArray(data.frames)||!data.frames.length||data.frames.length>5000)bad();
  let initialShape=null;
  for(const f of data.frames){
    if(!f||!Number.isInteger(f.size))bad();
    const w=f.width||f.size,h=f.height||f.size,n=w*h,total=f.resource_count||3;
    if(!Number.isInteger(w)||!Number.isInteger(h)||w<2||w>40||h<2||h>30||f.size!==w||
       !Number.isInteger(f.mine_count)||f.mine_count<1||f.mine_count>=n||typeof f.done!=='boolean')bad();
    const shape=[w,h,f.mine_count,f.rule_version||'legacy'].join('/');
    if(initialShape!==null&&shape!==initialShape)bad();initialShape=shape;
    const pair=p=>Array.isArray(p)&&p.length===2&&Number.isInteger(p[0])&&Number.isInteger(p[1])&&p[0]>=0&&p[0]<h&&p[1]>=0&&p[1]<w;
    const pairs=p=>Array.isArray(p)&&p.length<=n&&p.every(pair);
    if(!Array.isArray(f.revealed)||f.revealed.length>n||f.revealed.some(x=>!Array.isArray(x)||x.length!==3||!pair(x.slice(0,2))||!Number.isInteger(x[2])||x[2]<-1||x[2]>8)||
       new Set(f.revealed.map(x=>x.slice(0,2).join(','))).size!==f.revealed.length||
       (f.flags!==undefined&&!pairs(f.flags))||(f.ui_flags!==undefined&&!pairs(f.ui_flags)))bad();
    if(data.mode==='arena'&&(!pairs(f.positions)||f.positions.length!==2||
       !Array.isArray(f.alive)||f.alive.length!==2||!f.alive.every(x=>typeof x==='boolean')||
       !Number.isInteger(total)||total<1||total>20||
       !Array.isArray(f.scores)||f.scores.length!==2||!f.scores.every(x=>Number.isInteger(x)&&x>=0&&x<=total)||
       f.scores[0]+f.scores[1]>total||!pairs(f.diamonds)||f.diamonds.length>total||![0,1].includes(f.turn)||
       !Number.isInteger(f.steps)||f.steps<0||!Number.isInteger(f.max_steps)||f.max_steps<1||f.steps>f.max_steps||
       (f.lives!==undefined&&(!Array.isArray(f.lives)||f.lives.length!==2||!f.lives.every(x=>Number.isInteger(x)&&x>=0&&x<=3)))))bad();
    if(f.rule_version==='arena-v3.0'||f.scoring==='survival-v1'){
      if(data.mode!=='arena'||f.rule_version!=='arena-v3.0'||f.scoring!=='survival-v1'||!Array.isArray(f.lives)||f.lives.length!==2||!f.lives.every(x=>Number.isInteger(x)&&x>=0&&x<=3)||!Array.isArray(f.action_counts)||f.action_counts.length!==2||!f.action_counts.every(x=>Number.isInteger(x)&&x>=0)||f.action_counts[0]+f.action_counts[1]!==f.steps||!Array.isArray(f.utility_scores)||f.utility_scores.length!==2||!f.utility_scores.every(Number.isFinite)||!Array.isArray(f.recent_positions)||f.recent_positions.length!==2||!f.recent_positions.every(x=>pairs(x)&&x.length>0&&x.length<=24))bad();
      for(let i=0;i<2;i++)if(Math.abs(f.utility_scores[i]-(100*f.scores[i]+10*f.lives[i]-f.action_counts[i]/f.max_steps))>1e-6)bad();
    }
  }
  return true;
}
if(typeof module!=='undefined')module.exports={validateReplay};
