'use strict';
// Validate untrusted replay data before changing the displayed session.
function validateReplay(data) {
  const bad=()=>{throw Error('回放状态无效：请选择本程序导出的公开回放。')};
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
  }
  return true;
}
if(typeof module!=='undefined')module.exports={validateReplay};
