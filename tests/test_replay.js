const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const zlib = require('node:zlib');
const {validateReplay} = require('../web/replay.js');
let accepted = 0, rejected = 0;
function accept(value) { assert.equal(validateReplay(value),true); accepted++; }
function reject(value) { assert.throws(()=>validateReplay(value)); rejected++; }
function mutations(value, changes) { for (const change of changes) {const invalid=structuredClone(value);change(invalid);reject(invalid);} }

const valid = {schema:'arena-ui-replay-1',mode:'arena',controllers:['B1','B2'],source:'browser_automation',frames:[{size:9,mine_count:10,done:false,revealed:[[0,0,0]],positions:[[0,0],[8,8]],alive:[true,true],scores:[0,0],diamonds:[[4,4]],turn:0,steps:0,max_steps:200}]};
accept(valid);
mutations(valid,[x=>x.frames[0].positions=null,x=>x.frames[0].revealed.push([99,0,0]),x=>x.frames[0].scores=[5,0],x=>x.frames[0].revealed.push([0,0,2]),x=>x.mode='unknown',x=>x.frames[0].alive=[true],x=>x.frames[0].ui_flags=[[9,9]]]);
for(const invalid of [null,{},[],{...valid,frames:[]}])reject(invalid);

const expert=structuredClone(valid);Object.assign(expert.frames[0],{size:30,width:30,height:16,mine_count:99,rule_version:'arena-v2.0',resource_count:11,lives:[3,2],positions:[[0,0],[15,29]],max_steps:1600});
accept(expert);
mutations(expert,[x=>x.frames[0].positions[1]=[16,29],x=>x.frames[0].positions[1]=[15,30],x=>x.frames[0].lives=[4,3],x=>x.frames.push({...x.frames[0],height:15})]);

const survival=structuredClone(expert);Object.assign(survival.frames[0],{rule_version:'arena-v3.0',scoring:'survival-v1',action_counts:[0,0],utility_scores:[30,20],recent_positions:[[[0,0]],[[15,29]]]});
accept(survival);
mutations(survival,[x=>delete x.frames[0].lives,x=>x.frames[0].lives=[3],x=>x.frames[0].lives=[3,null],
 x=>x.frames[0].lives=[NaN,2],x=>x.frames[0].utility_scores=[31,20],x=>delete x.frames[0].action_counts,
 x=>x.frames[0].action_counts=[-1,1],x=>x.frames[0].recent_positions=[[],[[15,29]]],x=>x.frames[0].utility_scores=[NaN,20]]);

// Actual completed browser export, not a fabricated success fixture.
const browser=JSON.parse(fs.readFileSync(path.join(__dirname,'../docs/images/classic-race-v3-public.json'),'utf8'));
accept(browser);
assert.equal(browser.frames[0].done,false);
assert.equal(browser.frames.at(-1).done,true);
mutations(browser,[
 x=>x.frames[0].done=true,
 x=>x.frames[0].winner=1,
 x=>x.frames[0].results[0].safe_revealed++,
 x=>x.frames[0].boards[0].done=true,
 x=>x.frames[0].results[0].cleared=true,
 x=>{const b=x.frames[0].boards[0];b.revealed.push([...b.legal_cells[0],-1]);},
 x=>{const b=x.frames[0].boards[0];b.revealed.push([...b.legal_cells[0],-1],[...b.legal_cells[1],-1]);},
 x=>x.frames[0].boards[0].active_reveals=1,
 x=>x.frames[0].revision=1,
 x=>x.frames[0].boards[0].revision=99,
 x=>x.frames[0].results[0].reason='mine',
 x=>x.frames[0].computation_seconds[0]=-1,
 x=>delete x.frames[0].width,
 x=>{const f=x.frames.at(-1);f.winner=f.winner===0?1:0;},
 x=>x.frames.at(-1).done=false
]);

// Accept real loss histories too, so scrubbing checks do not accidentally require every board to win.
const collected=zlib.gunzipSync(fs.readFileSync(path.join(__dirname,'../data/classic_v3/train_public_replays.jsonl.gz'))).toString().trim().split('\n').map(JSON.parse);
for(const clearedBoards of [0,1]){
 const item=collected.find(value=>value.results.filter(result=>result.cleared).length===clearedBoards);
 assert.ok(item,'Expected a retained training failure case');
 accept({schema:'arena-ui-replay-1',mode:'classic_race',controllers:item.policies,source:'automated_training_feedback',
         frames:[item.replay.initial_observation,...item.replay.records.map(record=>record.next_observation)]});
}

// Exercise the real normalization function while replacing only UI DOM effects.
const source=fs.readFileSync(path.join(__dirname,'../web/app.js'),'utf8');
const functionSource=source.slice(source.indexOf('function enterReplay('),source.indexOf('\nfunction showFrame('));
assert.ok(functionSource.startsWith('function enterReplay('));
const context={validateReplay,$:()=>({}),showFrame:()=>{},replay:null};vm.createContext(context);vm.runInContext(functionSource,context);
const raw=structuredClone(browser.records);for(const record of raw.records)delete record.metadata.actor;
for(const value of [browser,raw,{episode_id:'public-dataset-wrapper',replay:raw}]){
 context.enterReplay(structuredClone(value));accepted++;
 assert.equal(context.replay.mode,'classic_race');
 const records=browser.records.records;
 for(let i=0;i<records.length;i++)assert.equal(context.replay.decisions[i+1].actor,records[i].actor);
 assert.equal(context.replay.frames.length,browser.frames.length);
}
console.log('Replay validation and normalization: '+accepted+' valid cases accepted; '+rejected+' malformed cases rejected.');
