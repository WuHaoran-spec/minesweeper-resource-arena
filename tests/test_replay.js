const assert = require('node:assert/strict');
const {validateReplay} = require('../web/replay.js');
const valid = {schema:'arena-ui-replay-1',mode:'arena',controllers:['B1','B2'],source:'browser_automation',frames:[{size:9,mine_count:10,done:false,revealed:[[0,0,0]],positions:[[0,0],[8,8]],alive:[true,true],scores:[0,0],diamonds:[[4,4]],turn:0,steps:0,max_steps:200}]};
assert.equal(validateReplay(valid),true);
for(const mutate of [x=>x.frames[0].positions=null,x=>x.frames[0].revealed.push([99,0,0]),x=>x.frames[0].scores=[5,0],x=>x.frames[0].revealed.push([0,0,2]),x=>x.mode='unknown',x=>x.frames[0].alive=[true],x=>x.frames[0].ui_flags=[[9,9]]]){
 const invalid=structuredClone(valid);mutate(invalid);assert.throws(()=>validateReplay(invalid));
}
for(const invalid of [null,{},[],{...valid,frames:[]}])assert.throws(()=>validateReplay(invalid));
console.log('Replay validation: valid round-trip schema and 11 malformed cases passed.');
