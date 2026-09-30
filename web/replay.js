'use strict';
// Validate untrusted replay data before changing the displayed session.
function validateReplay(data) {
  const bad = () => { throw Error('回放状态无效：请选择本程序导出的公开回放。'); };
  const pair = p => Array.isArray(p) && p.length === 2 && p.every(n => Number.isInteger(n) && n >= 0 && n < 9);
  const pairs = p => Array.isArray(p) && p.length <= 81 && p.every(pair);
  if (!data || data.schema !== 'arena-ui-replay-1' || !['arena','classic'].includes(data.mode) ||
      !Array.isArray(data.controllers) || data.controllers.length !== 2 || !data.controllers.every(x => typeof x === 'string') ||
      typeof data.source !== 'string' || !Array.isArray(data.frames) || data.frames.length < 1 || data.frames.length > 1000) bad();
  for (const f of data.frames) {
    if (!f || f.size !== 9 || f.mine_count !== 10 || typeof f.done !== 'boolean' ||
        !Array.isArray(f.revealed) || f.revealed.length > 81 || f.revealed.some(x => !Array.isArray(x) || x.length !== 3 || !pair(x.slice(0,2)) || !Number.isInteger(x[2]) || x[2] < -1 || x[2] > 8) ||
        new Set(f.revealed.map(x => x.slice(0,2).join(','))).size !== f.revealed.length ||
        (f.flags !== undefined && !pairs(f.flags)) || (f.ui_flags !== undefined && !pairs(f.ui_flags))) bad();
    if (data.mode === 'arena' && (!pairs(f.positions) || f.positions.length !== 2 ||
        !Array.isArray(f.alive) || f.alive.length !== 2 || !f.alive.every(x => typeof x === 'boolean') ||
        !Array.isArray(f.scores) || f.scores.length !== 2 || !f.scores.every(x => Number.isInteger(x) && x >= 0 && x <= 3) ||
        !pairs(f.diamonds) || f.diamonds.length > 3 || ![0,1].includes(f.turn) ||
        !Number.isInteger(f.steps) || f.steps < 0 || !Number.isInteger(f.max_steps) || f.steps > f.max_steps)) bad();
  }
  return true;
}
if (typeof module !== 'undefined') module.exports = {validateReplay};
