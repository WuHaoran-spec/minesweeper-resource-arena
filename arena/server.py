"""Loopback-only HTTP UI; policy callers receive public observations only."""
from __future__ import annotations
import argparse, copy, json, mimetypes, random, secrets, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from .env import Arena
from .classic import Classic

ROOT = Path(__file__).resolve().parent.parent

class Session:
    def __init__(self):
        self.lock = threading.RLock()
        self.snapshots = {}
        self.reset({})

    def reset(self, data):
        if not isinstance(data, dict):
            raise ValueError('请求必须为JSON对象')
        mode = data.get('mode', 'arena')
        if mode not in ('arena', 'classic', 'classic_race'):
            raise ValueError('未知模式')
        controllers = data.get('controllers', ['R1', 'C1'] if mode=='classic_race' else ['human', 'B2'])
        allowed = ({'human','R0','R1','C1'} if mode == 'classic_race' else
                   {'human', 'B0', 'B1', 'B2', 'B3', 'L3', 'L3_initial', 'L', 'L_initial', 'L_no_opponent', 'L_v2', 'E'})
        if not isinstance(controllers, list) or len(controllers) != 2 or any(not isinstance(c, str) or c not in allowed for c in controllers):
            raise ValueError('未知策略')
        source = data.get('source', 'unverified_interaction')
        if source not in ('unverified_interaction', 'browser_automation'):
            raise ValueError('未知数据来源')
        # UI seed never appears in observations, logs, or policy arguments.
        seed = secrets.randbits(52)
        preset = data.get('preset', 'legacy')
        if preset not in ('legacy', 'beginner', 'intermediate', 'expert'):
            raise ValueError('未知难度')
        if mode == 'arena':
            env = Arena(seed=seed, preset='legacy' if preset == 'beginner' else preset,
                        scoring=data.get('scoring','diamonds'))
        elif mode == 'classic_race':
            from .classic_race import ClassicRace
            env = ClassicRace(seed=seed,preset='beginner' if preset=='legacy' else preset)
        else:
            env = Classic(seed=seed) if preset == 'legacy' else Classic(seed=seed, preset=preset)
        # Commit a new session only after every input and environment is valid.
        self.mode, self.controllers, self.source = mode, controllers.copy(), source
        self.env = env
        self.epoch = secrets.token_hex(12)
        self.paused = False
        self.last = None
        self.flags = set()
        self.rng = random.Random()
        self.frames = [self._ui_frame()]
        return self.state()

    def _ui_frame(self):
        """Annotations belong to the replay display, never policy observations."""
        frame = self.env.observe()
        frame['ui_flags'] = ([list(p) for p in sorted(self.flags)] if self.mode == 'arena'
                             else copy.deepcopy(frame.get('flags', [])))
        return frame

    def state(self):
        return {'observation': self.env.observe(), 'mode': self.mode,
                'controllers': self.controllers.copy(), 'paused': self.paused,
                'epoch': self.epoch, 'decision': copy.deepcopy(self.last),
                'flags': [list(p) for p in sorted(self.flags)], 'source': self.source}

    def mutate(self, route, data):
        if not isinstance(data, dict):
            raise ValueError('请求必须为JSON对象')
        if route == '/api/reset':
            return self.reset(data)
        if data.get('epoch') != self.epoch:
            raise ValueError('旧对局请求已拒绝，请刷新状态')
        obs = self.env.observe()
        if route == '/api/pause':
            self.paused = bool(data.get('paused', True))
            return self.state()
        if data.get('revision') != obs['revision']:
            raise ValueError('状态已更新，旧动作已拒绝')
        if route == '/api/snapshot':
            if self.mode != 'arena':
                raise ValueError('快照分支仅用于争夺模式')
            token = secrets.token_hex(8)
            self.snapshots[token] = {'environment': self.env.private_snapshot(),
                                     'controllers': self.controllers.copy(),
                                     'flags': self.flags.copy(), 'rng_state': self.rng.getstate()}
            if len(self.snapshots) > 100:
                del self.snapshots[next(iter(self.snapshots))]
            return {'snapshot_id': token, 'observation': obs}
        if route == '/api/restore':
            snapshot = copy.deepcopy(self.snapshots[data['snapshot_id']])
            self.env = Arena.from_snapshot(snapshot['environment'])
            self.mode = 'arena'
            self.controllers = snapshot['controllers']
            self.flags = snapshot['flags']
            self.rng.setstate(snapshot['rng_state'])
            self.epoch = secrets.token_hex(12)
            self.source = 'counterfactual_replay'
            self.frames = [self._ui_frame()]
            self.last = None
            self.paused = True
            return self.state()
        if route == '/api/flag':
            if self.mode == 'classic_race':raise ValueError('竞速棋盘仅记录揭格行动')
            r, c = int(data['r']), int(data['c'])
            if not (0 <= r < obs.get('height', obs['size']) and 0 <= c < obs['size']):
                raise ValueError('标记超出棋盘')
            if self.mode == 'classic':
                self.env.flag(r, c)
            else:
                if (r,c) in self.flags: self.flags.remove((r,c))
                else: self.flags.add((r,c))
            self.frames.append(self._ui_frame())
            return self.state()
        if route == '/api/chord':
            if self.mode != 'classic': raise ValueError('快速展开仅用于经典模式')
            if self.paused: raise ValueError('对局暂停中')
            self.env.chord(int(data['r']), int(data['c']))
        elif route == '/api/action':
            if self.paused: raise ValueError('对局暂停中')
            if self.mode == 'classic':
                self.env.reveal(int(data['r']), int(data['c']))
            elif self.mode == 'classic_race':
                if self.controllers[obs['turn']] != 'human':raise ValueError('当前为AI轮次')
                self.last = None
                self.env.step([int(data['r']),int(data['c'])],metadata={'policy':'human_input','source':self.source})
            else:
                if self.controllers[obs['turn']] != 'human':
                    raise ValueError('当前为AI轮次')
                self.last = None
                self.env.step(data['action'], metadata={'policy':'human_input', 'source':self.source})
        elif route == '/api/ai':
            if self.mode not in ('arena','classic_race') or obs['done']: raise ValueError('无可执行AI轮次')
            if self.paused and not data.get('single'): raise ValueError('对局暂停中')
            policy = self.controllers[obs['turn']]
            if policy == 'human': raise ValueError('当前为人工轮次')
            if self.mode == 'classic_race':
                from .classic_race import choose
                started=time.perf_counter()
                self.last=choose(copy.deepcopy(obs['boards'][obs['turn']]),policy=policy,rng=self.rng)
                self.last['actor']=obs['turn']
                seconds=time.perf_counter()-started
                self.env.step(self.last['cell'],metadata={**self.last,'source':self.source},computation_seconds=seconds)
            else:
                from .policies import choose
                self.last = choose(copy.deepcopy(obs), policy=policy, rng=self.rng)
                self.env.step(self.last['action'], metadata={**self.last, 'source': self.source,
                               'actor_source': ('search_trained_policy' if policy == 'E' else
                                                'learning_policy' if policy.startswith('L') else 'heuristic_policy')})
        else:
            raise ValueError('未知接口')
        self.frames.append(self._ui_frame())
        return self.state()

    def replay(self):
        return {'schema':'arena-ui-replay-1', 'mode':self.mode,
                'source':self.source, 'controllers':self.controllers.copy(),
                'frames':copy.deepcopy(self.frames),
                'records':self.env.save_replay() if self.mode in ('arena','classic_race') else None,
                'notice':'公开观察回放，不含隐藏雷图；不证明真实人类参与。'}

class Handler(BaseHTTPRequestHandler):
    session: Session

    def respond(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith('/api/'):
            with self.session.lock:
                if path == '/api/state': return self.respond(self.session.state())
                if path == '/api/export': return self.respond(self.session.replay())
                if path == '/api/health': return self.respond({'ok':True})
                return self.respond({'error':'not found'},404)
        names = {'/':'index.html', '/app.js':'app.js','/replay.js':'replay.js','/style.css':'style.css'}
        if path not in names: return self.respond({'error':'not found'},404)
        file = ROOT / 'web' / names[path]
        raw = file.read_bytes()
        self.send_response(200)
        self.send_header('Content-Type', (mimetypes.guess_type(file.name)[0] or 'text/plain')+'; charset=utf-8')
        self.send_header('Content-Length',str(len(raw)))
        self.send_header('Content-Security-Policy', "default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'")
        self.end_headers(); self.wfile.write(raw)

    def do_POST(self):
        origin = self.headers.get('Origin')
        expected = f'http://{self.headers.get("Host", "")}'
        if origin and origin != expected: return self.respond({'error':'foreign origin'},403)
        if self.headers.get('Host','').split(':')[0] not in ('127.0.0.1','localhost'):
            return self.respond({'error':'loopback only'},403)
        try:
            length = int(self.headers.get('Content-Length',0))
            if not 0 < length < 100000: raise ValueError('请求大小不合法')
            data = json.loads(self.rfile.read(length))
            with self.session.lock: result = self.session.mutate(urlsplit(self.path).path,data)
            self.respond(result)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            self.respond({'error':str(error)},400)
        except Exception as error:
            self.respond({'error':f'{type(error).__name__}: {error}'},500)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(); Handler.session=Session()
    Handler.session.reset({'preset':'intermediate','scoring':'survival-v1','controllers':['human','B3']})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Minesweeper Resource Arena: http://127.0.0.1:{args.port}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close()

if __name__ == '__main__': main()
