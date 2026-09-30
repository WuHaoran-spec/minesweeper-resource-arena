"""Secondary classic benchmark: equal maps, separate information, fair turns.

ClassicRace is not the project's shared-resource arena. Both boards start with
the same protected center reveal. Terminal full-mine displays are scrubbed.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import uuid
import numpy as np

from .classic import Classic
from .env import neighbors, _metadata
from .risk import infer

VERSION = 'classic-race-v3'
FEATURE_NAMES = ['mine_marginal', 'safe_marginal', 'marginal_variance',
                 'unknown_neighbors', 'revealed_neighbors', 'neighbor_clue_mean',
                 'neighbor_degree', 'center_distance', 'revealed_fraction', 'bias']
_CACHE = {}


class ClassicRace:
    def __init__(self, seed=0, preset='beginner', first=0):
        if first not in (0, 1) or type(first) is not int: raise ValueError('Invalid first player')
        if preset not in ('beginner', 'intermediate', 'expert'): raise ValueError('Named classic preset required')
        self._seed, self._first = seed, first
        self._boards = [Classic(seed, preset=preset), Classic(seed, preset=preset)]
        self.width, self.height, self.preset = self._boards[0].width, self._boards[0].height, preset
        self.automatic_start = [self.height//2, self.width//2]
        self.game_id = uuid.uuid4().hex; self.revision = 0; self.turn = first
        self.active_reveals = [0, 0]; self.computation_seconds = [0., 0.]
        self._visible = [set(), set()]; self._history = []
        for actor, board in enumerate(self._boards):
            board.reveal(*self.automatic_start)
            self._visible[actor].update(cell for cell, number in board._revealed.items() if number >= 0)
        self.done = all(board.done for board in self._boards)
        if not self.done and self._boards[self.turn].done: self.turn = 1-self.turn
        self._initial = self.observe()

    def _public_board(self, actor):
        board = self._boards[actor]; observation = board.observe()
        observation['revealed'] = [item for item in observation['revealed'] if tuple(item[:2]) in self._visible[actor]]
        observation['known_safe'] = [item[:2] for item in observation['revealed'] if item[2] >= 0]
        observation['legal_cells'] = [] if board.done else [[r, c] for r in range(self.height) for c in range(self.width)
                                                              if (r, c) not in self._visible[actor]]
        observation['active_reveals'] = self.active_reveals[actor]
        observation['automatic_start'] = self.automatic_start.copy()
        observation['terminal_mine_display_scrubbed'] = True
        return observation

    def _result(self, actor):
        board = self._boards[actor]
        count = sum(board._revealed[cell] >= 0 for cell in self._visible[actor])
        return {'done': board.done, 'cleared': board.winner == 'win', 'safe_revealed': count,
                'active_reveals': self.active_reveals[actor], 'reason': board.reason}

    def observe(self):
        results = [self._result(actor) for actor in (0, 1)]
        ranks = [(int(v['cleared']), v['safe_revealed'], -v['active_reveals']) for v in results]
        winner = ('draw' if ranks[0] == ranks[1] else int(ranks[1] > ranks[0])) if self.done else None
        return {'schema': VERSION, 'rule_version': VERSION, 'game_id': self.game_id, 'revision': self.revision,
                'preset': self.preset, 'width': self.width, 'height': self.height, 'size': self.width,
                'turn': self.turn, 'done': self.done, 'winner': winner, 'boards': [self._public_board(a) for a in (0, 1)],
                'results': results, 'active_reveals': self.active_reveals.copy(),
                'computation_seconds': self.computation_seconds.copy(), 'automatic_start': self.automatic_start.copy()}

    def current_observation(self):
        if self.done: raise ValueError('Race is finished')
        return self._public_board(self.turn)

    def step(self, cell, metadata=None, computation_seconds=0.):
        if self.done: raise ValueError('Race is finished')
        if (not isinstance(cell, (list, tuple)) or len(cell) != 2 or any(type(v) is not int for v in cell)
                or list(cell) not in self.current_observation()['legal_cells']):
            raise ValueError('Choose one publicly unknown cell')
        if not isinstance(computation_seconds, (int, float)) or not math.isfinite(computation_seconds) or computation_seconds < 0:
            raise ValueError('Invalid computation duration')
        diagnostic = _metadata(metadata)
        before = self.observe(); actor = self.turn; board = self._boards[actor]
        before_safe = self._result(actor)['safe_revealed']
        board.reveal(*cell); self._visible[actor].add(tuple(cell))
        self._visible[actor].update(pos for pos, value in board._revealed.items() if value >= 0)
        self.active_reveals[actor] += 1; self.computation_seconds[actor] += computation_seconds; self.revision += 1
        self.done = all(value.done for value in self._boards)
        if not self.done:
            other = 1-actor; self.turn = other if not self._boards[other].done else actor
        after = self.observe()
        feedback = {'exploded': board._revealed[tuple(cell)] == -1,
                    'new_safe_revealed': self._result(actor)['safe_revealed']-before_safe,
                    'board_done': board.done, 'board_cleared': board.winner == 'win'}
        self._history.append({'step': self.revision, 'actor': actor, 'observation': before,
                              'decision_observation': before['boards'][actor], 'action': list(cell),
                              'metadata': diagnostic, 'computation_seconds': computation_seconds,
                              'feedback': feedback, 'next_observation': after})
        return after

    def save_replay(self):
        return deepcopy({'schema': 'classic-race-public-replay-v3', 'rule_version': VERSION,
                         'initial_observation': self._initial, 'records': self._history,
                         'final_observation': self.observe()})


def prepare(observation):
    """Feature matrix for public unknown cells only. Flags are not mine facts."""
    if observation['done']: raise ValueError('Never feed terminal classic displays to a policy')
    if 'boards' in observation: raise ValueError('Pass only the current independent board observation')
    width, height = observation['width'], observation['height']
    revealed = {(r, c): n for r, c, n in observation['revealed']}
    safe = [[r, c] for (r, c), number in revealed.items() if number >= 0]
    # Rebuild this adapter from public classic fields; do not trust flags or environment attributes.
    risk_observation = {'size': width, 'width': width, 'height': height,
                        'mine_count': observation['mine_count'], 'revealed': observation['revealed'], 'known_safe': safe}
    risk, status, detail = infer(risk_observation)
    cells = [[r, c] for r in range(height) for c in range(width) if (r, c) not in revealed]
    if 'legal_cells' in observation and cells != observation['legal_cells']:
        raise ValueError('Legal cells must depend only on public revealed cells')
    features = []
    for r, c in cells:
        around = list(neighbors((r, c), width, height)); values = [revealed[p] for p in around if p in revealed and revealed[p] >= 0]
        p = risk[r*width+c]
        features.append([p, 1-p, p*(1-p), sum(pos not in revealed for pos in around)/8,
                         len(values)/8, (sum(values)/len(values)/8) if values else 0., len(around)/8,
                         (abs(r-height//2)+abs(c-width//2))/(width+height-2), len(safe)/(width*height), 1.])
    return np.asarray(features, dtype=np.float64), cells, risk, status, detail


def choose(observation, policy='R1', rng=None):
    if policy == 'C1':
        from .classic_learning import choose_model
        return choose_model(observation, rng=rng)
    if policy not in ('R0', 'R1'): raise ValueError('Unknown classic policy')
    X, cells, risk, status, detail = prepare(observation)
    rng = rng or random
    if policy == 'R0': index = rng.randrange(len(cells)); scores = None
    else:
        # Lexicographic risk first: information preference can never override lower risk.
        index = min(range(len(cells)), key=lambda i: (X[i, 0], -(X[i, 4]+.25*X[i, 3]), cells[i]))
        scores = {'mine_marginal': float(X[index, 0]), 'frontier_preference': float(X[index, 4]+.25*X[index, 3])}
    return {'cell': cells[index], 'action': cells[index], 'policy': policy, 'risk_status': status, 'risk': risk,
            'diagnostics': {'method': 'uniform unknown-cell sampling' if policy == 'R0' else 'minimum marginal, then public information frontier',
                            'selected_features': X[index].tolist(), 'feature_names': FEATURE_NAMES,
                            'selection': scores, 'risk': detail}}
