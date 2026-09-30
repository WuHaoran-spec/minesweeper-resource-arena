"""Authoritative resource-arena rules. Strategies receive only ``observe()``.

The private snapshot API is for trusted replay branching, never policy input.
"""
from __future__ import annotations

from copy import deepcopy
from collections import deque
import json
import random
from typing import Any
import uuid
from .config import arena_preset


RULE_VERSION = "arena-v1.0"
SIZE = 9
MINE_COUNT = 10
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "wait": (0, 0)}
STARTS = ((0, 0), (8, 8))


def neighbors(cell: tuple[int, int], size: int = SIZE, height: int | None = None):
    """Eight-neighbor cells for classic clue semantics."""
    r, c = cell
    height = size if height is None else height
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < height and 0 <= c + dc < size:
                yield r + dr, c + dc


def _cells(values):
    return [list(cell) for cell in sorted(values)]


def _metadata(value: dict | None) -> dict:
    """Keep JSON diagnostics, refusing private-truth field names in public logs."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("metadata must be a JSON object")
    forbidden = {"seed", "map_seed", "mines", "mine_map", "hidden_mines", "private_snapshot", "future_results"}
    def validate(obj):
        if isinstance(obj, dict):
            for key, item in obj.items():
                if not isinstance(key, str) or key.lower() in forbidden:
                    raise ValueError("private truth is not allowed in public metadata")
                validate(item)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                validate(item)
    validate(value)
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("metadata must contain finite JSON values") from exc


class Arena:
    """A fixed-map, two-player, alternating-turn resource game.

    ``swap`` swaps the player labels' start cells without changing the map.
    ``first`` is the identity (0 or 1) that takes the first action.
    """

    def __init__(self, seed: int = 0, first: int = 0, swap: bool = False,
                 max_steps: int | None = None, preset: str = 'legacy', scoring: str = 'diamonds'):
        config = arena_preset(preset)
        if scoring not in ('diamonds', 'survival-v1') or (preset == 'legacy' and scoring != 'diamonds'):
            raise ValueError('unsupported scoring for this preset')
        self.scoring = scoring
        max_steps = config['max_steps'] if max_steps is None else max_steps
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(first) is not int or first not in (0, 1):
            raise ValueError("first must be player 0 or 1")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self._seed, self._first, self._swap = seed, first, bool(swap)
        self._config, self.preset = config, preset
        self.width, self.height = config['width'], config['height']
        self.mine_count, self.resource_count = config['mine_count'], config['resource_count']
        self.rule_version = config['rule_version']
        if scoring == 'survival-v1':
            self.rule_version = 'arena-v3.0'
        self._corners = ((0, 0), (self.height - 1, self.width - 1))
        self._starts = list(reversed(self._corners)) if swap else list(self._corners)
        self.max_steps = max_steps
        self.game_id = uuid.uuid4().hex
        self.revision = 0
        rng = random.Random(seed)
        cells = [(r, c) for r in range(self.height) for c in range(self.width)]
        zone = config['start_zone_size']
        start_safe = {(r, c) for r, c in cells if (r < zone and c < zone) or
                      (r >= self.height - zone and c >= self.width - zone)}
        diamond_cells = [x for x in cells if x not in start_safe and all(
            abs(x[0] - s[0]) + abs(x[1] - s[1]) >= config['diamond_min_start_distance'] for s in self._corners)]
        self._initial_diamonds = set(rng.sample(diamond_cells, self.resource_count))
        self._known_safe = start_safe | self._initial_diamonds
        self._mines = set(rng.sample([x for x in cells if x not in self._known_safe], self.mine_count))
        self._diamonds = self._initial_diamonds.copy()
        self.positions = self._starts.copy()
        self.scores = [0, 0]
        self.alive = [True, True]
        self.lives = [config['initial_lives'], config['initial_lives']]
        self.turn = first
        self.steps = 0
        self.action_counts = [0, 0]
        self.recent_positions = [[list(p)] for p in self.positions]
        self.done = False
        self.winner: int | str | None = None
        self.reason: str | None = None
        self._revealed = {cell: self._number(cell) for cell in self._corners}
        if config['zero_expansion']:
            for cell in self._corners:
                self._reveal_safe(cell)
        self._history: list[dict[str, Any]] = []
        self._initial_observation = self.observe()

    def _number(self, cell):
        return -1 if cell in self._mines else sum(x in self._mines for x in neighbors(cell, self.width, self.height))

    def _reveal_safe(self, cell):
        """Public zero flood; resources score only on physical arrival."""
        queue, visited = deque([cell]), set()
        while queue:
            current = queue.popleft()
            if current in visited or current in self._mines:
                continue
            visited.add(current)
            number = self._number(current)
            self._revealed[current] = number
            self._known_safe.add(current)
            if number == 0 and self._config['zero_expansion']:
                queue.extend(neighbors(current, self.width, self.height))

    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        r, c = self.positions[self.turn]
        # Deliberately consult only PUBLIC revealed values, never _mines.
        return [name for name, (dr, dc) in ACTIONS.items()
                if 0 <= r + dr < self.height and 0 <= c + dc < self.width
                and self._revealed.get((r + dr, c + dc)) != -1]

    def observe(self) -> dict:
        observation = {
            "rule_version": self.rule_version,
            "game_id": self.game_id,
            "revision": self.revision,
            "size": self.width,
            "mine_count": self.mine_count,
            "turn": self.turn,
            "positions": [list(x) for x in self.positions],
            "scores": self.scores.copy(),
            "alive": self.alive.copy(),
            "diamonds": _cells(self._diamonds),
            "known_safe": _cells(self._known_safe),
            "revealed": [[r, c, number] for (r, c), number in sorted(self._revealed.items())],
            "steps": self.steps,
            "max_steps": self.max_steps,
            "done": self.done,
            "winner": self.winner,
            "reason": self.reason,
            "legal_actions": self.legal_actions(),
        }
        if self.preset != 'legacy':
            observation.update(width=self.width, height=self.height, resource_count=self.resource_count,
                               lives=self.lives.copy(), preset=self.preset)
        if self.scoring == 'survival-v1':
            observation.update(scoring=self.scoring, action_counts=self.action_counts.copy(),
                               recent_positions=deepcopy(self.recent_positions),
                               utility_scores=self.utility_scores())
        return observation

    def utility_scores(self):
        """Versioned terminal utility; wall-clock UI delays never alter score."""
        return [100*self.scores[i]+10*self.lives[i]-self.action_counts[i]/self.max_steps for i in (0, 1)]

    def step(self, action: str, metadata: dict | None = None) -> dict:
        if self.done:
            raise ValueError("game is finished; reset to start a new game")
        if action not in self.legal_actions():
            raise ValueError("illegal action; no game state was changed")
        diagnostic = _metadata(metadata)
        before = self.observe()
        actor = self.turn
        dr, dc = ACTIONS[action]
        r, c = self.positions[actor]
        cell = r + dr, c + dc
        self.positions[actor] = cell
        exploded = cell in self._mines
        collected = cell in self._diamonds
        self._revealed[cell] = self._number(cell)
        respawned = False
        if exploded:
            self.lives[actor] -= 1
            self.alive[actor] = self.lives[actor] > 0
            if self.alive[actor]:
                self.positions[actor] = self._starts[actor]
                respawned = True
        else:
            self._reveal_safe(cell)
        if collected:
            self._diamonds.remove(cell)
            self.scores[actor] += 1
        self.steps += 1
        self.action_counts[actor] += 1
        self.recent_positions[actor].append(list(self.positions[actor]))
        self.recent_positions[actor] = self.recent_positions[actor][-24:]
        self.revision += 1
        if not self._diamonds:
            self.reason = "diamonds_collected"
        elif not any(self.alive):
            self.reason = "no_survivors"
        elif self.steps >= self.max_steps:
            self.reason = "max_steps"
        if self.reason:
            self.done = True
            # Compare integer numerators to avoid float rounding changing ties.
            values = ([self.max_steps*(100*self.scores[i]+10*self.lives[i])-self.action_counts[i] for i in (0,1)]
                      if self.scoring == 'survival-v1' else self.scores)
            self.winner = "draw" if values[0] == values[1] else int(values[1] > values[0])
        else:
            other = 1 - actor
            self.turn = other if self.alive[other] else actor
        after = self.observe()
        feedback = {"collected": collected, "exploded": exploded, "score_delta": int(collected)}
        if self.preset != 'legacy':
            feedback.update(life_lost=int(exploded), respawned=respawned, lives_after=self.lives[actor])
        if self.scoring == 'survival-v1':
            feedback['utility_delta'] = 100*int(collected)-10*int(exploded)-1/self.max_steps
        self._history.append({
            "rule_version": self.rule_version,
            "game_id": self.game_id,
            "step": self.steps,
            "actor": actor,
            "observation": before,
            "action": action,
            "metadata": diagnostic,
            "feedback": feedback,
            "next_observation": after,
        })
        return after

    def reset(self, seed: int | None = None, first: int | None = None,
              swap: bool | None = None, max_steps: int | None = None,
              preset: str | None = None, scoring: str | None = None) -> dict:
        """Reset atomically; the new opaque game_id invalidates old requests."""
        limit = max_steps if max_steps is not None else (None if preset is not None and preset != self.preset else self.max_steps)
        self.__init__(self._seed if seed is None else seed,
                      self._first if first is None else first,
                      self._swap if swap is None else swap,
                      limit, self.preset if preset is None else preset,
                      self.scoring if scoring is None else scoring)
        return self.observe()

    def render(self) -> dict:
        """The web UI renders this public state; no hidden-debug channel exists."""
        return self.observe()

    def save_replay(self) -> dict:
        """Public, directly playable observations; neither seed nor private map."""
        return deepcopy({
            "schema": "arena-public-replay-v1",
            "rule_version": self.rule_version,
            "game_id": self.game_id,
            "initial_observation": self._initial_observation,
            "records": self._history,
            "final_observation": self.observe(),
            "result": {"done": self.done, "winner": self.winner, "reason": self.reason, "scores": self.scores.copy()},
        })

    def private_snapshot(self) -> dict:
        """Trusted local-only state. MUST NOT be passed to policies or web clients."""
        snapshot = {
            "schema": "arena-private-snapshot-v1", "rule_version": self.rule_version,
            "seed": self._seed, "first": self._first, "swap": self._swap,
            "mines": _cells(self._mines), "initial_diamonds": _cells(self._initial_diamonds),
            "observation": self.observe(), "history": self._history,
            "initial_observation": self._initial_observation,
        }
        if self.preset != 'legacy':
            snapshot.update(schema='arena-private-snapshot-v2', config=self._config.copy())
        if self.scoring == 'survival-v1':
            snapshot.update(schema='arena-private-snapshot-v3', scoring=self.scoring)
        return deepcopy(snapshot)

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "Arena":
        value = deepcopy(snapshot)
        legacy = value.get('schema') == 'arena-private-snapshot-v1' and value.get('rule_version') == RULE_VERSION
        modern = value.get('schema') == 'arena-private-snapshot-v2' and value.get('rule_version') == 'arena-v2.0'
        survival = value.get('schema') == 'arena-private-snapshot-v3' and value.get('rule_version') == 'arena-v3.0'
        if not (legacy or modern or survival):
            raise ValueError("unsupported private snapshot version")
        obs = value["observation"]
        preset = 'legacy' if legacy else value['config']['preset']
        if (modern or survival) and value['config'] != arena_preset(preset):
            raise ValueError('snapshot preset configuration does not match its rule version')
        env = cls(value["seed"], value["first"], value["swap"], obs["max_steps"], preset=preset,
                  scoring=value.get('scoring', 'diamonds'))
        env._mines = {tuple(x) for x in value["mines"]}
        env._initial_diamonds = {tuple(x) for x in value["initial_diamonds"]}
        env._diamonds = {tuple(x) for x in obs["diamonds"]}
        env._known_safe = {tuple(x) for x in obs["known_safe"]}
        env._revealed = {(r, c): n for r, c, n in obs["revealed"]}
        env.positions = [tuple(x) for x in obs["positions"]]
        for name in ("game_id", "revision", "turn", "steps", "done", "winner", "reason", "scores", "alive"):
            setattr(env, name, deepcopy(obs[name]))
        env.lives = obs['lives'].copy() if modern or survival else [int(alive) for alive in obs['alive']]
        if survival:
            env.action_counts = obs['action_counts'].copy()
            env.recent_positions = deepcopy(obs['recent_positions'])
        env._history = value["history"]
        env._initial_observation = value["initial_observation"]
        return env
