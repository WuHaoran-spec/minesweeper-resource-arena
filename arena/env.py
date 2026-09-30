"""Authoritative resource-arena rules. Strategies receive only ``observe()``.

The private snapshot API is for trusted replay branching, never policy input.
"""
from __future__ import annotations

from copy import deepcopy
import json
import random
from typing import Any
import uuid


RULE_VERSION = "arena-v1.0"
SIZE = 9
MINE_COUNT = 10
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1), "wait": (0, 0)}
STARTS = ((0, 0), (8, 8))


def neighbors(cell: tuple[int, int], size: int = SIZE):
    """Eight-neighbor cells for classic clue semantics."""
    r, c = cell
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if (dr or dc) and 0 <= r + dr < size and 0 <= c + dc < size:
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

    def __init__(self, seed: int, first: int = 0, swap: bool = False, max_steps: int = 200):
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(first) is not int or first not in (0, 1):
            raise ValueError("first must be player 0 or 1")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("max_steps must be a positive integer")
        self._seed, self._first, self._swap = seed, first, bool(swap)
        self.max_steps = max_steps
        self.game_id = uuid.uuid4().hex
        self.revision = 0
        rng = random.Random(seed)
        cells = [(r, c) for r in range(SIZE) for c in range(SIZE)]
        self._initial_diamonds = set(rng.sample([x for x in cells if x not in STARTS], 3))
        self._known_safe = set(STARTS) | self._initial_diamonds
        self._mines = set(rng.sample([x for x in cells if x not in self._known_safe], MINE_COUNT))
        self._diamonds = self._initial_diamonds.copy()
        self.positions = list(reversed(STARTS)) if swap else list(STARTS)
        self.scores = [0, 0]
        self.alive = [True, True]
        self.turn = first
        self.steps = 0
        self.done = False
        self.winner: int | str | None = None
        self.reason: str | None = None
        self._revealed = {cell: self._number(cell) for cell in STARTS}
        self._history: list[dict[str, Any]] = []
        self._initial_observation = self.observe()

    def _number(self, cell):
        return -1 if cell in self._mines else sum(x in self._mines for x in neighbors(cell))

    def legal_actions(self) -> list[str]:
        if self.done:
            return []
        r, c = self.positions[self.turn]
        # Deliberately consult only PUBLIC revealed values, never _mines.
        return [name for name, (dr, dc) in ACTIONS.items()
                if 0 <= r + dr < SIZE and 0 <= c + dc < SIZE
                and self._revealed.get((r + dr, c + dc)) != -1]

    def observe(self) -> dict:
        return {
            "rule_version": RULE_VERSION,
            "game_id": self.game_id,
            "revision": self.revision,
            "size": SIZE,
            "mine_count": MINE_COUNT,
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
        if exploded:
            self.alive[actor] = False
        else:
            self._known_safe.add(cell)
        if collected:
            self._diamonds.remove(cell)
            self.scores[actor] += 1
        self.steps += 1
        self.revision += 1
        if not self._diamonds:
            self.reason = "diamonds_collected"
        elif not any(self.alive):
            self.reason = "no_survivors"
        elif self.steps >= self.max_steps:
            self.reason = "max_steps"
        if self.reason:
            self.done = True
            self.winner = "draw" if self.scores[0] == self.scores[1] else int(self.scores[1] > self.scores[0])
        else:
            other = 1 - actor
            self.turn = other if self.alive[other] else actor
        after = self.observe()
        self._history.append({
            "rule_version": RULE_VERSION,
            "game_id": self.game_id,
            "step": self.steps,
            "actor": actor,
            "observation": before,
            "action": action,
            "metadata": diagnostic,
            "feedback": {"collected": collected, "exploded": exploded, "score_delta": int(collected)},
            "next_observation": after,
        })
        return after

    def reset(self, seed: int | None = None, first: int | None = None,
              swap: bool | None = None, max_steps: int | None = None) -> dict:
        """Reset atomically; the new opaque game_id invalidates old requests."""
        self.__init__(self._seed if seed is None else seed,
                      self._first if first is None else first,
                      self._swap if swap is None else swap,
                      self.max_steps if max_steps is None else max_steps)
        return self.observe()

    def render(self) -> dict:
        """The web UI renders this public state; no hidden-debug channel exists."""
        return self.observe()

    def save_replay(self) -> dict:
        """Public, directly playable observations; neither seed nor private map."""
        return deepcopy({
            "schema": "arena-public-replay-v1",
            "rule_version": RULE_VERSION,
            "game_id": self.game_id,
            "initial_observation": self._initial_observation,
            "records": self._history,
            "final_observation": self.observe(),
            "result": {"done": self.done, "winner": self.winner, "reason": self.reason, "scores": self.scores.copy()},
        })

    def private_snapshot(self) -> dict:
        """Trusted local-only state. MUST NOT be passed to policies or web clients."""
        return deepcopy({
            "schema": "arena-private-snapshot-v1", "rule_version": RULE_VERSION,
            "seed": self._seed, "first": self._first, "swap": self._swap,
            "mines": _cells(self._mines), "initial_diamonds": _cells(self._initial_diamonds),
            "observation": self.observe(), "history": self._history,
            "initial_observation": self._initial_observation,
        })

    @classmethod
    def from_snapshot(cls, snapshot: dict) -> "Arena":
        value = deepcopy(snapshot)
        if value.get("schema") != "arena-private-snapshot-v1" or value.get("rule_version") != RULE_VERSION:
            raise ValueError("unsupported private snapshot version")
        obs = value["observation"]
        env = cls(value["seed"], value["first"], value["swap"], obs["max_steps"])
        env._mines = {tuple(x) for x in value["mines"]}
        env._initial_diamonds = {tuple(x) for x in value["initial_diamonds"]}
        env._diamonds = {tuple(x) for x in obs["diamonds"]}
        env._known_safe = {tuple(x) for x in obs["known_safe"]}
        env._revealed = {(r, c): n for r, c, n in obs["revealed"]}
        env.positions = [tuple(x) for x in obs["positions"]]
        for name in ("game_id", "revision", "turn", "steps", "done", "winner", "reason", "scores", "alive"):
            setattr(env, name, deepcopy(obs[name]))
        env._history = value["history"]
        env._initial_observation = value["initial_observation"]
        return env
