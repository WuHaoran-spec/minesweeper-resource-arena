"""Small classic mode for regression: fixed mines, flags and zero expansion."""
from __future__ import annotations

from collections import deque
import random
import uuid

from .env import neighbors


class Classic:
    def __init__(self, seed: int = 0, size: int = 9, mine_count: int = 10):
        if type(size) is not int or size < 2 or type(mine_count) is not int or not 1 <= mine_count < size * size:
            raise ValueError("invalid board size or mine count")
        self.size, self.mine_count = size, mine_count
        self.game_id = uuid.uuid4().hex
        self.revision = 0
        cells = [(r, c) for r in range(size) for c in range(size)]
        self._mines = set(random.Random(seed).sample(cells, mine_count))
        self._revealed: dict[tuple[int, int], int] = {}
        self._flags: set[tuple[int, int]] = set()
        self.done, self.winner, self.reason = False, None, None

    def _cell(self, r, c):
        if type(r) is not int or type(c) is not int or not (0 <= r < self.size and 0 <= c < self.size):
            raise ValueError("cell is outside the board")
        if self.done:
            raise ValueError("game is finished")
        return r, c

    def _number(self, cell):
        return -1 if cell in self._mines else sum(n in self._mines for n in neighbors(cell, self.size))

    def observe(self):
        return {
            "rule_version": "classic-v1.0", "game_id": self.game_id, "revision": self.revision,
            "size": self.size, "mine_count": self.mine_count,
            "revealed": [[r, c, n] for (r, c), n in sorted(self._revealed.items())],
            "flags": [list(x) for x in sorted(self._flags)],
            "done": self.done, "winner": self.winner, "reason": self.reason,
            "first_click_safe": False,
        }

    def flag(self, r, c):
        cell = self._cell(r, c)
        if cell in self._revealed:
            raise ValueError("cannot flag a revealed cell")
        if cell in self._flags:
            self._flags.remove(cell)
        else:
            self._flags.add(cell)
        self.revision += 1
        return self.observe()

    def reveal(self, r, c):
        cell = self._cell(r, c)
        if cell in self._flags:
            raise ValueError("remove the flag before revealing this cell")
        if cell in self._revealed:
            return self.observe()
        self.revision += 1
        if cell in self._mines:
            self._revealed[cell] = -1
            self.done, self.winner, self.reason = True, "loss", "mine"
            return self.observe()
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            if current in self._revealed or current in self._flags:
                continue
            number = self._number(current)
            self._revealed[current] = number
            if number == 0:
                queue.extend(n for n in neighbors(current, self.size) if n not in self._revealed and n not in self._flags)
        if len(self._revealed) == self.size * self.size - self.mine_count:
            self.done, self.winner, self.reason = True, "win", "all_safe_revealed"
        return self.observe()

    def render(self):
        return self.observe()
