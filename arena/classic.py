"""Small classic mode for regression: fixed mines, flags and zero expansion."""
from __future__ import annotations

from collections import deque
import random
import uuid

from .env import neighbors
from .config import classic_preset


class Classic:
    def __init__(self, seed: int = 0, size: int = 9, mine_count: int = 10,
                 *, preset: str | None = None, width: int | None = None, height: int | None = None):
        self._legacy = preset in (None, 'legacy') and width is None and height is None
        if preset not in (None, 'legacy'):
            config = classic_preset(preset)
            if width is not None or height is not None:
                raise ValueError('choose either a preset or custom dimensions')
            width, height, mine_count = config['width'], config['height'], config['mine_count']
        elif width is None and height is None:
            width = height = size
        elif width is None or height is None:
            raise ValueError('custom classic boards require both width and height')
        if (type(width) is not int or type(height) is not int or min(width, height) < 2 or
                type(mine_count) is not int or not 1 <= mine_count < width * height):
            raise ValueError("invalid board size or mine count")
        if not self._legacy and mine_count > width * height - min(3, width) * min(3, height):
            raise ValueError('too many mines to guarantee a safe first cell and its neighbors')
        self.size, self.width, self.height, self.mine_count = width, width, height, mine_count
        self.preset = 'legacy' if self._legacy else (preset or 'custom')
        self._seed, self._initialized = seed, self._legacy
        self.game_id = uuid.uuid4().hex
        self.revision = 0
        cells = [(r, c) for r in range(height) for c in range(width)]
        self._mines = set(random.Random(seed).sample(cells, mine_count)) if self._legacy else set()
        self._revealed: dict[tuple[int, int], int] = {}
        self._flags: set[tuple[int, int]] = set()
        self.done, self.winner, self.reason = False, None, None

    def _cell(self, r, c):
        if type(r) is not int or type(c) is not int or not (0 <= r < self.height and 0 <= c < self.width):
            raise ValueError("cell is outside the board")
        if self.done:
            raise ValueError("game is finished")
        return r, c

    def _number(self, cell):
        return -1 if cell in self._mines else sum(n in self._mines for n in neighbors(cell, self.width, self.height))

    def observe(self):
        observation = {
            "rule_version": "classic-v1.0" if self._legacy else "classic-v2.0", "game_id": self.game_id, "revision": self.revision,
            "size": self.size, "mine_count": self.mine_count,
            "revealed": [[r, c, n] for (r, c), n in sorted(self._revealed.items())],
            "flags": [list(x) for x in sorted(self._flags)],
            "done": self.done, "winner": self.winner, "reason": self.reason,
            "first_click_safe": not self._legacy,
        }
        if not self._legacy:
            observation.update(width=self.width, height=self.height, preset=self.preset,
                               first_click_neighbor_safe=True, mines_placed=self._initialized)
        return observation

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
        if not self._initialized:
            protected = {cell} | set(neighbors(cell, self.width, self.height))
            candidates = [(rr, cc) for rr in range(self.height) for cc in range(self.width)
                          if (rr, cc) not in protected]
            self._mines = set(random.Random(self._seed).sample(candidates, self.mine_count))
            self._initialized = True
        self.revision += 1
        self._open(cell)
        return self.observe()

    def _open(self, cell):
        if cell in self._mines:
            self._revealed[cell] = -1
            self.done, self.winner, self.reason = True, "loss", "mine"
            if not self._legacy:
                # Classic terminal display only. Arena never exposes this map.
                self._revealed.update({mine: -1 for mine in self._mines})
            return
        queue = deque([cell])
        while queue:
            current = queue.popleft()
            if current in self._revealed or current in self._flags:
                continue
            number = self._number(current)
            self._revealed[current] = number
            if number == 0:
                queue.extend(n for n in neighbors(current, self.width, self.height) if n not in self._revealed and n not in self._flags)
        if len(self._revealed) == self.width * self.height - self.mine_count:
            self.done, self.winner, self.reason = True, "win", "all_safe_revealed"

    def chord(self, r, c):
        """Reveal unflagged neighbors only when the flag count matches a clue.

        Flags are assertions by the player, not verified mine facts. A wrong
        matching flag configuration can therefore lose the game.
        """
        cell = self._cell(r, c)
        if self._legacy:
            raise ValueError('chording requires a classic-v2 preset')
        clue = self._revealed.get(cell)
        if clue is None or clue <= 0:
            raise ValueError('chording requires a revealed positive clue')
        around = list(neighbors(cell, self.width, self.height))
        if sum(n in self._flags for n in around) != clue:
            raise ValueError('neighbor flag count must match the clue')
        targets = [n for n in around if n not in self._flags and n not in self._revealed]
        if not targets:
            return self.observe()
        self.revision += 1
        for target in targets:
            if target not in self._revealed:
                self._open(target)
            if self.done:
                break
        return self.observe()

    def render(self):
        return self.observe()
