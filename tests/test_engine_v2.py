"""Rectangular challenge rules; private truth appears only in test fixtures."""
from copy import deepcopy
import json
import random
import unittest
from pathlib import Path

from arena import Arena, Classic
from arena.config import arena_preset
from arena.env import neighbors


class ArenaChallengeTests(unittest.TestCase):
    def test_machine_config_matches_runtime_presets(self):
        path = Path(__file__).resolve().parents[1] / 'config' / 'rules_v2.json'
        public_config = json.loads(path.read_text(encoding='utf-8'))
        for name, values in public_config['presets'].items():
            runtime = arena_preset(name)
            for key, value in values.items():
                self.assertEqual(runtime[key], value)
            self.assertEqual(runtime['rule_version'], public_config['rule_version'])

    def test_presets_rectangular_counts_and_known_prior(self):
        for preset, width, height, mines, resources, limit in (
                ('intermediate', 16, 16, 40, 7, 800), ('expert', 30, 16, 99, 11, 1600)):
            for seed in range(10):
                env = Arena(seed, preset=preset)
                obs = env.observe()
                self.assertEqual((obs['size'], obs['width'], obs['height']), (width, width, height))
                self.assertEqual((len(env._mines), len(obs['diamonds'])), (mines, resources))
                self.assertEqual(obs['lives'], [3, 3])
                self.assertEqual(obs['max_steps'], limit)
                start_zone = {(r, c) for r in range(height) for c in range(width)
                              if (r < 3 and c < 3) or (r >= height-3 and c >= width-3)}
                self.assertFalse(start_zone & env._mines)
                self.assertFalse(start_zone & env._diamonds)
                self.assertTrue(start_zone <= env._known_safe)
                for r, c in env._diamonds:
                    self.assertGreaterEqual(r + c, 6)
                    self.assertGreaterEqual(height-1-r + width-1-c, 6)
                self.assertGreater(len(obs['revealed']), 2)
                self.assertEqual(obs['scores'], [0, 0])  # Zero flood never collects.
                self.assertFalse(env._mines & env._known_safe)
                for r, c, number in obs['revealed']:
                    expected = sum(abs(r-mr) <= 1 and abs(c-mc) <= 1 for mr, mc in env._mines)
                    self.assertEqual(number, expected)

    def test_rectangular_corners_and_edges_never_wrap(self):
        self.assertEqual(set(neighbors((15, 29), 30, 16)), {(14, 28), (14, 29), (15, 28)})
        self.assertEqual(set(neighbors((0, 29), 30, 16)), {(0, 28), (1, 28), (1, 29)})
        env = Arena(40, first=1, preset='expert')
        self.assertEqual(set(env.legal_actions()), {'up', 'left', 'wait'})
        with self.assertRaises(ValueError):
            env.step('right')

    def test_respawn_life_loss_and_elimination_keep_score_and_map(self):
        env = Arena(40, swap=True, preset='expert')
        initial_mines = env._mines.copy()
        env.scores[0] = 2
        for lives_after in (2, 1, 0):
            mine = next((r, c) for r, c in sorted(env._mines)
                        if c > 0 and (r, c-1) not in env._mines and (r, c) not in env._revealed)
            env.turn, env.positions[0] = 0, (mine[0], mine[1]-1)
            obs = env.step('right')
            self.assertEqual(obs['lives'][0], lives_after)
            self.assertEqual(obs['alive'][0], lives_after > 0)
            self.assertEqual(obs['scores'][0], 2)
            self.assertEqual(obs['turn'], 1)
            self.assertEqual(env._mines, initial_mines)
            self.assertEqual(env._revealed[mine], -1)
            self.assertEqual(env.positions[0], (15, 29) if lives_after else mine)
            self.assertFalse(obs['done'])
        env.step('wait')
        self.assertEqual(env.turn, 1)

    def test_respawn_when_opponent_dead_does_not_skip_live_actor(self):
        env = Arena(22, preset='intermediate')
        env.alive[1], env.lives[1] = False, 0
        mine = next((r, c) for r, c in sorted(env._mines) if c > 0 and (r, c-1) not in env._mines)
        env.positions[0] = (mine[0], mine[1]-1)
        env.step('right')
        self.assertEqual(env.turn, 0)
        self.assertEqual(env.lives, [2, 0])
        self.assertEqual(env.positions[0], (0, 0))
        self.assertIn('wait', env.legal_actions())

    def test_hidden_mine_differences_never_change_public_mask(self):
        a = Arena(30, preset='expert')
        # Deliberate sparse-observation fixture with two valid hidden variants.
        a._revealed = {(0, 0): 0, (15, 29): 0}
        a._known_safe = {cell for cell in a._known_safe
                         if (cell[0] < 3 and cell[1] < 3) or (cell[0] >= 13 and cell[1] >= 27)} | a._diamonds
        a._diamonds.discard((4, 5)); a._diamonds.discard((5, 4))
        a._known_safe.discard((4, 5)); a._known_safe.discard((5, 4))
        a.positions[0] = (4, 4)
        a._mines -= {(4, 4), (4, 5), (5, 4)}
        a._mines.add((4, 5))
        b = deepcopy(a)
        b._mines.remove((4, 5)); b._mines.add((5, 4))
        self.assertEqual(a.observe(), b.observe())
        self.assertIn('right', a.legal_actions())
        a.step('right'); b.step('right')
        self.assertEqual(a.lives[0], 2)
        self.assertEqual(b.lives[0], 3)
        for obs in (a.observe(), b.observe()):
            self.assertFalse({'seed', 'mines', 'map_seed', 'config'} & set(obs))

    def test_v2_snapshot_and_public_replay_restore_full_configuration(self):
        env = Arena(19, first=1, swap=True, preset='expert', max_steps=65)
        rng = random.Random(42)
        for _ in range(20):
            env.step(rng.choice(env.legal_actions()))
        snapshot = json.loads(json.dumps(env.private_snapshot()))
        self.assertEqual(snapshot['schema'], 'arena-private-snapshot-v2')
        self.assertEqual(snapshot['config']['height'], 16)
        restored = Arena.from_snapshot(snapshot)
        while not env.done:
            action = rng.choice(env.legal_actions())
            self.assertEqual(env.step(action), restored.step(action))
        self.assertEqual(env.save_replay(), restored.save_replay())
        self.assertNotIn('mines', env.save_replay())
        self.assertNotIn('seed', env.save_replay())
        self.assertEqual(env.reason, 'max_steps')

    def test_swap_preserves_base_map_and_reset_switches_preset_defaults(self):
        a, b = Arena(55, preset='expert'), Arena(55, first=1, swap=True, preset='expert')
        self.assertEqual(a._mines, b._mines)
        self.assertEqual(a._diamonds, b._diamonds)
        a.reset(preset='intermediate')
        self.assertEqual((a.width, a.height, a.max_steps), (16, 16, 800))
        a.reset(preset='legacy')
        self.assertEqual(a.max_steps, 200)
        self.assertNotIn('lives', a.observe())
        self.assertEqual(a.observe()['rule_version'], 'arena-v1.0')
        cfg = arena_preset('expert'); cfg['width'] = 3
        self.assertEqual(arena_preset('expert')['width'], 30)


class ClassicChallengeTests(unittest.TestCase):
    def test_deferred_first_click_and_neighbor_safety_on_rectangle(self):
        for seed in range(12):
            for cell in ((0, 0), (0, 29), (15, 0), (15, 29), (7, 14)):
                env = Classic(seed, preset='expert')
                self.assertFalse(env._mines)
                self.assertFalse(env.observe()['mines_placed'])
                obs = env.reveal(*cell)
                self.assertEqual(len(env._mines), 99)
                self.assertFalse(({cell} | set(neighbors(cell, 30, 16))) & env._mines)
                self.assertEqual(env._revealed[cell], 0)
                self.assertTrue(obs['first_click_neighbor_safe'])
                self.assertEqual((obs['width'], obs['height']), (30, 16))
                self.assertTrue(all(0 <= r < 16 and 0 <= c < 30 for r, c, n in obs['revealed']))

    def _fixture(self):
        env = Classic(0, width=5, height=4, mine_count=2)
        env._initialized = True
        env._mines = {(1, 1), (3, 4)}
        env._revealed = {(0, 0): 1}
        return env

    def test_chord_correct_flags_and_incorrect_flags(self):
        good = self._fixture()
        good.flag(1, 1)
        good.chord(0, 0)
        self.assertTrue({(0, 1), (1, 0)} <= set(good._revealed))
        self.assertFalse(good.done)
        bad = self._fixture()
        bad.flag(0, 1)
        self.assertFalse(any(n == -1 for r, c, n in bad.observe()['revealed']))
        bad.chord(0, 0)
        self.assertEqual(bad.winner, 'loss')
        self.assertEqual({(r, c) for r, c, n in bad.observe()['revealed'] if n == -1}, bad._mines)

    def test_chord_mismatch_is_atomic_and_flags_do_not_move_mines(self):
        env = self._fixture()
        before = env.observe()
        with self.assertRaises(ValueError): env.chord(0, 0)
        self.assertEqual(env.observe(), before)
        mines = env._mines.copy()
        env.flag(0, 1)
        self.assertEqual(env._mines, mines)
        with self.assertRaises(ValueError): env.reveal(0, 1)

    def test_rectangular_flood_win_and_legacy_terminal_compatibility(self):
        env = self._fixture()
        for r in range(4):
            for c in range(5):
                if not env.done and (r, c) not in env._mines:
                    env.reveal(r, c)
        self.assertEqual(env.winner, 'win')
        self.assertEqual(len(env._revealed), 18)
        legacy = Classic(3)
        legacy.reveal(*next(iter(legacy._mines)))
        self.assertEqual(len(legacy._revealed), 1)
        self.assertNotIn('height', legacy.observe())
        self.assertFalse(legacy.observe()['first_click_safe'])


if __name__ == '__main__':
    unittest.main()
