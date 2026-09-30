"""Rule and information-boundary tests; private truth is used only by tests."""
import json
from pathlib import Path
import random
import unittest

from arena import Arena, Classic
from arena.env import ACTIONS, MINE_COUNT, RULE_VERSION, SIZE, STARTS, neighbors


def fixture(mines=None, diamonds=None, max_steps=200):
    """Deliberately constructed test map, never a training/evaluation sample."""
    env = Arena(0, max_steps=max_steps)
    env._mines = set(mines or ([(2, c) for c in range(9)] + [(0, 2)]))
    env._initial_diamonds = set(diamonds or [(0, 1), (4, 4), (4, 5)])
    env._diamonds = env._initial_diamonds.copy()
    env._known_safe = set(STARTS) | env._initial_diamonds
    env._revealed = {cell: env._number(cell) for cell in STARTS}
    env._initial_observation = env.observe()
    return env


class ArenaRulesTests(unittest.TestCase):
    def test_fixed_count_clues_and_safe_conditions(self):
        for seed in range(100):
            env = Arena(seed)
            self.assertEqual(len(env._mines), 10)
            self.assertEqual(len(env._diamonds), 3)
            self.assertFalse(env._mines & env._known_safe)
            self.assertFalse(set(STARTS) & env._diamonds)
            self.assertEqual(len(env._known_safe), 5)
            self.assertEqual(len(env.observe()["revealed"]), 2)
            for r in range(9):
                for c in range(9):
                    expected = -1 if (r, c) in env._mines else sum(
                        abs(r - mr) <= 1 and abs(c - mc) <= 1 and (r, c) != (mr, mc)
                        for mr, mc in env._mines)
                    self.assertEqual(env._number((r, c)), expected)

    def test_frozen_config_matches_authoritative_constants(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / "config" / "rules_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["rule_version"], RULE_VERSION)
        self.assertEqual(cfg["size"], SIZE)
        self.assertEqual(cfg["mine_count"], MINE_COUNT)
        self.assertEqual(cfg["starts"], [list(x) for x in STARTS])
        self.assertEqual(cfg["actions"], list(ACTIONS))
        self.assertEqual(cfg["default_max_steps"], Arena(0).max_steps)

    def test_map_is_identical_under_seat_and_first_swap(self):
        original = Arena(73)
        for first in (0, 1):
            for swap in (False, True):
                env = Arena(73, first=first, swap=swap)
                self.assertEqual(env._mines, original._mines)
                self.assertEqual(env._diamonds, original._diamonds)
                self.assertEqual(env.turn, first)
                self.assertEqual(env.positions[0], STARTS[int(swap)])

    def test_collect_only_on_arrival_and_only_once(self):
        env = fixture()
        self.assertEqual(env.scores, [0, 0])
        self.assertNotIn((0, 1), env._revealed)
        env.step("right", {"policy": "test", "risk": 0.0, "computation_status": "known_safe"})
        self.assertEqual(env.scores, [1, 0])
        self.assertNotIn((0, 1), env._diamonds)
        env.step("wait")
        env.step("left")
        env.step("wait")
        env.step("right")
        self.assertEqual(env.scores, [1, 0])

    def test_elimination_skips_player_and_preserves_score(self):
        env = fixture()
        initial_mines = env._mines.copy()
        env.step("right")
        env.step("wait")
        env.step("right")
        self.assertEqual(env.alive, [False, True])
        self.assertEqual(env.scores, [1, 0])
        self.assertFalse(env.done)
        self.assertEqual(env.turn, 1)
        self.assertEqual(env._revealed[(0, 2)], -1)
        env.step("wait")
        self.assertEqual(env.turn, 1)
        self.assertEqual(env._mines, initial_mines)

    def test_score_is_the_only_win_criterion(self):
        env = fixture(max_steps=4)
        for action in ("right", "wait", "right", "wait"):
            env.step(action)
        self.assertEqual(env.alive, [False, True])
        self.assertEqual(env.winner, 0)
        self.assertEqual(env.reason, "max_steps")

    def test_last_resource_ends_game(self):
        env = fixture()
        env._diamonds = {(0, 1)}
        env.scores = [1, 1]
        env.step("right")
        self.assertTrue(env.done)
        self.assertEqual(env.reason, "diamonds_collected")
        self.assertEqual(env.winner, 0)

    def test_both_eliminated_draw(self):
        env = fixture(mines=[(0, 1), (8, 7)] + [(3, c) for c in range(8)], diamonds=[(4, 4), (4, 5), (4, 6)])
        env.step("right")
        env.step("left")
        self.assertTrue(env.done)
        self.assertEqual(env.reason, "no_survivors")
        self.assertEqual(env.winner, "draw")

    def test_200_total_actions_including_wait(self):
        env = Arena(7)
        for _ in range(199):
            env.step("wait")
            self.assertFalse(env.done)
        env.step("wait")
        self.assertTrue(env.done)
        self.assertEqual(env.steps, 200)
        self.assertEqual(env.winner, "draw")
        self.assertEqual(env.legal_actions(), [])
        with self.assertRaises(ValueError):
            env.step("wait")

    def test_unknown_mines_do_not_change_legal_mask(self):
        diamonds = [(4, 4), (4, 5), (4, 6)]
        a = fixture(mines=[(0, 1)] + [(3, c) for c in range(9)], diamonds=diamonds)
        b = fixture(mines=[(1, 0)] + [(3, c) for c in range(9)], diamonds=diamonds)
        oa, ob = a.observe(), b.observe()
        oa.pop("game_id"); ob.pop("game_id")
        self.assertEqual(oa, ob)
        self.assertEqual(a.legal_actions(), ["down", "right", "wait"])
        a.step("right")
        b.step("right")
        self.assertFalse(a.alive[0])
        self.assertTrue(b.alive[0])

    def test_public_exploded_mine_is_excluded(self):
        env = fixture()
        env._revealed[(8, 7)] = -1
        env.turn = 1
        self.assertNotIn("left", env.legal_actions())
        self.assertIn("wait", env.legal_actions())

    def test_invalid_input_is_atomic(self):
        env = Arena(0)
        before = env.private_snapshot()
        for action in ("up", "left", "teleport", "", None):
            with self.assertRaises(ValueError):
                env.step(action)
            self.assertEqual(before, env.private_snapshot())
        with self.assertRaises(ValueError):
            env.step("wait", {"seed": 5})
        with self.assertRaises(ValueError):
            env.step("wait", {"diagnostics": ({"seed": 5},)})
        self.assertEqual(before, env.private_snapshot())

    def test_shared_occupancy_is_permitted(self):
        env = fixture()
        env.positions[1] = (1, 0)
        env.step("down")
        self.assertEqual(env.positions[0], env.positions[1])

    def test_arena_does_not_expand_zero_region(self):
        env = fixture(mines=[(4, c) for c in range(9)] + [(5, 0)])
        self.assertEqual(env._number((1, 0)), 0)
        env.step("down")
        self.assertEqual(len(env._revealed), 3)

    def test_observation_and_replay_are_detached_and_private(self):
        env = Arena(1024)
        obs = env.observe()
        expected_keys = {"rule_version", "game_id", "revision", "size", "mine_count", "turn", "positions", "scores", "alive", "diamonds", "known_safe", "revealed", "steps", "max_steps", "done", "winner", "reason", "legal_actions"}
        self.assertEqual(set(obs), expected_keys)
        obs["positions"][0][0] = 7
        obs["scores"][0] = 99
        obs["known_safe"].clear()
        self.assertEqual(env.positions[0], (0, 0))
        self.assertEqual(env.scores, [0, 0])
        self.assertEqual(len(env._known_safe), 5)
        env.step("wait")
        replay = env.save_replay()
        def inspect(obj):
            if isinstance(obj, dict):
                self.assertFalse({"seed", "mines", "mine_map", "private_snapshot"} & set(obj))
                for value in obj.values(): inspect(value)
            elif isinstance(obj, list):
                for value in obj: inspect(value)
        inspect(replay)
        json.dumps(replay, allow_nan=False)
        replay["records"][0]["next_observation"]["scores"][0] = 99
        self.assertEqual(env.save_replay()["records"][0]["next_observation"]["scores"], [0, 0])

    def test_snapshot_branch_and_replay_are_consistent(self):
        env = Arena(86)
        rng = random.Random(19)
        for _ in range(12):
            if env.done: break
            env.step(rng.choice(env.legal_actions()), {"policy": "random-test"})
        restored = Arena.from_snapshot(json.loads(json.dumps(env.private_snapshot())))
        self.assertEqual(env.observe(), restored.observe())
        for _ in range(40):
            if env.done: break
            action = rng.choice(env.legal_actions())
            self.assertEqual(env.step(action), restored.step(action))
        self.assertEqual(env.save_replay(), restored.save_replay())
        frames = env.save_replay()
        previous = frames["initial_observation"]
        for record in frames["records"]:
            self.assertEqual(previous, record["observation"])
            self.assertIn(record["action"], previous["legal_actions"])
            previous = record["next_observation"]
        self.assertEqual(previous, frames["final_observation"])

    def test_reset_rotates_id_and_clears_old_actions(self):
        env = Arena(2)
        original_id = env.game_id
        env.step("wait")
        obs = env.reset()
        self.assertNotEqual(original_id, obs["game_id"])
        self.assertEqual(obs["revision"], 0)
        self.assertEqual(obs["steps"], 0)
        self.assertEqual(env.save_replay()["records"], [])

    def test_random_legal_actions_terminate_and_conserve_resources(self):
        rng = random.Random(444)
        for seed in range(100):
            env = Arena(seed, first=seed % 2, swap=bool(seed % 3))
            original = env._mines.copy()
            while not env.done:
                env.step(rng.choice(env.legal_actions()), {"policy": "random-test"})
                self.assertLessEqual(env.steps, 200)
                self.assertEqual(sum(env.scores) + len(env._diamonds), 3)
                self.assertEqual(env._mines, original)
                self.assertFalse(env._known_safe & env._mines)
                if not env.done:
                    self.assertTrue(env.alive[env.turn])


class ClassicRulesTests(unittest.TestCase):
    def test_fixed_mines_and_no_first_click_protection(self):
        env = Classic(7)
        mines = env._mines.copy()
        env.reveal(*next(iter(mines)))
        self.assertEqual(env.winner, "loss")
        self.assertEqual(env._mines, mines)
        self.assertFalse(env.observe()["first_click_safe"])

    def test_flag_is_only_a_mark(self):
        env = Classic(5)
        safe = next((r, c) for r in range(9) for c in range(9) if (r, c) not in env._mines)
        mines = env._mines.copy()
        env.flag(*safe)
        self.assertIn(list(safe), env.observe()["flags"])
        with self.assertRaises(ValueError): env.reveal(*safe)
        self.assertEqual(env._mines, mines)
        env.flag(*safe)
        env.reveal(*safe)
        self.assertIn(safe, env._revealed)

    def test_zero_region_and_win(self):
        env = Classic(0)
        env._mines = {(8, c) for c in range(9)} | {(7, 8)}
        env.reveal(0, 0)
        self.assertGreater(len(env._revealed), 1)
        self.assertFalse(set(env._revealed) & env._mines)
        for r in range(9):
            for c in range(9):
                if not env.done and (r, c) not in env._mines:
                    env.reveal(r, c)
        self.assertEqual(env.winner, "win")
        self.assertEqual(len(env._revealed), 71)

    def test_no_hidden_map_in_classic_observation(self):
        obs = Classic(123).observe()
        self.assertNotIn("seed", obs)
        self.assertNotIn("mines", obs)
        self.assertEqual(obs["revealed"], [])


if __name__ == "__main__":
    unittest.main()
