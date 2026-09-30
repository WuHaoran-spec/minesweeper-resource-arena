"""Independent exact-count checks and conservative failure-label regression.

Truth below belongs only to constructed test fixtures. It is never a policy
input or a probability target obtained from a single observed mine outcome.
"""
import itertools
import random
import unittest

from arena.risk import infer


def adjacent(i, size):
    """Independent eight-neighbor implementation for the reference enumerator."""
    r, c = divmod(i, size)
    return {j for j in range(size * size)
            if j != i and abs(j // size - r) <= 1 and abs(j % size - c) <= 1}


def exhaustive(obs):
    size = obs['size']
    safe = {r * size + c for r, c in obs['known_safe']}
    safe |= {r * size + c for r, c, n in obs['revealed'] if n >= 0}
    bombs = {r * size + c for r, c, n in obs['revealed'] if n == -1}
    eligible = sorted(set(range(size * size)) - safe - bombs)
    layouts = []
    for subset in itertools.combinations(eligible, obs['mine_count'] - len(bombs)):
        mines = set(subset) | bombs
        if all(len(adjacent(r * size + c, size) & mines) == n
               for r, c, n in obs['revealed'] if n >= 0):
            layouts.append(mines)
    if not layouts:
        raise ValueError('No compatible layouts in independent enumerator')
    return [sum(i in mines for mines in layouts) / len(layouts)
            for i in range(size * size)], len(layouts)


class IndependentRiskTests(unittest.TestCase):
    def test_600_random_public_states_match_complete_enumeration(self):
        for seed in range(600):
            with self.subTest(case=seed):
                rng = random.Random(seed + 170000)
                size = rng.choice((3, 4))
                count = rng.randrange(1, min(size * size - 1, 5))
                truth = set(rng.sample(range(size * size), count))
                safe = rng.sample(sorted(set(range(size * size)) - truth),
                                  rng.randrange(1, size * size - count + 1))
                shown = rng.sample(safe, rng.randrange(len(safe) + 1))
                bombs = rng.sample(sorted(truth), rng.randrange(count + 1))
                obs = {'size': size, 'mine_count': count,
                       'known_safe': [[i // size, i % size] for i in safe],
                       'revealed': [[i // size, i % size, len(adjacent(i, size) & truth)]
                                    for i in shown] + [[i // size, i % size, -1] for i in bombs]}
                actual, status, detail = infer(obs)
                expected, models = exhaustive(obs)
                self.assertEqual(status, 'exact_global_model_count')
                self.assertEqual(detail['models'], models)
                for p, q in zip(actual, expected):
                    self.assertAlmostEqual(p, q, places=12)
                self.assertAlmostEqual(sum(actual), count, places=12)

    def test_disconnected_components_receive_global_weight(self):
        obs = {'size': 4, 'mine_count': 3, 'known_safe': [[0, 0], [3, 3]],
               'revealed': [[0, 0, 1], [3, 3, 1]]}
        probabilities, status, detail = infer(obs)
        self.assertEqual(detail['components'], 2)
        self.assertEqual(detail['models'], 3 * 3 * 8)
        self.assertEqual(status, 'exact_global_model_count')
        self.assertAlmostEqual(probabilities[1], 1 / 3)
        self.assertAlmostEqual(probabilities[2], 1 / 8)
        self.assertAlmostEqual(sum(probabilities), 3.)

    def test_public_safety_and_flags_have_different_meanings(self):
        obs = {'size': 3, 'mine_count': 2, 'known_safe': [[0, 0], [1, 1], [2, 2]],
               'revealed': [], 'flags': [[0, 1]]}
        probabilities, status, detail = infer(obs)
        self.assertEqual(detail['models'], 15)
        self.assertEqual(status, 'exact_global_model_count')
        self.assertEqual([probabilities[i] for i in (0, 4, 8)], [0., 0., 0.])
        self.assertAlmostEqual(probabilities[1], 1 / 3)
        obs['flags'] = []
        self.assertEqual(infer(obs)[0], probabilities)

    def test_large_frontier_cutoff_is_never_exact(self):
        revealed = [[2, 2, 1], [5, 0, 2], [5, 4, 1], [7, 6, 1], [2, 4, 1],
                    [0, 7, 1], [5, 3, 1], [3, 4, 1], [7, 4, 1], [4, 3, 1],
                    [8, 5, 0], [3, 8, 0], [0, 0, 1], [4, 5, 1]]
        obs = {'size': 9, 'mine_count': 10, 'known_safe': [row[:2] for row in revealed],
               'revealed': revealed}
        probabilities, status, detail = infer(obs)
        self.assertEqual(status, 'approximate_cutoff')
        self.assertNotIn('models', detail)
        self.assertIn('reason', detail)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))
        for r, c, _ in revealed:
            self.assertEqual(probabilities[r * 9 + c], 0.)

    def test_returned_probability_list_does_not_mutate_cache(self):
        obs = {'size': 3, 'mine_count': 1, 'known_safe': [[0, 0]], 'revealed': []}
        probabilities, _, detail = infer(obs)
        probabilities[0] = 1.
        detail['models'] = -1
        fresh, _, fresh_detail = infer(obs)
        self.assertEqual(fresh[0], 0.)
        self.assertEqual(fresh_detail['models'], 8)

    def test_impossible_observations_are_rejected(self):
        invalid = [
            {'size': 3, 'mine_count': 1, 'known_safe': [[0, 0]], 'revealed': [[0, 0, -1]]},
            {'size': 3, 'mine_count': 1, 'known_safe': [], 'revealed': [[0, 0, 0], [0, 0, 1]]},
            {'size': 3, 'mine_count': 0, 'known_safe': [[0, 0]], 'revealed': [[0, 0, 1]]},
            {'size': 2, 'mine_count': 1, 'known_safe': [[0, 0], [0, 1], [1, 0], [1, 1]],
             'revealed': [[0, 0, 1]]},
            {'size': 3, 'mine_count': 1, 'known_safe': [[-1, 0]], 'revealed': []},
            {'size': 3, 'mine_count': 1, 'known_safe': [], 'revealed': [[0, 0, 8]]},
            {'size': 3, 'mine_count': 1, 'known_safe': [], 'revealed': [[3, 0, 0]]},
        ]
        for obs in invalid:
            with self.subTest(observation=obs):
                with self.assertRaises(ValueError):
                    infer(obs)


if __name__ == '__main__':
    unittest.main()
