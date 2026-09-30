import itertools
import random
import unittest
from pathlib import Path
import numpy as np
from arena.risk import infer, neighbors
from arena.env import Arena
from arena.policies import choose, prepare, load_model


class RiskAndPolicyTests(unittest.TestCase):
    def test_exact_posterior_matches_exhaustive_uniform_maps(self):
        for seed in range(24):
            rng = random.Random(seed)
            true = set(rng.sample(range(16), 3))
            safe = rng.sample(sorted(set(range(16))-true), 4)
            revealed = [[i//4, i%4, len(set(neighbors(i, 4)) & true)] for i in safe[:3]]
            obs = {'size': 4, 'mine_count': 3, 'known_safe': [[i//4, i%4] for i in safe], 'revealed': revealed}
            p, status, _ = infer(obs)
            possibilities = []
            for values in itertools.combinations(sorted(set(range(16))-set(safe)), 3):
                values = set(values)
                if all(len(set(neighbors(r*4+c, 4)) & values) == n for r, c, n in revealed):
                    possibilities.append(values)
            expected = [sum(i in x for x in possibilities)/len(possibilities) for i in range(16)]
            self.assertEqual(status, 'exact_global_model_count')
            np.testing.assert_allclose(p, expected, atol=1e-12)
            self.assertAlmostEqual(sum(p), 3.)

    def test_public_exploded_mine_remains_one(self):
        p, status, _ = infer({'size':3, 'mine_count':2, 'known_safe':[[0,0]], 'revealed':[[0,0,1],[1,1,-1]]})
        self.assertEqual(p[4], 1.)
        self.assertEqual(p[0], 0.)
        self.assertAlmostEqual(sum(p), 2.)

    def test_policies_make_only_public_legal_choices(self):
        for seed in range(8):
            env = Arena(seed)
            for _ in range(35):
                obs = env.observe()
                if obs['done']:
                    break
                policy = ('B0', 'B1', 'B2')[obs['steps']%3]
                decision = choose(obs, policy, random.Random(seed+obs['steps']))
                self.assertIn(decision['action'], obs['legal_actions'])
                self.assertEqual(len(decision['risk']), 81)
                self.assertTrue(all(0 <= x <= 1 for x in decision['risk']))
                env.step(decision['action'], decision)

    def test_no_opponent_features_invariant_to_opponent_position(self):
        obs = Arena(42).observe()
        first = prepare(obs, True)[0]
        obs['positions'][1] = [4, 5]
        obs['scores'][1] = 2
        obs['alive'][1] = False
        np.testing.assert_array_equal(first, prepare(obs, True)[0])

    @unittest.skipUnless(Path('models/L_final.npz').exists(), 'Run training to create weights')
    def test_saved_model_loaded_and_legal(self):
        self.assertEqual(load_model('L')['W1'].shape, (27, 32))
        obs = Arena(928).observe()
        result = choose(obs, 'L')
        self.assertIn(result['action'], obs['legal_actions'])
        self.assertIn('action_logits', result['diagnostics'])


if __name__ == '__main__':
    unittest.main()
