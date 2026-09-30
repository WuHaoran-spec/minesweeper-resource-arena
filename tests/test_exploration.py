"""Teacher-free parameter-search policy and objective contract checks."""
from copy import deepcopy
import random
import unittest
from unittest.mock import patch
import numpy as np

from arena.env import Arena
from arena.exploration import decide, prepare_exploration, public_only, utility, weights_id


class ExplorationPolicyTests(unittest.TestCase):
    def test_terminal_objective_has_no_step_reward(self):
        self.assertEqual(utility(1, [0, 7]), 1.25)
        self.assertEqual(utility(0, [7, 0]), -.25)
        self.assertEqual(utility('draw', [3, 3]), .5)
        self.assertEqual(utility(0, [4, 3]), -1/28)

    def test_decision_requires_public_observation_and_no_teacher_action(self):
        observation = Arena(19, preset='intermediate').observe()
        original = deepcopy(observation)
        with patch('arena.policies.choose', side_effect=AssertionError('Teacher action called')):
            result = decide(observation, np.arange(8, dtype=float), rng=random.Random(1))
        self.assertEqual(observation, original)
        self.assertIn(result['action'], observation['legal_actions'])
        self.assertEqual(len(result['risk']), 256)
        self.assertEqual(result['diagnostics']['parameter_sha256'], weights_id(np.arange(8)))
        self.assertEqual(result['diagnostics']['action_scores'][result['action']],
                         max(result['diagnostics']['action_scores'].values()))
        public_only(result)

    def test_legal_mask_and_out_of_distribution_diagnostic(self):
        observation = Arena(0, preset='expert').observe()
        features, legal, risk, *_ = prepare_exploration(observation)
        self.assertEqual(features.shape, (5, 8))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(len(risk), 480)
        self.assertEqual(list(legal), [False, True, False, True, True])
        result = decide(observation, np.zeros(8), rng=random.Random(1), epsilon=1.)
        self.assertIn(result['action'], observation['legal_actions'])
        self.assertTrue(result['diagnostics']['sampled_exploration_action'])
        self.assertIn('out_of_distribution', result['diagnostics'])

    def test_invalid_parameters_and_terminal_refused(self):
        observation = Arena(1, preset='intermediate').observe()
        for weights in (np.zeros(7), [float('nan')]*8, [float('inf')]*8):
            with self.assertRaises(ValueError): decide(observation, weights)
        observation['done'] = True
        with self.assertRaises(ValueError): decide(observation, np.zeros(8))

    def test_private_nested_fields_refused(self):
        for field in ('seed', 'map_seed', 'mines', 'hidden_mines', 'private_snapshot', 'future_results'):
            with self.assertRaises(AssertionError): public_only({'diagnostics': [{field: []}]})


if __name__ == '__main__':
    unittest.main()
