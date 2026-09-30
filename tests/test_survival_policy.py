import copy
import hashlib
import json
import unittest
import numpy as np
from arena.env import Arena
from arena.policies import ACTIONS, MODEL_DIR, choose
from arena.survival import prepare_survival, risk_weight, load_survival


class SurvivalPolicyTests(unittest.TestCase):
    def test_rectangular_public_features_and_legal_mask(self):
        for preset in ('intermediate', 'expert'):
            obs = Arena(38891, preset=preset, scoring='survival-v1').observe()
            before = copy.deepcopy(obs)
            X, mask, p, status, detail, targets, base, costs = prepare_survival(obs)
            self.assertEqual(X.shape, (5, 40))
            self.assertEqual(len(p), obs['width']*obs['height'])
            self.assertTrue(np.isfinite(X).all())
            self.assertEqual(mask.tolist(), [a in obs['legal_actions'] for a in ACTIONS])
            self.assertEqual(len(targets), obs['resource_count'])
            self.assertIn(choose(obs, 'B3')['action'], obs['legal_actions'])
            self.assertEqual(obs, before)

    def test_lower_life_increases_public_risk_path_penalty(self):
        obs = Arena(476329, preset='expert', scoring='survival-v1').observe()
        low = copy.deepcopy(obs); low['lives'][obs['turn']] = 1
        normal = prepare_survival(obs); cautious = prepare_survival(low)
        self.assertEqual([risk_weight(i) for i in (3, 2, 1)], [8., 18., 72.])
        np.testing.assert_array_equal(normal[1], cautious[1])
        self.assertTrue(np.all(cautious[6] >= normal[6]-1e-6))
        self.assertGreater(cautious[4]['risk_weight'], normal[4]['risk_weight'])

    def test_public_repeat_history_increases_only_visited_cost(self):
        obs = Arena(38129, preset='intermediate', scoring='survival-v1').observe()
        repeated = copy.deepcopy(obs)
        repeated['recent_positions'][0] = [[1, 0], [0, 0], [1, 0], [0, 0]]
        normal, altered = prepare_survival(obs), prepare_survival(repeated)
        np.testing.assert_array_equal(normal[1], altered[1])
        np.testing.assert_array_equal(normal[6], altered[6])
        self.assertGreater(altered[7][1, 0], normal[7][1, 0])
        np.testing.assert_array_equal(normal[7][3], altered[7][3])

    def test_input_resource_order_and_opaque_ids_do_not_change_action(self):
        obs = Arena(19417, preset='expert', scoring='survival-v1').observe()
        reordered = copy.deepcopy(obs); reordered['diamonds'].reverse()
        reordered['game_id'] = 'unrelated'; reordered['revision'] = 8991
        np.testing.assert_array_equal(prepare_survival(obs)[0], prepare_survival(reordered)[0])
        self.assertEqual(choose(obs, 'B3')['action'], choose(reordered, 'B3')['action'])

    def test_old_rule_distribution_explicit(self):
        obs = Arena(38129, preset='intermediate').observe()
        self.assertIn('out_of_distribution', choose(obs, 'B3')['diagnostics'])

    @unittest.skipUnless((MODEL_DIR/'survival_v3_final_training.json').exists(), 'Run real survival training first')
    def test_real_checkpoint_and_public_loading(self):
        info = json.loads((MODEL_DIR/'survival_v3_final_training.json').read_text(encoding='utf-8'))
        model, digest = load_survival()
        self.assertEqual(model['W1'].shape, (40, 48))
        self.assertEqual(sum(value.size for value in model.values()), 2017)
        self.assertGreater(info['optimizer_steps'], 0)
        self.assertEqual(info['sha256'], digest)
        self.assertEqual(hashlib.sha256((MODEL_DIR/'survival_v3_final.npz').read_bytes()).hexdigest(), digest)
        for preset in ('intermediate', 'expert'):
            obs = Arena(93751, preset=preset, scoring='survival-v1').observe()
            decision = choose(obs, 'L3')
            self.assertIn(decision['action'], obs['legal_actions'])
            self.assertEqual(decision['diagnostics']['model_sha256'], digest)
            self.assertNotIn('out_of_distribution', decision['diagnostics'])


if __name__ == '__main__':
    unittest.main()
