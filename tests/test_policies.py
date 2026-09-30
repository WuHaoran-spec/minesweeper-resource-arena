import copy
import hashlib
import json
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from arena.env import Arena
from arena.policies import MODEL_DIR, choose, load_model, prepare


class TrainedPolicyContractTests(unittest.TestCase):
    def test_shipped_weights_match_actual_training_logs(self):
        for policy,stem in [('L','L_final'),('L_initial','L_initial'),('L_no_opponent','L_no_opponent')]:
            model=load_model(policy)
            path=MODEL_DIR/(stem+'.npz')
            info=json.loads((MODEL_DIR/(stem+'_training.json')).read_text(encoding='utf-8'))
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),info['sha256'])
            self.assertEqual(sum(v.size for v in model.values()),929)
            self.assertEqual(info['selected']['validation_loss'],min(x['validation_loss'] for x in info['history']))

    def test_models_play_complete_legal_games(self):
        for policy in ['L','L_initial','L_no_opponent']:
            for seed in [83191,94409]:
                env=Arena(seed)
                while not env.done:
                    obs=env.observe()
                    decision=choose(obs,policy)
                    self.assertIn(decision['action'],obs['legal_actions'])
                    self.assertTrue(all(np.isfinite(v) for v in decision['diagnostics']['action_logits'].values()))
                    self.assertEqual(decision['diagnostics']['target_kind'],'inferred_from_selected_action_not_network_head')
                    env.step(decision['action'],decision)
                self.assertLessEqual(env.steps,200)

    def test_b1_action_and_ablation_features_ignore_opponent_fields(self):
        for seed in [31871,67678,92771]:
            obs=Arena(seed).observe()
            other=copy.deepcopy(obs)
            other['positions'][1]=[4,4]
            other['scores'][1]=3
            other['alive'][1]=False
            self.assertEqual(choose(obs,'B1')['action'],choose(other,'B1')['action'])
            np.testing.assert_array_equal(prepare(obs,True)[0],prepare(other,True)[0])
            self.assertEqual(choose(obs,'L_no_opponent')['action'],choose(other,'L_no_opponent')['action'])

    def test_missing_weight_never_silently_uses_heuristic(self):
        with patch('arena.policies.MODEL_DIR',Path('this_weight_directory_does_not_exist')):
            with self.assertRaises(FileNotFoundError):
                choose(Arena(19877).observe(),'L')


if __name__=='__main__': unittest.main()
