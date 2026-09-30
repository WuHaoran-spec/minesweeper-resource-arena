"""Session integration and public-information boundary regression tests."""
from copy import deepcopy
from itertools import combinations
import json
import random
import unittest
from unittest.mock import patch

from arena.env import Arena
from arena.risk import infer
from arena.policies import choose, prepare
from arena.server import Session


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.session = Session()
        self.session.reset({'controllers': ['human', 'human'], 'source': 'browser_automation'})

    def payload(self, **extra):
        return {'epoch': self.session.epoch, 'revision': self.session.env.revision, **extra}

    def act(self, action='wait'):
        return self.session.mutate('/api/action', self.payload(action=action))

    def test_reset_invalidates_epoch_and_old_requests_are_atomic(self):
        old = self.payload(action='wait')
        self.session.reset({'controllers': ['human', 'human']})
        before = self.session.state()
        with self.assertRaisesRegex(ValueError, '旧对局'):
            self.session.mutate('/api/action', old)
        self.assertEqual(before, self.session.state())

    def test_old_revision_is_rejected_for_action_and_snapshot(self):
        old = self.payload(action='wait')
        self.act()
        before = self.session.state()
        for endpoint in ('/api/action', '/api/snapshot', '/api/restore'):
            with self.assertRaisesRegex(ValueError, '旧动作'):
                self.session.mutate(endpoint, old)
        self.assertEqual(before, self.session.state())

    def test_invalid_reset_preserves_entire_current_session(self):
        self.act()
        self.session.mutate('/api/flag', self.payload(r=4, c=4))
        before, replay = self.session.state(), self.session.replay()
        invalid = [None, [], {'mode':'bad'}, {'mode':'classic', 'controllers':['human','bad']},
                   {'controllers':'human'}, {'controllers':[[], 'B2']},
                   {'mode':'classic', 'source':'fabricated_human'}, {'source':[]}]
        for data in invalid:
            with self.assertRaises(ValueError):
                self.session.mutate('/api/reset', data)
            self.assertEqual(before, self.session.state())
            self.assertEqual(replay, self.session.replay())

    def test_pause_blocks_human_action_and_ai_auto_but_allows_single(self):
        self.session.mutate('/api/pause', self.payload(paused=True))
        with self.assertRaisesRegex(ValueError, '暂停'):
            self.act()
        self.session.reset({'controllers':['B0', 'B0'], 'source':'browser_automation'})
        self.session.mutate('/api/pause', self.payload(paused=True))
        with self.assertRaisesRegex(ValueError, '暂停'):
            self.session.mutate('/api/ai', self.payload())
        state = self.session.mutate('/api/ai', self.payload(single=True))
        self.assertEqual(state['observation']['steps'], 1)
        self.assertTrue(state['paused'])
        record = self.session.replay()['records']['records'][-1]
        self.assertEqual(record['metadata']['source'], 'browser_automation')
        self.assertEqual(record['metadata']['actor_source'], 'heuristic_policy')

    def test_controller_enforcement_and_illegal_action(self):
        self.session.reset({'controllers':['B0','human']})
        with self.assertRaisesRegex(ValueError, 'AI轮次'): self.act()
        self.session.reset({'controllers':['human','B0']})
        with self.assertRaisesRegex(ValueError, '人工轮次'):
            self.session.mutate('/api/ai', self.payload())
        before = self.session.state()
        with self.assertRaises(ValueError): self.act('teleport')
        self.assertEqual(before, self.session.state())

    def test_snapshot_restores_state_flags_and_labels_branch(self):
        self.act()
        self.session.mutate('/api/flag', self.payload(r=4, c=4))
        saved = self.session.state()
        result = self.session.mutate('/api/snapshot', self.payload())
        self.assertEqual(set(result), {'snapshot_id', 'observation'})
        self.act()
        self.session.mutate('/api/flag', self.payload(r=5, c=5))
        old_epoch = self.session.epoch
        restored = self.session.mutate('/api/restore', self.payload(snapshot_id=result['snapshot_id']))
        self.assertNotEqual(restored['epoch'], old_epoch)
        self.assertEqual(restored['observation'], saved['observation'])
        self.assertEqual(restored['flags'], [[4, 4]])
        self.assertEqual(restored['source'], 'counterfactual_replay')
        self.assertTrue(restored['paused'])
        self.session.mutate('/api/pause', self.payload(paused=False))
        self.act()
        replay = self.session.replay()
        self.assertEqual(replay['source'], 'counterfactual_replay')
        self.assertEqual(replay['records']['records'][-1]['metadata']['source'], 'counterfactual_replay')

    def test_bad_restore_is_atomic(self):
        before = self.session.state()
        with self.assertRaises(KeyError):
            self.session.mutate('/api/restore', self.payload(snapshot_id='missing'))
        self.assertEqual(before, self.session.state())

    def test_flag_is_not_a_risk_constraint(self):
        before = self.session.env.observe()
        p_before = infer(before)
        decision_before = choose(before, 'B2')
        self.session.mutate('/api/flag', self.payload(r=4, c=4))
        after = self.session.env.observe()
        self.assertEqual(before, after)
        self.assertEqual(p_before, infer(after))
        self.assertEqual(decision_before, choose(after, 'B2'))
        self.assertEqual(self.session.state()['flags'], [[4, 4]])
        self.assertEqual(self.session.replay()['frames'][-1]['ui_flags'], [[4, 4]])
        self.assertNotIn('ui_flags', after)
        self.assertEqual(len(self.session.replay()['frames']), 2)

    def test_policy_call_receives_public_observation_only(self):
        self.session.reset({'controllers':['B2', 'human'], 'source':'browser_automation'})
        public = self.session.env.observe()
        def fake_choose(obs, policy, rng):
            self.assertEqual(obs, public)
            obs['scores'][0] = 999  # Defensive copy also protects state.
            return {'action':'wait', 'policy':policy, 'risk_status':'test_stub', 'risk':[]}
        with patch('arena.policies.choose', side_effect=fake_choose):
            self.session.mutate('/api/ai', self.payload())
        self.assertEqual(self.session.env.scores, [0,0])

    def test_public_json_excludes_private_truth_and_marks_automation(self):
        self.act()
        saved = self.session.mutate('/api/snapshot', self.payload())
        exports = [self.session.state(), self.session.replay(), saved]
        def inspect(obj):
            if isinstance(obj, dict):
                self.assertFalse({'seed','mines','mine_map','hidden_mines','private_snapshot','rng_state'} & set(obj))
                for value in obj.values(): inspect(value)
            elif isinstance(obj, list):
                for value in obj: inspect(value)
        for value in exports:
            inspect(value)
            json.dumps(value, ensure_ascii=False, allow_nan=False)
        record = self.session.replay()['records']['records'][0]
        self.assertEqual(record['metadata']['source'], 'browser_automation')
        self.assertEqual(record['metadata']['policy'], 'human_input')
        self.assertIn('不证明真实人类参与', self.session.replay()['notice'])

    def test_return_values_cannot_mutate_session(self):
        state = self.session.state()
        state['controllers'][0] = 'B0'
        self.assertEqual(self.session.controllers[0], 'human')
        data = {'controllers':['human','B2']}
        self.session.reset(data)
        data['controllers'][0] = 'B0'
        self.assertEqual(self.session.controllers[0], 'human')

    def test_classic_flag_and_snapshot_contract(self):
        self.session.reset({'mode':'classic', 'source':'browser_automation'})
        self.session.mutate('/api/flag', self.payload(r=0,c=0))
        self.assertEqual(self.session.env.observe()['flags'], [[0,0]])
        self.assertEqual(self.session.replay()['frames'][-1]['ui_flags'], [[0,0]])
        self.assertEqual(len(self.session.replay()['frames']), 2)
        with self.assertRaises(ValueError):
            self.session.mutate('/api/snapshot', self.payload())
        with self.assertRaises(ValueError):
            self.session.mutate('/api/action', self.payload(r=0,c=0))
        self.assertEqual(self.session.env.observe()['revealed'], [])

    def test_snapshot_memory_cap_evicts_oldest(self):
        first = self.session.mutate('/api/snapshot', self.payload())['snapshot_id']
        for _ in range(100):
            last = self.session.mutate('/api/snapshot', self.payload())['snapshot_id']
        self.assertEqual(len(self.session.snapshots), 100)
        self.assertNotIn(first, self.session.snapshots)
        self.assertIn(last, self.session.snapshots)


class RiskBoundaryTests(unittest.TestCase):
    def test_small_exact_posteriors_against_independent_enumeration(self):
        rng = random.Random(882)
        for _ in range(20):
            cells = [(r,c) for r in range(4) for c in range(4)]
            truth = set(rng.sample(cells,3))
            known = set(rng.sample([x for x in cells if x not in truth],4))
            reveal = rng.sample(sorted(known),3)
            def clue(cell, mines):
                r,c = cell
                return sum(abs(r-mr)<=1 and abs(c-mc)<=1 for mr,mc in mines)
            observed = [[r,c,clue((r,c),truth)] for r,c in reveal]
            obs = {'size':4,'mine_count':3,'known_safe':[list(x) for x in known], 'revealed':observed}
            candidates = []
            for proposed in combinations([x for x in cells if x not in known],3):
                if all(clue((r,c),proposed)==n for r,c,n in observed):
                    candidates.append(set(proposed))
            p,status,_ = infer(obs)
            self.assertEqual(status,'exact_global_model_count')
            self.assertGreater(len(candidates),0)
            for i,cell in enumerate(cells):
                expected = sum(cell in m for m in candidates)/len(candidates)
                self.assertAlmostEqual(p[i],expected,places=12)

    def test_b1_and_no_opponent_features_ignore_opponent_position_and_score(self):
        obs = Arena(88).observe()
        altered = deepcopy(obs)
        altered['positions'][1] = [3,5]
        altered['scores'][1] = 2
        altered['alive'][1] = False
        self.assertEqual(choose(obs,'B1'), choose(altered,'B1'))
        X = prepare(obs, no_opponent=True)[0]
        X_other = prepare(altered, no_opponent=True)[0]
        self.assertTrue((X == X_other).all())


if __name__ == '__main__':
    unittest.main()
