"""Exercise new modes through the same Session used by the HTTP UI."""
import copy
import json
import unittest
from unittest.mock import patch
from arena.server import Session


class NewModeIntegration(unittest.TestCase):
    def payload(self, session, **extra):
        return dict(epoch=session.epoch, revision=session.env.revision, **extra)

    def test_race_policy_receives_only_own_board_and_exports_real_action(self):
        s=Session();s.reset(dict(mode='classic_race',preset='beginner',controllers=['R1','C1'],source='browser_automation'))
        before=s.env.observe();own=copy.deepcopy(before['boards'][0]);cell=own['legal_cells'][0]
        def policy(obs,policy,rng):
            self.assertEqual(obs,own)
            self.assertNotIn('boards',obs)
            obs['revealed'].clear()
            return dict(cell=cell,action=cell,policy=policy,risk=[])
        with patch('arena.classic_race.choose',side_effect=policy):
            state=s.mutate('/api/ai',self.payload(s))
        self.assertEqual(state['observation']['revision'],1)
        self.assertEqual(state['decision']['actor'],0)
        replay=s.replay();record=replay['records']['records'][0]
        self.assertEqual(record['decision_observation'],own)
        self.assertEqual(record['metadata']['source'],'browser_automation')
        self.assertEqual(len(replay['frames']),2)
        json.dumps(replay,allow_nan=False)

    def test_race_human_ai_pause_and_invalid_reset_boundaries(self):
        s=Session();s.reset(dict(mode='classic_race',preset='beginner',controllers=['human','C1']))
        with self.assertRaises(ValueError):s.mutate('/api/ai',self.payload(s))
        s.mutate('/api/pause',self.payload(s,paused=True))
        cell=s.env.observe()['boards'][0]['legal_cells'][0]
        with self.assertRaises(ValueError):s.mutate('/api/action',self.payload(s,r=cell[0],c=cell[1]))
        before=s.state()
        for change in (dict(mode='classic_race',controllers=['B3','C1']),dict(mode='arena',scoring='invented')):
            with self.assertRaises(ValueError):s.reset(change)
            self.assertEqual(before,s.state())
        for endpoint in ('/api/flag','/api/snapshot'):
            with self.assertRaises(ValueError):s.mutate(endpoint,self.payload(s,r=0,c=0))

    def test_survival_model_is_loaded_and_snapshot_preserves_scoring(self):
        s=Session();s.reset(dict(preset='intermediate',scoring='survival-v1',controllers=['L3','B3'],source='browser_automation'))
        state=s.mutate('/api/ai',self.payload(s))
        self.assertEqual(state['decision']['policy'],'L3')
        self.assertEqual(state['observation']['action_counts'],[1,0])
        saved=s.mutate('/api/snapshot',self.payload(s))
        s.mutate('/api/ai',self.payload(s))
        restored=s.mutate('/api/restore',self.payload(s,snapshot_id=saved['snapshot_id']))
        self.assertEqual(restored['observation'],saved['observation'])
        self.assertEqual(restored['source'],'counterfactual_replay')
        self.assertEqual(restored['observation']['scoring'],'survival-v1')


if __name__=='__main__':unittest.main()
