"""Integration checks for the larger UI presets and independent search policy."""
import unittest
from unittest.mock import patch
from arena.server import Session

class ChallengeSessionTests(unittest.TestCase):
    def payload(self, session, **extra):
        return {'epoch':session.epoch,'revision':session.env.revision,**extra}

    def test_expert_reset_flag_bounds_and_snapshot_preserve_dimensions(self):
        session=Session();session.reset({'preset':'expert','controllers':['human','human']})
        obs=session.state()['observation']
        self.assertEqual((obs['width'],obs['height'],obs['mine_count']), (30,16,99))
        session.mutate('/api/flag',self.payload(session,r=15,c=29))
        before=session.state()
        with self.assertRaises(ValueError):session.mutate('/api/flag',self.payload(session,r=16,c=29))
        self.assertEqual(session.state(),before)
        saved=session.mutate('/api/snapshot',self.payload(session))
        session.mutate('/api/action',self.payload(session,action='right'))
        session.mutate('/api/restore',self.payload(session,snapshot_id=saved['snapshot_id']))
        self.assertEqual(session.state()['observation'],obs)
        self.assertEqual(session.state()['source'],'counterfactual_replay')

    def test_classic_expert_corner_and_chord_route(self):
        session=Session();session.reset({'preset':'expert','mode':'classic'})
        result=session.mutate('/api/action',self.payload(session,r=15,c=29))
        obs=result['observation']
        self.assertTrue(obs['first_click_safe'])
        self.assertFalse(obs['done'])
        self.assertIn([15,29,0],obs['revealed'])
        before=session.state()
        with self.assertRaises(ValueError):session.mutate('/api/chord',self.payload(session,r=15,c=29))
        self.assertEqual(session.state(),before)

    def test_search_policy_source_is_not_teacher_or_human(self):
        session=Session();session.reset({'preset':'intermediate','controllers':['E','B2'],'source':'browser_automation'})
        def public_choice(obs,policy,rng):
            self.assertEqual(policy,'E');self.assertNotIn('seed',obs);self.assertNotIn('mines',obs)
            return {'action':'wait','policy':'E','risk_status':'fixture','risk':[]}
        with patch('arena.policies.choose',side_effect=public_choice):
            session.mutate('/api/ai',self.payload(session))
        record=session.replay()['records']['records'][0]
        self.assertEqual(record['metadata']['actor_source'],'search_trained_policy')
        self.assertEqual(record['metadata']['source'],'browser_automation')

if __name__=='__main__':unittest.main()
