import copy
import unittest
from arena.env import Arena


class SurvivalScoringTests(unittest.TestCase):
    def test_legacy_payload_and_default_scoring_stay_versioned(self):
        for preset,version in [('legacy','arena-v1.0'),('intermediate','arena-v2.0')]:
            obs=Arena(12,preset=preset).observe()
            self.assertEqual(obs['rule_version'],version)
            self.assertNotIn('utility_scores',obs)
        with self.assertRaises(ValueError):Arena(1,scoring='survival-v1')

    def test_gems_dominate_lives_and_lives_dominate_actions(self):
        env=Arena(14,preset='intermediate',scoring='survival-v1')
        env.scores=[1,0];env.lives=[0,3];env.action_counts=[env.max_steps,0]
        self.assertGreater(env.utility_scores()[0],env.utility_scores()[1])
        env.scores=[1,1];env.lives=[2,1]
        self.assertGreater(env.utility_scores()[0],env.utility_scores()[1])

    def test_terminal_uses_integer_efficiency_tie_break(self):
        env=Arena(15,preset='intermediate',scoring='survival-v1',max_steps=3)
        for _ in range(3):env.step('wait')
        self.assertEqual(env.action_counts,[2,1])
        self.assertEqual(env.winner,1)
        self.assertEqual(env.reason,'max_steps')

    def test_public_history_is_bounded_and_roundtrips(self):
        env=Arena(16,preset='expert',scoring='survival-v1')
        for _ in range(54):env.step('wait')
        obs=env.observe()
        self.assertEqual([len(x) for x in obs['recent_positions']],[24,24])
        restored=Arena.from_snapshot(env.private_snapshot())
        self.assertEqual(restored.observe(),obs)
        self.assertEqual(restored.step('wait'),env.step('wait'))
        obs['recent_positions'][0][0][0]=999
        self.assertNotEqual(env.observe()['recent_positions'][0][0][0],999)

    def test_scoring_does_not_change_unknown_action_mask_or_map(self):
        old=Arena(17,preset='intermediate');new=Arena(17,preset='intermediate',scoring='survival-v1')
        self.assertEqual(old._mines,new._mines)
        self.assertEqual(old.legal_actions(),new.legal_actions())
        new._mines={(0,1)}
        self.assertEqual(old.legal_actions(),new.legal_actions())

    def test_feedback_is_realized_utility_not_probability_label(self):
        env=Arena(18,preset='intermediate',scoring='survival-v1')
        before=env.utility_scores()[0];env.step('wait')
        f=env.save_replay()['records'][-1]['feedback']
        self.assertAlmostEqual(f['utility_delta'],env.utility_scores()[0]-before)
        self.assertEqual(f['utility_delta'],-1/env.max_steps)


if __name__=='__main__':unittest.main()
