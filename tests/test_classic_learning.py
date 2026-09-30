import unittest
import numpy as np
from arena.classic_learning import observed_utility,forward,public_only,MODELS,digest
from arena.classic_race import ClassicRace,choose


class ClassicFeedbackTests(unittest.TestCase):
    def test_target_is_actual_local_utility_not_probability(self):
        self.assertEqual(observed_utility({'exploded':True,'new_safe_revealed':0}),0.)
        self.assertEqual(observed_utility({'exploded':False,'new_safe_revealed':1}),1.025)
        self.assertEqual(observed_utility({'exploded':False,'new_safe_revealed':25}),1.25)

    def test_network_uses_saved_normalization_and_tanh(self):
        model={'mean':np.ones(10),'scale':np.full(10,2.),'W1':np.ones((10,16)),
               'b1':np.zeros(16),'W2':np.ones((16,1)),'b2':np.array([.5])}
        np.testing.assert_allclose(forward(np.ones((2,10)),model),[.5,.5])
        np.testing.assert_allclose(forward(np.full((1,10),1.2),model),[16*np.tanh(1)+.5])

    def test_nested_private_truth_rejected(self):
        for key in ('seed','map_seed','mines','private_snapshot'):
            with self.assertRaises(AssertionError):public_only({'rows':[{key:[]} ]})

    def test_actual_checkpoint_load_and_out_of_distribution_diagnostic(self):
        for preset,area in (('beginner',81),('expert',480)):
            observation=ClassicRace(22,preset=preset).current_observation()
            result=choose(observation,'C1')
            self.assertIn(result['cell'],observation['legal_cells'])
            self.assertEqual(len(result['risk']),area)
            self.assertEqual(result['diagnostics']['model_sha256'],digest(MODELS/'classic_v3_final.npz'))
            self.assertFalse(result['diagnostics']['score_is_mine_probability'])
            self.assertEqual('out_of_distribution' in result['diagnostics'],preset!='beginner')


if __name__=='__main__':unittest.main()
