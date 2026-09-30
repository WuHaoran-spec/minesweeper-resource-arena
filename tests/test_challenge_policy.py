import copy
import json
import unittest
import numpy as np
from arena.env import Arena
from arena.policies import MODEL_DIR, choose, prepare_v2, load_model


class ChallengePolicyTests(unittest.TestCase):
    def test_rectangular_features_and_all_resources(self):
        for preset,width,height,count in [('intermediate',16,16,7),('expert',30,16,11)]:
            obs=Arena(582762,preset=preset).observe()
            X,mask,p,status,detail,targets,base,costs=prepare_v2(obs)
            self.assertEqual(X.shape,(5,32));self.assertEqual(len(p),width*height)
            self.assertEqual(len(targets),count);self.assertEqual(base.shape,(5,count))
            self.assertEqual(len(detail['network_candidate_resources']),3)
            self.assertTrue(np.isfinite(X).all())
            for policy in ['B0','B1','B2','L']:
                self.assertIn(choose(obs,policy)['action'],obs['legal_actions'])
            self.assertIn('out_of_distribution',choose(obs,'L')['diagnostics'])

    def test_resource_input_order_is_not_a_hidden_feature(self):
        obs=Arena(932624,preset='expert').observe();altered=copy.deepcopy(obs)
        altered['diamonds'].reverse()
        np.testing.assert_array_equal(prepare_v2(obs)[0],prepare_v2(altered)[0])
        self.assertEqual(choose(obs,'B2')['action'],choose(altered,'B2')['action'])

    def test_b1_and_no_opponent_features_are_invariant(self):
        obs=Arena(734761,preset='expert').observe();other=copy.deepcopy(obs)
        other['positions'][1]=[5,10];other['scores'][1]=5;other['alive'][1]=False;other['lives'][1]=0
        self.assertEqual(choose(obs,'B1')['action'],choose(other,'B1')['action'])
        np.testing.assert_array_equal(prepare_v2(obs,True)[0],prepare_v2(other,True)[0])

    def test_fewer_than_three_remaining_resources(self):
        obs=Arena(324783,preset='expert').observe();obs['diamonds']=obs['diamonds'][:1]
        X,*_=prepare_v2(obs)
        self.assertEqual(X.shape,(5,32));self.assertTrue(np.isfinite(X).all())

    @unittest.skipUnless((MODEL_DIR/'challenge_final_training.json').exists(),'Run actual challenge training first')
    def test_real_challenge_checkpoint_loaded(self):
        info=json.loads((MODEL_DIR/'challenge_final_training.json').read_text(encoding='utf-8'))
        model=load_model('L_v2')
        self.assertEqual(model['W1'].shape,(32,48));self.assertEqual(sum(x.size for x in model.values()),1633)
        self.assertGreater(info['optimizer_steps'],0)
        for preset in ['intermediate','expert']:
            obs=Arena(843967,preset=preset).observe();decision=choose(obs,'L_v2')
            self.assertIn(decision['action'],obs['legal_actions'])
            self.assertNotIn('out_of_distribution',decision['diagnostics'])
            self.assertIn('action_logits',decision['diagnostics'])


if __name__=='__main__':unittest.main()
