"""Bounded integration checks for isolated, paired multi-initialization batches."""
from contextlib import redirect_stdout
from copy import deepcopy
import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from arena import policies, survival, survival_training as training, training_batch as batch
from arena.challenge_training import public_only, read, sha, write
from arena.env import Arena


class TrainingBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='arena-batch-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.globals_before = (training.DATA, training.MODEL_DIR, policies.MODEL_DIR,
                               survival.MODEL_DIR, training.OPTIMIZER_SEED)
        self.baseline = policies.MODEL_DIR / 'survival_v3_final.npz'
        self.baseline_hash = sha(self.baseline)

    def tearDown(self):
        self.assertEqual(self.globals_before, (training.DATA, training.MODEL_DIR,
                         policies.MODEL_DIR, survival.MODEL_DIR, training.OPTIMIZER_SEED))
        self.assertEqual(self.baseline_hash, sha(self.baseline))

    @staticmethod
    def examples():
        rng = np.random.default_rng(77)
        X = rng.normal(0, .25, (12, 5, training.FEATURES)).astype(np.float32)
        mask = np.ones((12, 5), dtype=bool)
        mask[:, 4] = False
        return {'X': X, 'mask': mask, 'y': np.arange(12, dtype=np.int64) % 4}

    def frozen_fixture(self):
        out, private = self.root/'output', self.root/'private'
        truth = {'private': True, 'splits': {}}
        write(private/'manifest.json', truth)
        protocol = {'private_manifest_sha256': sha(private/'manifest.json'),
                    'source_sha256': {'arena/env.py': sha(batch.ROOT/'arena/env.py')},
                    'baseline_sha256': self.baseline_hash,
                    'collection_opponent_L2_sha256': sha(policies.MODEL_DIR/'challenge_final.npz')}
        write(out/'protocol.json', protocol)
        return out, private, truth, protocol

    def test_paths_reject_repository_ancestors_and_private_overlap(self):
        out, private = self.root/'output', self.root/'private'
        self.assertEqual(batch.validate_paths(out, private), (out.resolve(), private.resolve()))
        for public_path, private_path in (
                (batch.ROOT, private), (batch.ROOT.parent, private),
                (out, batch.ROOT/'private'), (out, out/'private'),
                (private/'output', private), (out, out)):
            with self.subTest(output=str(public_path), private=str(private_path)):
                with self.assertRaises(ValueError):
                    batch.validate_paths(public_path, private_path)

    def test_freeze_refuses_existing_output_before_generating_maps(self):
        out, private = self.root/'output', self.root/'private'
        out.mkdir()
        marker = out/'existing.txt'
        marker.write_text('preserve this result', encoding='utf-8')
        with patch.object(batch, 'Arena', side_effect=AssertionError('must not generate maps')):
            with self.assertRaises(FileExistsError):
                batch.create_protocol(out, private, [])
        self.assertEqual(marker.read_text(encoding='utf-8'), 'preserve this result')
        self.assertFalse(private.exists())

    def test_load_rejects_modified_private_source_and_old_model_hashes(self):
        out, private, truth, protocol = self.frozen_fixture()
        self.assertEqual(batch.load_batch(out, private)[2], truth)
        for field, message in (('private_manifest_sha256', 'Private manifest changed'),
                               ('baseline_sha256', 'Baseline changed'),
                               ('collection_opponent_L2_sha256', 'Collection opponent changed')):
            with self.subTest(field=field):
                changed = deepcopy(protocol)
                changed[field] = '0'*64
                write(out/'protocol.json', changed)
                with self.assertRaisesRegex(ValueError, message):
                    batch.load_batch(out, private)
        changed = deepcopy(protocol)
        changed['source_sha256']['arena/env.py'] = '0'*64
        write(out/'protocol.json', changed)
        with self.assertRaisesRegex(ValueError, 'Frozen implementation changed'):
            batch.load_batch(out, private)

    def test_explicit_default_optimizer_seed_reproduces_weights_in_new_directories(self):
        examples = self.examples()
        implicit_dir, explicit_dir = self.root/'implicit', self.root/'explicit'
        with redirect_stdout(io.StringIO()):
            a = training.fit(examples, examples, 'candidate.npz', 2, model_dir=implicit_dir)
            b = training.fit(examples, examples, 'candidate.npz', 2,
                             optimizer_seed=training.OPTIMIZER_SEED, model_dir=explicit_dir)
        for key in a:
            np.testing.assert_array_equal(a[key], b[key])
        first = read(implicit_dir/'candidate_training.json')
        second = read(explicit_dir/'candidate_training.json')
        self.assertEqual(first['history'], second['history'])
        self.assertEqual(first['selected'], second['selected'])
        self.assertEqual(first['optimizer_seed'], training.OPTIMIZER_SEED)
        self.assertEqual(second['optimizer_seed'], training.OPTIMIZER_SEED)
        self.assertTrue((implicit_dir/'candidate.npz').exists())
        self.assertTrue((explicit_dir/'candidate.npz').exists())

    def test_optimizer_exception_does_not_rebind_global_paths_or_create_checkpoint(self):
        destination = self.root/'failure'
        with patch.object(training, 'measure', side_effect=RuntimeError('injected measurement failure')):
            with self.assertRaisesRegex(RuntimeError, 'injected measurement failure'):
                training.fit(self.examples(), self.examples(), 'candidate.npz', 1,
                             optimizer_seed=19, model_dir=destination)
        self.assertFalse((destination/'candidate.npz').exists())
        self.assertFalse((destination/'candidate_training.json').exists())

    def test_three_runs_use_independent_initializations_and_explicit_model_directory(self):
        out = self.root/'batch'
        example = self.examples()
        for split in ('train', 'validation'):
            path = out/'data'/(split+'_features.npz')
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **example)
        protocol = {'run_seeds': list(batch.RUN_SEEDS), 'epochs_per_run': 1,
                    'candidate_nomination': 'lowest validation cross entropy'}
        with patch.object(batch, 'load_batch', return_value=(out, self.root/'private', {}, protocol)), \
                redirect_stdout(io.StringIO()):
            batch.train_all(out, self.root/'private')
        states, results = [], []
        for index, seed in enumerate(batch.RUN_SEEDS, 1):
            path = out/'models'/f'L3_run{index:02d}.npz'
            with np.load(path, allow_pickle=False) as checkpoint:
                states.append(checkpoint['W1'].copy())
            result = read(path.with_name(path.stem+'_training.json'))
            self.assertEqual(result['optimizer_seed'], seed)
            self.assertFalse(result['test_used_for_training'])
            self.assertEqual(result['sha256'], sha(path))
            results.append(result)
        self.assertTrue(all(not np.array_equal(states[i], states[j])
                            for i in range(3) for j in range(i)))
        selection = read(out/'selection_before_test.json')
        expected = min(results, key=lambda item: item['selected']['validation_loss'])
        self.assertEqual(selection['policy'], expected['run_id'])
        self.assertEqual(selection['sha256'], expected['sha256'])
        self.assertFalse(selection['test_read'])
        self.assertFalse(selection['game_default_changed'])
        with patch.object(batch, 'load_batch', return_value=(out, self.root/'private', {}, protocol)), \
                patch.object(training, 'fit', side_effect=AssertionError('completed runs must be reused')), \
                redirect_stdout(io.StringIO()):
            batch.train_all(out, self.root/'private')
        self.assertEqual(read(out/'selection_before_test.json'), selection)
        changed = dict(selection, sha256='0'*64)
        write(out/'selection_before_test.json', changed)
        with patch.object(batch, 'load_batch', return_value=(out, self.root/'private', {},
                          dict(protocol, baseline_sha256=self.baseline_hash))), \
                patch.object(batch, 'Arena', side_effect=RuntimeError('must reject before first test map')):
            with self.assertRaises(AssertionError):
                batch.evaluate(out, self.root/'private')
        self.assertFalse((out/'evaluation').exists())

    def test_collection_default_and_balanced_first_keep_public_only_replays(self):
        truth = {'splits': {preset: {'train': [0, 1, 2, 3]} for preset in batch.PRESETS}}
        protocol = {'source_sha256': {'arena/survival.py': sha(batch.ROOT/'arena/survival.py')}}
        real_arena = Arena

        def tiny_arena(*args, **kwargs):
            # This only bounds fixture trajectories; trusted verify uses the same constructor.
            return real_arena(*args, **kwargs, max_steps=2)

        for balanced in (False, True):
            destination = self.root/('balanced' if balanced else 'legacy')
            kwargs = {'balance_first': True} if balanced else {}
            with patch.object(training, 'Arena', side_effect=tiny_arena), redirect_stdout(io.StringIO()):
                training.collect('train', frozen=(truth, protocol), data_dir=destination, **kwargs)
            with gzip.open(destination/'train_public_replays.jsonl.gz', 'rt', encoding='utf-8') as stream:
                episodes = [json.loads(line) for line in stream]
            self.assertEqual(len(episodes), 16)
            first_counts, swap_counts = {False: 0, True: 0}, {False: 0, True: 0}
            for episode in episodes:
                public_only(episode)
                mi = int(episode['map_id'].rsplit('m', 1)[1])
                learner = episode['learner_seat']
                self.assertEqual(episode['first'], learner ^ (mi % 2) if balanced else learner)
                self.assertEqual(episode['swap'], bool((mi//2) % 2) if balanced else bool(mi % 2))
                first_counts[episode['first'] == learner] += 1
                swap_counts[episode['swap']] += 1
                self.assertEqual(episode['replay']['final_observation']['steps'], 2)
            self.assertEqual(first_counts, {False: 8, True: 8} if balanced else {False: 0, True: 16})
            self.assertEqual(swap_counts, {False: 8, True: 8})
            self.assertEqual(read(destination/'train_collection.json')['balanced_teacher_first'], balanced)

    def test_candidate_matches_live_l3_using_only_detached_public_observations(self):
        with np.load(self.baseline, allow_pickle=False) as data:
            model = {key: data[key].copy() for key in data.files}
        for preset in batch.PRESETS:
            for first in (0, 1):
                env = Arena(7, preset=preset, first=first, swap=bool(first), scoring='survival-v1')
                for _ in range(6):
                    obs = json.loads(json.dumps(env.observe()))
                    public_only(obs)
                    original = deepcopy(obs)
                    expected = policies.choose(obs, 'L3')
                    with patch.object(batch, 'Arena', side_effect=AssertionError('policy cannot create hidden environment')):
                        actual = batch.candidate_decision(obs, model, 'test_candidate', self.baseline_hash)
                    for key in ('action', 'target', 'risk', 'risk_status'):
                        self.assertEqual(actual[key], expected[key])
                    self.assertEqual(actual['diagnostics']['action_logits'], expected['diagnostics']['action_logits'])
                    self.assertEqual(actual['diagnostics']['model_sha256'], expected['diagnostics']['model_sha256'])
                    self.assertEqual(obs, original)
                    env.step(policies.choose(obs, 'B3')['action'])

    def test_paired_effects_bootstrap_base_maps_not_seat_first_variants(self):
        rows = []
        for preset in batch.PRESETS:
            for opponent in batch.OPPONENTS:
                for map_id, variants in (('map_a', [(0, 0), (0, 1), (1, 0), (1, 1)]),
                                         ('map_b', [(0, 0)])):
                    for swap, first in variants:
                        for name in ('L3_previous', 'L3_run01', 'L3_run02', 'L3_run03'):
                            good = (name != 'L3_previous') == (map_id == 'map_a')
                            rows.append({'preset': preset, 'policy0': opponent, 'policy1': name,
                                         'map_id': map_id, 'swap': swap, 'first': first,
                                         'winner': 1 if good else 0, 'gems1': int(good),
                                         'lives1': int(good), 'actions1': int(good),
                                         'normalized_utility1': float(good)})
        effects = batch.paired_effects(rows)
        self.assertEqual(len(effects), 60)
        for effect in effects:
            self.assertEqual(effect['base_maps'], 2)
            self.assertEqual(effect['difference'], 0.)
            self.assertEqual(effect['paired_map_bootstrap_95ci'], [-1., 1.])


if __name__ == '__main__':
    unittest.main()
