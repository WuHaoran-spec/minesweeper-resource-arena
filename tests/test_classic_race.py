from copy import deepcopy
import random
import unittest
from arena.classic_race import ClassicRace, choose, prepare


class ClassicRaceTests(unittest.TestCase):
    def test_same_map_same_protected_center_independent_boards(self):
        race=ClassicRace(31,preset='expert',first=1)
        self.assertEqual(race.automatic_start,[8,15])
        self.assertEqual(race._boards[0]._mines,race._boards[1]._mines)
        obs=race.observe()
        self.assertEqual(obs['boards'][0]['revealed'],obs['boards'][1]['revealed'])
        self.assertEqual(obs['active_reveals'],[0,0])
        self.assertEqual(obs['turn'],1)
        obs['boards'][0]['revealed'].clear()
        self.assertTrue(race.observe()['boards'][0]['revealed'])

    def test_unknown_actual_mines_are_legal_and_terminal_full_mines_scrubbed(self):
        race=ClassicRace(9);mine=list(sorted(race._boards[0]._mines)[0])
        self.assertIn(mine,race.current_observation()['legal_cells'])
        after=race.step(mine)
        self.assertFalse(after['done']);self.assertEqual(after['turn'],1)
        visible=[item for item in after['boards'][0]['revealed'] if item[2]<0]
        self.assertEqual(visible,[mine+[-1]])
        self.assertEqual(len(race._boards[0]._mines),10)
        self.assertTrue(after['boards'][0]['done']);self.assertFalse(after['boards'][1]['done'])
        with self.assertRaises(ValueError):choose(after['boards'][0],'R1')

    def test_losing_board_skipped_survivor_can_finish_and_rank_has_no_wall_clock(self):
        race=ClassicRace(2);race.step(list(sorted(race._boards[0]._mines)[0]),computation_seconds=0.)
        while not race.done:
            self.assertEqual(race.turn,1)
            board=race._boards[1]
            cell=next(cell for cell in race.current_observation()['legal_cells'] if tuple(cell) not in board._mines)
            race.step(cell,computation_seconds=100.)
        obs=race.observe();self.assertEqual(obs['winner'],1)
        self.assertTrue(obs['results'][1]['cleared']);self.assertEqual(obs['results'][1]['safe_revealed'],71)
        self.assertGreater(obs['computation_seconds'][1],obs['computation_seconds'][0])
        self.assertEqual(len(race.save_replay()['records']),sum(obs['active_reveals']))
        with self.assertRaises(ValueError):race.step([0,0])

    def test_invalid_actions_atomic_and_private_metadata_refused(self):
        race=ClassicRace(3);before=race.observe()
        for cell in ([-1,0],[9,0],race.automatic_start,[True,0]):
            with self.assertRaises(ValueError):race.step(cell)
            self.assertEqual(before,race.observe())
        with self.assertRaises(ValueError):race.step(before['boards'][0]['legal_cells'][0],{'seed':3})
        self.assertEqual(before,race.observe())

    def test_board_only_policy_and_public_legal_cells(self):
        race=ClassicRace(4)
        with self.assertRaises(ValueError):prepare(race.observe())
        obs=race.current_observation();bad=deepcopy(obs);bad['legal_cells'].pop()
        with self.assertRaises(ValueError):prepare(bad)
        for policy in ('R0','R1'):
            choice=choose(obs,policy,random.Random(5))
            self.assertIn(choice['cell'],obs['legal_cells'])
            self.assertEqual(len(choice['risk']),81)


if __name__=='__main__':unittest.main()
