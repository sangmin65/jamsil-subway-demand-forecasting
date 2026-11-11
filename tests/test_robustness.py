"""Information-boundary regression tests for the follow-up temporal audit."""
import unittest
import numpy as np
import pandas as pd
from tools.verify_robustness import arrays, select, ROOT
from src.dataset import TARGETS, TEAM_COLS
from src.evaluation import metrics


class BoundaryTests(unittest.TestCase):
    def setUp(self):
        dates = pd.date_range('2024-01-01', '2024-03-31')
        self.frame = pd.DataFrame({'date': np.repeat(dates, 18)})
        self.frame['승차인원'] = np.arange(len(self.frame)) + 1
        self.frame['하차인원'] = np.arange(len(self.frame)) + 2
        self.frame['has_kbo_record'] = 0
        self.frame['listed_game_count'] = 0
        for col in TEAM_COLS:
            self.frame[col] = 'NO_LISTED_RECORD'
        for slot in (1, 2):
            self.frame[f'scheduled_start_minutes_{slot}'] = np.nan

    def test_future_demand_does_not_change_fit_or_same_day_inputs(self):
        original, pre, names = arrays(self.frame, '2024-01-31', 14)
        changed = self.frame.copy()
        changed.loc[changed.date >= '2024-02-15', TARGETS] += 100000
        altered, new_pre, new_names = arrays(changed, '2024-01-31', 14)
        self.assertEqual(pre, new_pre)
        self.assertEqual(names, new_names)
        before = select(original, '2024-01-15', '2024-02-15')
        after = select(altered, '2024-01-15', '2024-02-15')
        for key in ('history', 'history_raw', 'context', 'context_raw', 'dates'):
            np.testing.assert_array_equal(before[key], after[key])
        self.assertFalse(np.array_equal(before['y'][-1], after['y'][-1]))

    def test_future_team_does_not_enter_training_vocabulary(self):
        changed = self.frame.copy()
        changed.loc[changed.date >= '2024-02-01', 'listed_game_count'] = 1
        changed.loc[changed.date >= '2024-02-01', 'has_kbo_record'] = 1
        changed.loc[changed.date >= '2024-02-01', 'home_team_1'] = 'FUTURE_ONLY'
        d, pre, names = arrays(changed, '2024-01-31', 3)
        self.assertNotIn('FUTURE_ONLY', pre['teams']['home_team_1'])
        i = names.index('home_team_1=__UNSEEN_TEAM__')
        self.assertTrue((select(d, '2024-02-01', '2024-03-31')['context'][:, i] == 1).all())

    def test_score_output_direction_and_clipping(self):
        y = np.full((2, 36), 10.)
        p = y.copy(); p[:, ::2] = -2.; p[:, 1::2] = 14.
        scored = metrics(y, p)
        self.assertEqual(scored['board_mae'], 10)
        self.assertEqual(scored['alight_mae'], 4)
        self.assertEqual(scored['mae'], 7)


if __name__ == '__main__':
    unittest.main()
