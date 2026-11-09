import copy
import unittest
import numpy as np
from tests.test_setup import fixture
from src.dataset import fit_preprocessor, build_window, read_json
from src.experiment import ROOT
from src.baselines import baseline_predictions


class AblationTests(unittest.TestCase):
    def test_remove_game_preserve_history_calendar_targets(self):
        f = fixture(); p = fit_preprocessor(f)
        cfg = read_json(ROOT / 'configs/initial.json')
        full, names = build_window(f, cfg, p, {'dates': []}, 14)
        cfg['context_mode'] = 'calendar_only'
        reduced, kept = build_window(f, cfg, p, {'dates': []}, 14)
        self.assertEqual(len(kept), 20)
        for split in full:
            for key in ['history', 'history_raw', 'y', 'dates']:
                np.testing.assert_array_equal(full[split][key], reduced[split][key])
            np.testing.assert_array_equal(full[split]['context'][:, :20], reduced[split]['context'])
        changed = f.copy()
        changed['has_kbo_record'] = changed['listed_game_count'] = 1
        changed['scheduled_start_minutes_1'] = 1110
        changed['home_team_1'] = 'UNSEEN'
        other, _ = build_window(changed, cfg, fit_preprocessor(changed), {'dates': []}, 14)
        np.testing.assert_array_equal(other['validation']['context'], reduced['validation']['context'])

    def test_baselines_have_no_target_or_future_leakage(self):
        f = fixture(); cfg = read_json(ROOT / 'configs/initial.json')
        splits, _ = build_window(f, cfg, fit_preprocessor(f), {'dates': []}, 14)
        train, val = splits['train']['dates'], splits['validation']['dates']
        a = baseline_predictions(f, train, val)
        altered = f.copy(); altered.loc[altered.date >= '2025-01-01', '승차인원'] += 1e8
        b = baseline_predictions(altered, train, val)
        np.testing.assert_array_equal(a['weekday_hour_mean'], b['weekday_hour_mean'])
        np.testing.assert_array_equal(a['last_week'][:7], b['last_week'][:7])
        expected = f.loc[f.date == '2024-12-25', ['승차인원','하차인원']].to_numpy().reshape(36)
        np.testing.assert_array_equal(a['last_week'][0], expected)
        self.assertTrue(np.isfinite(a['weekday_hour_mean']).all())


if __name__ == '__main__':
    unittest.main()
