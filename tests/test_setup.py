"""No model training: synthetic leakage, alignment and stopping-policy checks."""
import copy
import unittest

import numpy as np
import pandas as pd

from src.dataset import fit_preprocessor, build_window, make_context, read_json
from src.evaluation import metrics, StopTracker
from src.experiment import ROOT, require_review


def fixture():
    dates = pd.date_range("2024-01-01", "2025-12-31")
    n = len(dates)*18
    df = pd.DataFrame({"date": dates.repeat(18), "승차인원": np.arange(n)+100,
                       "하차인원": np.arange(n)+200, "has_kbo_record": 0, "listed_game_count": 0})
    for slot in (1, 2):
        df[f"scheduled_start_minutes_{slot}"] = np.nan
        for side in ("home", "away"):
            df[f"{side}_team_{slot}"] = "NO_LISTED_RECORD"
    return df


class SetupTests(unittest.TestCase):
    def setUp(self):
        self.frame = fixture()
        self.config = read_json(ROOT / "configs/initial.json")
        self.holidays = {"dates": []}

    def test_future_cannot_change_fitted_statistics_or_vocabulary(self):
        original = fit_preprocessor(self.frame)
        future = self.frame.date.dt.year == 2025
        self.frame.loc[future, "승차인원"] = 1e9
        self.frame.loc[future, "home_team_1"] = "FUTURE_ONLY"
        self.assertEqual(original, fit_preprocessor(self.frame))

    def test_alignment_common_dates_and_history_boundaries(self):
        pre = fit_preprocessor(self.frame)
        common_dates = None
        for window in (3, 7, 14):
            splits, _ = build_window(self.frame, self.config, pre, self.holidays, window)
            self.assertEqual([len(splits[s]["y"]) for s in ("train", "validation", "test")], [352, 181, 178])
            days = {s: d["dates"].tolist() for s, d in splits.items()}
            if common_dates is not None:
                self.assertEqual(common_dates, days)
            common_dates = days
            for split in splits.values():
                for i, date in enumerate(split["dates"]):
                    start = (pd.Timestamp(str(date))-pd.Timestamp("2024-01-01")).days*18
                    self.assertEqual(split["history_raw"][i, -1, 0], start-1+100)
                    self.assertEqual(split["history_raw"][i, 0, 0], start-window*18+100)
                    self.assertEqual(split["y"][i, 0], start+100)
                    self.assertEqual(split["y"][i, 1], start+200)
            self.assertFalse(set(days["test"]) & set(self.config["excluded_test_dates"]))

    def test_unseen_and_no_record_are_distinct(self):
        pre = fit_preprocessor(self.frame)
        future_day = self.frame.date == pd.Timestamp("2025-01-01")
        self.frame.loc[future_day, ["has_kbo_record", "listed_game_count"]] = 1
        self.frame.loc[future_day, "home_team_1"] = "FUTURE_ONLY"
        self.frame.loc[future_day, "away_team_1"] = "FUTURE_OPPONENT"
        _, context, names = make_context(self.frame, pre, self.holidays)
        unknown = names.index("home_team_1=__UNSEEN_TEAM__")
        absent = names.index("home_team_1=NO_LISTED_RECORD")
        self.assertEqual(context[366, unknown], 1)
        self.assertEqual(context[366, absent], 0)
        self.assertEqual(context[0, absent], 1)

    def test_clip_metrics_and_equal_target_weights(self):
        y = np.ones((2, 36))*10
        p = y.copy()
        p[:, ::2] = -2
        result = metrics(y, p)
        self.assertEqual(result["mae"], 5)
        self.assertEqual(result["raw_mae"], 6)
        self.assertEqual(result["negative_fraction"], 0.5)

    def test_strict_improvement_ties_and_wait_reset(self):
        stop = StopTracker(2, 0)
        self.assertEqual(stop.observe(10, 1), (True, False))
        self.assertEqual(stop.observe(10, 2), (False, False))
        self.assertEqual(stop.observe(9, 3), (True, False))
        self.assertEqual(stop.observe(10, 4), (False, False))
        self.assertEqual(stop.observe(9, 5), (False, True))
        self.assertEqual(stop.best_epoch, 3)

    def test_review_guard(self):
        with self.assertRaises(ValueError):
            require_review(self.config)
        approved = copy.deepcopy(self.config)
        approved["team_review"] = {"approved": True, "reviewer": "test fixture", "reviewed_at": "test fixture"}
        require_review(approved)


if __name__ == "__main__":
    unittest.main()
