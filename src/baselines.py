"""Validation-only baselines with explicit historical information boundaries."""
import numpy as np
import pandas as pd
from .dataset import TARGETS


def baseline_predictions(frame, train_dates, validation_dates):
    daily_dates = pd.DatetimeIndex(sorted(frame.date.unique()))
    values = frame[TARGETS].to_numpy(dtype=float).reshape(len(daily_dates), 36)
    train_positions = daily_dates.get_indexer(pd.to_datetime(train_dates))
    val_positions = daily_dates.get_indexer(pd.to_datetime(validation_dates))
    if (train_positions < 0).any() or (val_positions < 7).any():
        raise ValueError("Missing baseline history")
    if not (daily_dates[train_positions] < pd.Timestamp('2025-01-01')).all():
        raise ValueError("Mean baseline must use training dates only")
    weekday_means = np.stack([values[train_positions[daily_dates[train_positions].weekday == w]].mean(axis=0)
                              for w in range(7)])
    return {'last_week': values[val_positions - 7],
            'weekday_hour_mean': weekday_means[daily_dates[val_positions].weekday]}
