"""Original-unit metrics with explicit clipping policy and sample counts."""
import numpy as np


def metrics(y, prediction):
    y = np.asarray(y, dtype=float).reshape(-1, 18, 2)
    raw = np.asarray(prediction, dtype=float).reshape(y.shape)
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite predictions")
    errors = np.maximum(raw, 0) - y
    mae = np.abs(errors).mean(axis=(0, 1))
    rmse = np.sqrt(np.square(errors).mean(axis=(0, 1)))
    return {"days": len(y), "hourly_records": len(y)*18,
            "mae": float(mae.mean()), "board_mae": float(mae[0]), "alight_mae": float(mae[1]),
            "board_rmse": float(rmse[0]), "alight_rmse": float(rmse[1]),
            "raw_mae": float(np.abs(raw-y).mean()), "negative_fraction": float((raw < 0).mean())}


def grouped_metrics(y, prediction, dates, statuses):
    dates = np.asarray(dates)
    groups = {"month": np.array([d[:7] for d in dates]),
              "note_status": np.array([statuses.get(d, "no_record_unknown") for d in dates])}
    report = {"overall": metrics(y, prediction)}
    for group, labels in groups.items():
        report[group] = {label: metrics(y[labels == label], prediction[labels == label]) for label in sorted(set(labels))}
    return report


class StopTracker:
    """Strict improvement, first winner on ties; independent of library defaults."""
    def __init__(self, patience, min_delta):
        self.patience, self.min_delta = patience, min_delta
        self.best, self.best_epoch, self.wait = float("inf"), None, 0

    def observe(self, value, epoch):
        if not np.isfinite(value):
            raise ValueError("Non-finite validation MAE")
        improved = value < self.best - self.min_delta
        if improved:
            self.best, self.best_epoch, self.wait = value, epoch, 0
        else:
            self.wait += 1
        return improved, self.wait >= self.patience
