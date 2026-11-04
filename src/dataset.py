"""Build approved features; never fit preprocessing on validation/test dates."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

TARGETS = ["승차인원", "하차인원"]
TEAM_COLS = [f"{side}_team_{slot}" for slot in (1, 2) for side in ("home", "away")]
NO_RECORD = "NO_LISTED_RECORD"
UNKNOWN = "__UNSEEN_TEAM__"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_frame(path):
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame = frame[(frame["호선"] == "2호선") & (frame["역명"] == "종합운동장역")].copy()
    frame["date"] = pd.to_datetime(frame["날짜"], format="%Y-%m-%d")
    frame = frame[frame.date.between("2024-01-01", "2025-12-31")]
    frame["hour_actual"] = frame["시간대"].str.extract(r"^(\d+)시-\d+시$").astype(int)
    frame = frame.sort_values(["date", "hour_actual"]).reset_index(drop=True)
    expected = pd.MultiIndex.from_product([pd.date_range("2024-01-01", "2025-12-31"), range(6, 24)])
    actual = pd.MultiIndex.from_frame(frame[["date", "hour_actual"]])
    if not actual.equals(expected):
        raise ValueError("Expected exactly 731 days x 18 unique hourly records")
    expected_bins = frame.hour_actual.map(lambda h: f"{h:02d}시-{h+1:02d}시")
    if not (expected_bins == frame["시간대"]).all():
        raise ValueError("Unexpected hourly interval")
    values = frame[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Missing, negative or non-finite passenger target")
    return frame


def fit_preprocessor(frame):
    train = frame[frame.date.dt.year == 2024]
    counts = train[TARGETS].to_numpy(dtype=float)
    scale = counts.std(axis=0, ddof=0)
    vocabulary = {}
    for col in TEAM_COLS:
        if train[col].isna().any():
            raise ValueError(f"Missing team category: {col}")
        vocabulary[col] = [NO_RECORD, UNKNOWN] + sorted(set(train[col]) - {NO_RECORD, UNKNOWN})
    return {"mean": counts.mean(axis=0).tolist(), "scale": np.where(scale == 0, 1, scale).tolist(),
            "zero_variance": (scale == 0).tolist(), "teams": vocabulary,
            "fit_period": ["2024-01-01", "2024-12-31"], "fit_rows": len(train),
            "target_order": TARGETS, "history_order": "date, hour, boarding/alighting"}


def make_context(frame, preprocessor, holidays):
    cols = ["has_kbo_record", "listed_game_count"] + TEAM_COLS + [f"scheduled_start_minutes_{s}" for s in (1, 2)]
    if (frame.groupby("date")[cols].nunique(dropna=False) > 1).any().any():
        raise ValueError("Daily context varies across hours")
    daily = frame.groupby("date", sort=True).first()
    names = [f"weekday_{i}" for i in range(7)] + [f"month_{i}" for i in range(1, 13)] + ["is_holiday", "has_kbo_record", "listed_game_count"]
    holiday_dates = {x["date"] for x in holidays["dates"] if x["is_holiday"] == 1}
    blocks = [np.eye(7)[daily.index.weekday], np.eye(12)[daily.index.month - 1],
              np.array([d.strftime("%Y-%m-%d") in holiday_dates for d in daily.index])[:, None],
              daily[["has_kbo_record", "listed_game_count"]].to_numpy(dtype=float)]
    counts = daily.listed_game_count.to_numpy(dtype=float)
    if not np.isfinite(counts).all() or not np.isin(counts, [0, 1, 2]).all():
        raise ValueError("Unsupported game count; review feature capacity")
    if not np.array_equal(daily.has_kbo_record.to_numpy(), (counts > 0).astype(int)):
        raise ValueError("Record flag/count mismatch")
    time_indices = []
    for slot in (1, 2):
        times = pd.to_numeric(daily[f"scheduled_start_minutes_{slot}"], errors="raise")
        present = times.notna().to_numpy()
        if not np.isfinite(times[present]).all() or not times[present].between(0, 1439).all():
            raise ValueError("Invalid scheduled minutes")
        if (present & (counts < slot)).any():
            raise ValueError("Time supplied for absent game record")
        time_indices.append(len(names))
        names += [f"scheduled_minutes_{slot}", f"time_present_{slot}"]
        blocks += [np.column_stack([times.fillna(0), present])]
        for side in ("home", "away"):
            col = f"{side}_team_{slot}"
            vocab = preprocessor["teams"][col]
            values = daily[col]
            if values.isna().any() or ((counts < slot) & (values != NO_RECORD)).any():
                raise ValueError(f"Invalid record absence category: {col}")
            indices = [vocab.index(v) if v in vocab else vocab.index(UNKNOWN) for v in values]
            blocks.append(np.eye(len(vocab))[indices])
            names += [f"{col}={v}" for v in vocab]
    raw = np.concatenate(blocks, axis=1).astype("float32")
    if not np.isfinite(raw).all():
        raise ValueError("Non-finite context")
    scaled = raw.copy()
    scaled[:, time_indices] /= 1440.0
    return raw, scaled, names


def build_window(frame, config, preprocessor, holidays, window):
    if window not in config["windows"]:
        raise ValueError("Unconfigured history window")
    daily = frame[TARGETS].to_numpy(dtype="float32").reshape(731, 18, 2)
    raw_context, context, names = make_context(frame, preprocessor, holidays)
    dates = pd.date_range("2024-01-01", "2025-12-31")
    target_positions = np.arange(14, len(dates))
    histories = np.stack([daily[i-window:i].reshape(window*18, 2) for i in target_positions])
    normalized = ((histories - preprocessor["mean"]) / preprocessor["scale"]).astype("float32")
    target_dates = dates[target_positions].strftime("%Y-%m-%d").to_numpy(dtype=str)
    result = {}
    split_masks = {
        "train": target_dates < "2025-01-01",
        "validation": (target_dates >= "2025-01-01") & (target_dates <= "2025-06-30"),
        "test": (target_dates >= "2025-07-01") & ~np.isin(target_dates, config["excluded_test_dates"]),
    }
    for split, mask in split_masks.items():
        result[split] = {"history": normalized[mask], "history_raw": histories[mask],
                         "context": context[target_positions][mask], "context_raw": raw_context[target_positions][mask],
                         "y": daily[target_positions][mask].reshape(-1, 36), "dates": target_dates[mask]}
    return result, names


def inputs(data, model):
    if model in ("lstm", "cnn"):
        return {"history": data["history"], "context": data["context"]}
    raw = model == "rf"
    history = data["history_raw" if raw else "history"]
    return np.concatenate([history.reshape(len(history), -1), data["context_raw" if raw else "context"]], axis=1)
