"""CLI: preflight builds models without training; train requires recorded team review."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import load_frame, fit_preprocessor, build_window, inputs, read_json, sha256
from .models import build_rf, build_neural
from .evaluation import metrics, grouped_metrics, StopTracker
from .summarize_note_status import note_category

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def require_review(config):
    review = config["team_review"]
    if review.get("approved") is not True or not review.get("reviewer") or not review.get("reviewed_at"):
        raise ValueError("Training blocked: record actual team review in config first")


def environment():
    import tensorflow as tf
    return {"python": platform.python_version(), "platform": platform.platform(),
            "devices": [{"name": d.name, "type": d.device_type} for d in tf.config.list_physical_devices()],
            "packages": {p: importlib.metadata.version(p) for p in ("tensorflow", "keras", "scikit-learn", "pandas", "numpy")},
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}


def note_statuses(path):
    rows = pd.read_csv(path, keep_default_na=False)
    categories = defaultdict(set)
    for row in rows.to_dict("records"):
        if str(row["season_year"]) in ("2024", "2025") and row["stadium"] == "잠실":
            categories[row["date"]].add(note_category(row["note"]))
    return {day: " + ".join(sorted(values)) for day, values in categories.items()}


def predict_neural(model, x, batch_size):
    size = len(next(iter(x.values()))) if isinstance(x, dict) else len(x)
    result = []
    for start in range(0, size, batch_size):
        batch = {k: v[start:start+batch_size] for k, v in x.items()} if isinstance(x, dict) else x[start:start+batch_size]
        result.append(model(batch, training=False).numpy())
    return np.concatenate(result)


def train_neural(model, train, validation, name, config, out):
    from tensorflow import keras
    x_train, x_val = inputs(train, name), inputs(validation, name)
    settings = config["training"]
    tracker = StopTracker(settings["patience"], settings["min_delta"])
    schedule = config.get("lr_plateau")
    reducer = None
    if schedule is not None:
        if not (0 < schedule["patience"] < settings["patience"] and
                0 < schedule["factor"] < 1 and schedule["min_lr"] > 0 and schedule["min_delta"] >= 0):
            raise ValueError("Invalid plateau schedule or patience must be below early stopping patience")
        reducer = keras.callbacks.ReduceLROnPlateau(monitor="val_clipped_mae", mode="min", **schedule)
        reducer.set_model(model)
        reducer.on_train_begin()

    class RecordValidation(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.rows, self.reason = [], "maximum_epochs"

        def on_train_batch_end(self, batch, logs=None):
            if not np.isfinite(logs["loss"]):
                raise ValueError("Non-finite training loss")

        def on_epoch_end(self, epoch, logs=None):
            pred = predict_neural(self.model, x_val, settings["batch_size"])
            scored = metrics(validation["y"], pred)
            improved, stop = tracker.observe(scored["mae"], epoch + 1)
            if improved:
                self.model.save_weights(out / "best.weights.h5")
            rate = float(self.model.optimizer.learning_rate.numpy())
            if reducer is not None:
                reducer.on_epoch_end(epoch, {"val_clipped_mae": scored["mae"]})
            next_rate = float(self.model.optimizer.learning_rate.numpy())
            train_mae = logs["loss"] if config.get("loss", "mean_absolute_error") == "mean_absolute_error" else logs["train_mae"]
            self.rows.append({"epoch": epoch+1, "train_loss": float(logs["loss"]), "train_mae": float(train_mae), **scored,
                              "learning_rate": rate, "next_learning_rate": next_rate,
                              "lr_reduced": next_rate < rate,
                              "optimizer_steps": int(self.model.optimizer.iterations.numpy())})
            save_json(out / "learning_curve.json", self.rows)
            if stop:
                self.reason = "early_stopping"
                self.model.stop_training = True

    callback = RecordValidation()
    model.fit(x_train, train["y"], epochs=settings["epochs"], batch_size=settings["batch_size"],
              shuffle=settings["shuffle"], callbacks=[callback], verbose=2)
    model.load_weights(out / "best.weights.h5")
    model.save(out / "model.keras")
    return predict_neural(model, x_val, settings["batch_size"]), {
        "best_epoch": tracker.best_epoch, "stopped_epoch": len(callback.rows), "stop_reason": callback.reason,
        "parameters": model.count_params(),
        "optimizer": config.get("optimizer", "adam"),
        "loss": config.get("loss", "mean_absolute_error"),
        "huber_delta": config.get("huber_delta"),
        "lr_reductions": sum(row["lr_reduced"] for row in callback.rows),
        "best_learning_rate": callback.rows[tracker.best_epoch-1]["learning_rate"],
        "total_optimizer_steps": callback.rows[-1]["optimizer_steps"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "train"])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/initial.json")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--kbo-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory; existing runs never overwritten")
    args = parser.parse_args()
    config = read_json(args.config)
    if args.mode == "train":
        require_review(config)
    args.output.mkdir(parents=True, exist_ok=False)
    save_json(args.output / "config.json", config)
    try:
        run(args, config)
    except Exception as error:
        save_json(args.output / "failure.json", {"type": type(error).__name__, "message": str(error)})
        raise


def run(args, config):
    frame = load_frame(args.source)
    preprocessor = fit_preprocessor(frame)
    holidays_path = ROOT / "data/reference/holidays_2024_2025.json"
    holidays = read_json(holidays_path)
    statuses = note_statuses(args.kbo_source)
    report = {"mode": args.mode, "training_executed": False, "created_utc": datetime.now(timezone.utc).isoformat(),
              "environment": environment(), "sources": {"joined_sha256": sha256(args.source),
              "kbo_sha256": sha256(args.kbo_source), "holidays_sha256": sha256(holidays_path), "config_sha256": sha256(args.config)},
              "code_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "src").glob("*.py"))},
              "dependency_lock_sha256": sha256(ROOT / "requirements-lock.txt"),
              "schedule_warning": "Retrospective schedules; completeness and pre-forecast availability unverified",
              "conditions": []}
    save_json(args.output / "preprocessor.json", preprocessor)
    for window in config["windows"]:
        splits, context_names = build_window(frame, config, preprocessor, holidays, window)
        if [len(splits[s]["y"]) for s in ("train", "validation", "test")] != [352, 181, 178]:
            raise ValueError("Unexpected split sizes")
        for name in config["models"]:
            out = args.output / f"{name}_{window}d"
            out.mkdir()
            entry = {"model": name, "window": window, "counts": {s: len(d["y"]) for s, d in splits.items()},
                     "context_features": len(context_names)}
            save_json(out / "context_features.json", context_names)
            save_json(out / "target_dates.json", {s: d["dates"].tolist() for s, d in splits.items()})
            started = time.perf_counter()
            model = build_rf(config) if name == "rf" else build_neural(name, window, len(context_names), config)
            if name == "rf":
                save_json(out / "model_config.json", model.get_params())
            else:
                (out / "model_config.json").write_text(model.to_json(), encoding="utf-8")
                entry["parameters"] = model.count_params()
                entry["weight_devices"] = sorted({str(v.value.device) for v in model.trainable_weights})
                if config.get("require_gpu") and not all("GPU:" in d for d in entry["weight_devices"]):
                    raise RuntimeError("GPU required, but neural weights are not all on GPU")
            if args.mode == "preflight":
                # Only synthetic zero inputs, no fit, no real validation/test predictions.
                if name != "rf":
                    x = {"history": np.zeros((2, window*18, 2), "float32"), "context": np.zeros((2, len(context_names)), "float32")} if name in ("lstm", "cnn") else np.zeros((2, window*36+len(context_names)), "float32")
                    prediction = predict_neural(model, x, 16)
                    if prediction.shape != (2, 36) or not np.isfinite(prediction).all():
                        raise ValueError("Forward shape or finite check failed")
                    entry["synthetic_forward_shape"] = list(prediction.shape)
            else:
                if name == "rf":
                    import joblib
                    model.fit(inputs(splits["train"], name), splits["train"]["y"])
                    prediction = model.predict(inputs(splits["validation"], name))
                    joblib.dump(model, out / "model.joblib")
                else:
                    prediction, details = train_neural(model, splits["train"], splits["validation"], name, config, out)
                    entry.update(details)
                scored = grouped_metrics(splits["validation"]["y"], prediction, splits["validation"]["dates"], statuses)
                save_json(out / "validation_metrics.json", scored)
                np.savez_compressed(out / "validation_predictions.npz", dates=splits["validation"]["dates"],
                                    truth=splits["validation"]["y"], raw=prediction, prediction=np.maximum(prediction, 0))
                entry["validation_mae"] = scored["overall"]["mae"]
                report["training_executed"] = True
            entry["seconds"] = time.perf_counter() - started
            report["conditions"].append(entry)
            save_json(args.output / "report.json", report)
    if args.mode == "train":
        report["ranking"] = sorted(report["conditions"], key=lambda item: item["validation_mae"])
        report["final_test_evaluated"] = False
        save_json(args.output / "report.json", report)
    print(json.dumps({"mode": args.mode, "conditions": len(report["conditions"]), "training_executed": report["training_executed"]}))


if __name__ == "__main__":
    main()
