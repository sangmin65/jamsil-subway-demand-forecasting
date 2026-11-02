"""Read-only CSV audit. Uses Python standard library; never trains or cleans data."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

FILES = {
    "subway": "종합운동장역_2·9호선_2024-2025_1-12월_시간대별.csv",
    "joined": "종합운동장역_2024-2025_모델입력_경기결합.csv",
    "kbo": "KBO원천데이터_정리.csv",
}
KEY = ("날짜", "호선", "역명", "시간대")
COUNTS = ("승차인원", "하차인원")


def read_table(path):
    payload = path.read_bytes()
    for encoding in ("utf-8-sig", "cp949"):
        try:
            content = payload.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Unsupported encoding: {path.name}")
    rows = list(csv.DictReader(content.splitlines()))
    if not rows or any(None in row or None in row.values() for row in rows):
        raise ValueError(f"Empty or malformed CSV: {path.name}")
    return rows, {
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "encoding": encoding,
        "rows": len(rows),
        "columns": list(rows[0]),
        "empty_cells": {k: sum(r[k] == "" for r in rows) for k in rows[0]},
    }


def audit(source):
    tables, inventory = {}, {}
    for label, name in FILES.items():
        tables[label], inventory[label] = read_table(source / name)
    raw, joined, kbo = (tables[k] for k in FILES)
    key = lambda row: tuple(row[k] for k in KEY)
    index = {key(r): r for r in raw}
    start, end = date(2024, 1, 1), date(2025, 12, 31)
    days = [(start + timedelta(days=i)).isoformat() for i in range((end-start).days+1)]
    expected = {
        (day, line, "종합운동장역", f"{hour:02d}시-{hour+1:02d}시")
        for day in days for line in ("2호선", "9호선") for hour in range(6, 24)
    }
    errors = {
        "raw_duplicate_keys": len(raw) - len(index),
        "raw_missing_expected_keys": len(expected - index.keys()),
        "raw_unexpected_keys": len(index.keys() - expected),
        "invalid_passenger_counts": 0,
        "joined_duplicate_keys": len(joined) - len({key(r) for r in joined}),
        "joined_missing_keys": len(index.keys() - {key(r) for r in joined}),
        "joined_extra_keys": len({key(r) for r in joined} - index.keys()),
        "joined_count_mismatches": 0,
        "lag_mismatches": 0,
    }
    for row in raw:
        for field in COUNTS:
            try:
                value = float(row[field])
                valid = value >= 0 and value.is_integer()
            except ValueError:
                valid = False
            errors["invalid_passenger_counts"] += not valid
    for row in joined:
        original = index.get(key(row))
        errors["joined_count_mismatches"] += bool(
            original and any(row[c] != original[c] for c in COUNTS)
        )
        for lag in (1, 7):
            previous_day = (date.fromisoformat(row["날짜"]) - timedelta(days=lag)).isoformat()
            previous = index.get((previous_day,) + key(row)[1:])
            for prefix, field in zip(("board", "alight"), COUNTS):
                actual = row[f"{prefix}_lag_{lag}d"]
                try:
                    valid = float(actual) == float(previous[field]) if previous else actual == ""
                except ValueError:
                    valid = False
                errors["lag_mismatches"] += not valid
    selected = [r for r in kbo if start <= date.fromisoformat(r["date"]) <= end]
    return {
        "scope": "2024-2025; both supplied lines inspected, model line selection pending",
        "inventory": inventory,
        "structural_errors": errors,
        "kbo": {
            "rows_by_year": dict(Counter(r["season_year"] for r in kbo)),
            "in_scope_rows": len(selected),
            "out_of_scope_rows": len(kbo) - len(selected),
            "in_scope_notes": dict(Counter(r["note"] for r in selected)),
            "multiple_record_dates": {d: n for d, n in Counter(r["date"] for r in selected).items() if n > 1},
        },
        "schedule_flags": {
            name: dict(Counter(r[name] for r in joined))
            for name in ("schedule_record_status", "schedule_completeness_verified", "schedule_snapshot_verified")
        },
        "limitations": [
            "Structural checks do not verify values against official sources.",
            "No listed game is not proof of a non-game day.",
            "Schedule completeness and prediction-time availability require verification.",
            "No cleaning, model choice, split selection or training performed.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve() in {(args.source_dir / n).resolve() for n in FILES.values()}:
        parser.error("Report must not overwrite source data")
    report = audit(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = sum(report["structural_errors"].values())
    print(f"Structural errors: {failures}; report: {args.output}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
