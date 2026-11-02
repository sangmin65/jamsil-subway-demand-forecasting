"""Count dates for a proposed split; does not create training data or approve it."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

EXPLORED = {"2025-09-25", "2025-09-26", "2025-09-27", "2025-09-28", "2025-09-30", "2025-10-01"}
SPLITS = {"train": ("2024-01-01", "2024-12-31"), "validation": ("2025-01-01", "2025-06-30"), "test": ("2025-07-01", "2025-12-31")}


def summarize(path):
    records = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["stadium"] == "잠실":
                records[row["date"]].append(row)
    result = {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "status": "proposal_only", "splits": {}, "monthly": {}}
    for label, (start, end) in SPLITS.items():
        counts, effective = Counter(), Counter()
        current = date.fromisoformat(start)
        while current <= date.fromisoformat(end):
            day = current.isoformat()
            rows = records[day]
            played = [r for r in rows if r["away_score"] and r["home_score"] and r["actual_start_time"]]
            other = [r for r in rows if r not in played]
            status = ("played_and_other" if played and other else "played_only" if played
                      else "other_only" if rows else "no_record_unknown")
            # Calendar proxies for coverage review, NOT official season labels.
            band = ("winter_months_proxy" if current.month in (1, 2, 11, 12)
                    else "march_transition" if current.month == 3 else "apr_oct_proxy")
            for field in ("days", status, band):
                counts[field] += 1
            month = result["monthly"].setdefault(day[:7], {})
            month[status] = month.get(status, 0) + 1
            # Common dates for 14-day input; prior-split history is allowed.
            if current >= date(2024, 1, 15) and day not in EXPLORED:
                for field in ("days", status, band):
                    effective[field] += 1
            current += timedelta(days=1)
        assert sum(counts[s] for s in ("played_only", "played_and_other", "other_only", "no_record_unknown")) == counts["days"]
        result["splits"][label] = {"start": start, "end": end,
                                   "calendar": dict(counts), "common_dates_proposal": dict(effective)}
    result["limitations"] = [
        "Score/start fields indicate played records in supplied CSV, not independent official verification.",
        "Other-only days include cancellation or other statuses; no-record days remain unknown.",
        "Calendar month proxies are not verified season/offseason labels or model features.",
        "Common-date counts assume complete subway history, no additional filtering, and six explored dates excluded.",
    ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kbo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kbo.resolve() == args.output.resolve():
        parser.error("Output must not overwrite input")
    report = summarize(args.kbo)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["splits"], ensure_ascii=False, indent=2))
