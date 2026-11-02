"""Summarize note values without treating GameCenter status as evidence."""
import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path


def note_category(note):
    if '취소' in note:
        return 'explicit_cancellation'
    if note == '-':
        return 'no_special_note'
    return 'other_note_unknown'


def summarize(path):
    by_day = defaultdict(list)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['season_year'] in ('2024', '2025') and row['stadium'] == '잠실':
                by_day[row['date']].append(row['note'])
    report = {'basis':'note only; gamecenter_status excluded; dash does not prove completion', 'splits':{}}
    for name, first, last in [('train','2024-01-01','2024-12-31'), ('validation','2025-01-01','2025-06-30'), ('test','2025-07-01','2025-12-31')]:
        daily, record_notes = Counter(), Counter()
        day = date.fromisoformat(first)
        while day <= date.fromisoformat(last):
            notes = by_day[day.isoformat()]
            categories = sorted({note_category(n) for n in notes})
            daily[' + '.join(categories) if categories else 'no_record_unknown'] += 1
            record_notes.update(notes)
            day += timedelta(days=1)
        report['splits'][name] = {'daily_status':dict(daily),'record_notes':dict(record_notes)}
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    if args.source.resolve() == args.output.resolve():
        p.error('Do not overwrite source')
    result = summarize(args.source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
