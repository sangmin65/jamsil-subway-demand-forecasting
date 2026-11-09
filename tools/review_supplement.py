"""Audit collected supplements without changing training data or labels."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from collect_supplement import OUT, ROOT, save


def read(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def counts(records, fields):
    return [{'group': list(key), 'count': count} for key, count in
            sorted(Counter(tuple(r[f] for f in fields) for r in records).items())]


def main():
    games, requests = read('kbo_records.json'), read('kbo_requests.json')
    events, pages = read('seoul_events.json'), read('seoul_requests.json')
    source = ROOT.parent / '인공지능공학 과제용/KBO원천데이터_정리.csv'
    with source.open(encoding='utf-8-sig', newline='') as stream:
        old = [r for r in csv.DictReader(stream) if r['date'][:4] in ('2024', '2025')]
    fields = ['date', 'scheduled_start_time', 'stadium', 'away_team', 'home_team']
    key = lambda r: tuple(r[f] for f in fields)
    before, after = {key(r): r for r in old}, {key(r): r for r in games}
    assert len(before) == len(old) and len(after) == len(games), 'Duplicate game keys'
    assert len(requests) == 72 and not any(r['short_row_indices'] for r in requests)
    assert len({(r['year'], r['month']) for r in pages}) == 24
    for year, month in {(r['year'], r['month']) for r in pages}:
        group = [r for r in pages if (r['year'], r['month']) == (year, month)]
        ids = [i for r in group for i in r['ids']]
        assert len(ids) == len(set(ids)) == group[0]['total']
    assert len(events) == len({r['id'] for r in events})
    assert all(r['title'] and r['venue'] and r['start_date'] <= r['end_date'] for r in events)
    meta_files = list(OUT.rglob('*.meta.json'))
    for meta_path in meta_files:
        meta = json.loads(meta_path.read_text(encoding='utf-8'))
        raw_path = Path(str(meta_path).removesuffix('.meta.json'))
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == meta['sha256'], raw_path
    additions = [after[k] for k in sorted(after.keys() - before.keys())]
    absent = [{f: before[k][f] for f in fields + ['note']} for k in sorted(before.keys() - after.keys())]
    changes = [dict(key=dict(zip(fields, k)), old_note=before[k]['note'], current_note=after[k]['note'])
               for k in sorted(before.keys() & after.keys()) if before[k]['note'] != after[k]['note']]
    for r in games:
        r['year'] = r['date'][:4]
    for r in events:
        r['year'] = r['start_date'][:4]
    non_baseball_venue = [r for r in events if r['venue'] != '잠실야구장']
    result = dict(
        collection_date='2026-11-09', original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        verified_source_hashes=len(meta_files), kbo_request_count=len(requests),
        kbo_current_records=len(games), kbo_original_records=len(old),
        kbo_by_year_series=counts(games, ['year', 'series']),
        matching_keys=len(before.keys() & after.keys()), additions=additions,
        original_only_preserve=absent, changed_notes_review=changes,
        seoul_month_count=24, seoul_page_count=len(pages), seoul_unique_listings=len(events),
        seoul_by_year_venue=counts(events, ['year', 'venue']),
        non_baseball_venue_listings=len(non_baseball_venue),
        non_baseball_venue_by_year_category=counts(non_baseball_venue, ['year', 'category']),
        multi_date_listings=sum(r['start_date'] != r['end_date'] for r in events),
        limitations=[
            'Current official pages are retrospective; prediction-time availability remains unverified.',
            'All monthly responses were collected; this does not prove historical event completeness.',
            'Absent records do not imply no event. Preserve original-only cancellation records.',
            'Venue-filtered listing counts are not performance counts or distinct event dates.',
            'Date ranges and time text need review before daily expansion; setup days may be included.',
            'No original training inputs, note labels, split, model, or final evaluation were changed.',
        ])
    destination = ROOT / 'reports/metrics/supplement_review_2026-11-09.json'
    save(destination, result)
    print(json.dumps({k: result[k] for k in ['kbo_current_records', 'matching_keys',
                     'seoul_unique_listings', 'non_baseball_venue_listings', 'verified_source_hashes']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
