"""Compare archived schedule responses with delivered CSV without executing notebook."""
import argparse
import csv
import hashlib
import html
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path


def clean(value):
    return html.unescape(re.sub(r'<[^>]*>', '', value or '')).strip()


def audit(archive, source):
    fields = ['date', 'scheduled_start_time', 'stadium', 'note', 'away_team', 'home_team']
    with source.open(encoding='utf-8-sig', newline='') as stream:
        delivered = [r for r in csv.DictReader(stream) if r['season_year'] in ('2024', '2025')]
    expected = {f'raw_schedule_json/{y}_{m:02d}.json' for y in (2024, 2025) for m in range(3, 11)}
    parsed, months = [], []
    with zipfile.ZipFile(archive) as z:
        corrupt = z.testzip()
        missing = sorted(expected - set(z.namelist()))
        logs = {n: z.read(n).decode('utf-8-sig').strip() for n in ('schedule_errors.csv', 'detail_errors.csv') if n in z.namelist()}
        for name in sorted(expected & set(z.namelist())):
            year = name.split('/')[-1][:4]
            data = json.loads(z.read(name))
            current, count, skipped = None, 0, 0
            for item in data.get('rows', []):
                cells = item.get('row', [])
                if cells and cells[0].get('Class') == 'day':
                    current = clean(cells[0]['Text'])
                    cells = cells[1:]
                if len(cells) < 8:
                    skipped += 1
                    continue
                if '잠실' not in clean(cells[6]['Text']):
                    continue
                match = re.search(r'(\d{2})\.(\d{2})', current or '')
                if not match:
                    raise ValueError(f'Missing date in {name}')
                play = cells[1]['Text']
                spans = re.findall(r'<span[^>]*>(.*?)</span>', play, re.S)
                if len(spans) < 2:
                    raise ValueError(f'Missing teams in {name}')
                parsed.append(dict(zip(fields, [f'{year}-{match[1]}-{match[2]}', clean(cells[0]['Text']), clean(cells[6]['Text']), clean(cells[7]['Text']), clean(spans[0]), clean(spans[-1])])))
                count += 1
            months.append({'file':name, 'response_rows':len(data.get('rows', [])), 'jamsil_rows':count, 'short_rows':skipped})
        archived = list(csv.DictReader(io.StringIO(z.read('kbo_jamsil_all_columns.csv').decode('utf-8-sig'))))
    key = lambda r: tuple(r[f] for f in fields)
    a, b = Counter(map(key, parsed)), Counter(map(key, delivered))
    shared = [k for k in delivered[0] if k in archived[0]]
    c = Counter(tuple(r[k] for k in shared) for r in delivered)
    d = Counter(tuple(r[k] for k in shared) for r in archived if r['season_year'] in ('2024', '2025'))
    return {'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'csv_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'corrupt_member':corrupt, 'missing_month_files':missing, 'error_logs':logs, 'months':months,
            'raw_schedule_rows':len(parsed), 'delivered_rows':len(delivered),
            'raw_only_records':[{'key':list(k),'count':v} for k,v in (a-b).items()],
            'csv_only_records':[{'key':list(k),'count':v} for k,v in (b-a).items()],
            'archive_vs_delivered_shared_column_difference':sum((c-d).values())+sum((d-c).values()),
            'limitations':['Archived-response consistency is not proof of official schedule completeness.',
                            'Blank logs may cover only the last run because notebook overwrites logs.',
                            'No preseason, pre-prediction snapshot or rescheduling linkage verified.',
                            'gamecenter_status excluded from evidence.']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--csv', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.resolve() in (args.archive.resolve(), args.csv.resolve()):
        p.error('Do not overwrite inputs')
    result = audit(args.archive, args.csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
