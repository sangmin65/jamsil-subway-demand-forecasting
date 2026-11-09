"""Collect public 2024-2025 KBO schedules and Seoul stadium listings with provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/external/supplement_2026-11-09'
KBO = 'https://www.koreabaseball.com/ws/Schedule.asmx/GetScheduleList'
SEOUL = 'https://func.seoul.go.kr/stadium/'
SERIES = {'preseason': '1', 'regular': '0,9,6', 'postseason': '3,4,5,7'}


def clean(value):
    return ' '.join(html.unescape(re.sub(r'<[^>]*>', ' ', value or '')).split())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def fetch(url, filename, data=None):
    path = OUT / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = path.with_suffix(path.suffix + '.meta.json')
    if path.exists() and meta_path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, data=urllib.parse.urlencode(data).encode() if data else None,
                                     headers={'User-Agent': 'Mozilla/5.0', 'Referer': 'https://www.koreabaseball.com/Schedule/Schedule.aspx' if data else url})
    with urllib.request.urlopen(request, timeout=35) as response:
        raw = response.read()
        meta = dict(url=url, request_body=data, http_status=response.status,
                    retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(raw).hexdigest())
    path.write_bytes(raw)
    save(meta_path, meta)
    time.sleep(0.25)
    return raw


def baseball():
    games, requests = [], []
    for year in (2024, 2025):
        for series, codes in SERIES.items():
            for month in range(1, 13):
                payload = dict(leId='1', srIdList=codes, seasonId=str(year), gameMonth=f'{month:02}', teamId='')
                filename = f'kbo/{year}_{month:02}_{series}.json'
                data = json.loads(fetch(KBO, filename, payload))
                assert isinstance(data.get('rows'), list), filename
                current = None
                matches = 0
                skipped = []
                for index, item in enumerate(data['rows']):
                    cells = item.get('row', [])
                    if cells and cells[0].get('Class') == 'day':
                        current = clean(cells[0]['Text'])
                        cells = cells[1:]
                    if len(cells) < 8:
                        skipped.append(index)
                        continue
                    venue = clean(cells[6]['Text'])
                    if '잠실' not in venue:
                        continue
                    date = re.search(r'(\d{2})\.(\d{2})', current or '')
                    teams = re.findall(r'<span[^>]*>(.*?)</span>', cells[1]['Text'], re.S)
                    assert date and len(teams) >= 2, filename
                    links = re.findall(r"href=['\"]([^'\"]*GameCenter[^'\"]*)", ' '.join(c['Text'] for c in cells))
                    games.append(dict(date=f'{year}-{date[1]}-{date[2]}', series=series,
                                      scheduled_start_time=clean(cells[0]['Text']), stadium=venue,
                                      away_team=clean(teams[0]), home_team=clean(teams[-1]),
                                      note=clean(cells[7]['Text']),
                                      gamecenter_url=urllib.parse.urljoin('https://www.koreabaseball.com',links[0].replace('&amp;', '&')) if links else None,
                                      source_url=KBO, raw_file=filename, row_index=index,
                                      preforecast_availability_verified=False))
                    matches += 1
                requests.append(dict(year=year, month=month, series=series, response_rows=len(data['rows']),
                                     jamsil_records=matches, short_row_indices=skipped, raw_file=filename))
                print(year, month, series, matches, flush=True)
    save(OUT / 'kbo_records.json', games)
    save(OUT / 'kbo_requests.json', requests)


def field(fragment, classname):
    match = re.search(r'<[^>]+class="' + re.escape(classname) + r'(?: [^"]*)?"[^>]*>(.*?)</[^>]+>', fragment, re.S)
    return clean(match[1]) if match else None


def events():
    records, requests = {}, []
    for year in (2024, 2025):
        for month in range(1, 13):
            ids, page, total = [], 1, None
            while total is None or len(ids) < total:
                query = dict(searchYear=year, searchMonth=month, searchCodeArrB='5,6,7,8,9,10', fetchStart=page)
                url = SEOUL + 'monthList.do?' + urllib.parse.urlencode(query)
                filename = f'seoul/{year}_{month:02}_page{page}.html'
                text = fetch(url, filename).decode('utf-8')
                match = re.search(r'class="search-result-num">([0-9,]+)', text)
                assert match, filename
                total = int(match[1].replace(',', ''))
                fragments = re.findall(r'<a[^>]+onclick="fnGoView\(\'(\d+)\'\);".*?</a>', text, re.S)
                blocks = re.findall(r'<a[^>]+onclick="fnGoView\(\'\d+\'\);".*?</a>', text, re.S)
                if total:
                    assert fragments and not set(fragments).intersection(ids), filename
                ids.extend(fragments)
                for identifier, block in zip(fragments, blocks):
                    if identifier not in records:
                        records[identifier] = dict(id=identifier, title=field(block,'event-name'),
                                                   venue=field(block,'event-place'), category=field(block,'event-field'),
                                                   listing_date_text=field(block,'event-date'), scheduled_time_text=field(block,'event-time'),
                                                   source_url=SEOUL+'stadiumView.do?month=Y&schSeq='+identifier,
                                                   observed_months=[], preforecast_availability_verified=False)
                    records[identifier]['observed_months'].append(f'{year}-{month:02}')
                requests.append(dict(year=year, month=month, page=page, total=total, ids=fragments, source_url=url, raw_file=filename))
                page += 1
                assert page < 100
            assert len(ids) == total, (year, month, len(ids), total)
            print(year, month, 'events', total, flush=True)
    save(OUT/'seoul_listings.json', list(records.values()))
    save(OUT/'seoul_requests.json', requests)
    for index, (identifier, record) in enumerate(records.items()):
        text = fetch(record['source_url'], f'seoul/detail_{identifier}.html').decode('utf-8')
        plain = clean(text)
        dates = re.search(r'이용기간\s+(\d{8})\s*~\s*(\d{8})\s*/\s*(.*?)\s*이용연령', plain)
        assert dates, identifier
        record.update(start_date=datetime.strptime(dates[1], '%Y%m%d').date().isoformat(),
                      end_date=datetime.strptime(dates[2], '%Y%m%d').date().isoformat(),
                      detail_time_text=dates[3], raw_file=f'seoul/detail_{identifier}.html',
                      published_at=None, daily_occurrence_verified=False)
        if index % 20 == 0:
            print('details', index+1, '/',len(records),flush=True)
    save(OUT / 'seoul_events.json', list(records.values()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('part', choices=['baseball','events'])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        baseball() if args.part == 'baseball' else events()
    except Exception as error:
        save(OUT/f'{args.part}_error.json', dict(error=repr(error), utc=datetime.now(timezone.utc).isoformat()))
        raise
