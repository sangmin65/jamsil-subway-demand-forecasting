"""Read-only source audit; never modifies model inputs or expands venue bookings into events."""
import calendar
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from collect_supplement import clean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/external/supplement_2026-11-09'
OUT = ROOT / 'data/external/event_verification_2026-11-10'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def fetch(url, name, headers=None):
    path = OUT / name
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=30) as r:
        raw = r.read()
    data = json.loads(raw)
    save(path, data)
    save(path.with_suffix('.meta.json'), dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
          sha256=hashlib.sha256(path.read_bytes()).hexdigest(), headers=headers))
    return data


def main():
    games = []
    for year in (2024, 2025):
        for month in range(1, 13):
            url = f'https://api.kbl.or.kr/match/list?fromDate={year}{month:02}01&toDate={year}{month:02}{calendar.monthrange(year,month)[1]}&tcodeList=all'
            rows = fetch(url, f'kbl_{year}_{month:02}.json', {'Channel':'WEB','TeamCode':'XX','lang':'ko','X-Requested-With':'XMLHttpRequest'})
            assert isinstance(rows, list)
            games.extend(dict(r, source_url=url) for r in rows)
    assert len(games) == len({g['gmkey'] for g in games})
    events = json.loads((SOURCE/'seoul_events.json').read_text(encoding='utf-8'))
    events = [r for r in events if r['venue'] != '잠실야구장']
    evidence_path = ROOT/'reports/metrics/event_manual_evidence_2026-11-10.json'
    evidence = json.loads(evidence_path.read_text(encoding='utf-8')) if evidence_path.exists() else {}
    reviewed = []
    for e in events:
        raw = (SOURCE/e['raw_file']).read_bytes()
        plain = clean(raw.decode('utf-8'))
        m = re.search(r'이용기간\s+(\d{8})\s*~\s*(\d{8})\s*/\s*(.*?)\s*이용연령', plain)
        assert m and m[1] == e['start_date'].replace('-','') and m[2] == e['end_date'].replace('-','') and m[3] == e['detail_time_text']
        row = dict(id=e['id'], title=e['title'], original_date_range=[e['start_date'],e['end_date']],
                   original_time=e['detail_time_text'], original_venue=e['venue'], source_url=e['source_url'],
                   raw_sha256=hashlib.sha256(raw).hexdigest(), extraction_verified=True,
                   status='source_only_unconfirmed', flags=[], corroboration=[],
                   actual_occurrence_verified=False, preforecast_availability_verified=False,
                   usable_as_training_input=False)
        if e['start_date'] != e['end_date']:
            row['flags'].append('range_is_not_daily_occurrence')
        same = [r['id'] for r in events if r['id'] != e['id'] and r['title'].replace(' ','') == e['title'].replace(' ','') and r['start_date']==e['start_date'] and r['end_date']==e['end_date'] and r['venue']==e['venue']]
        if same:
            row['flags'].append('duplicate_listing_candidate')
            row['duplicate_ids'] = same
        if '프로농구' in e['title']:
            teamtext = re.search(r'\((.*?)\)', e['title'])[1]
            teams = teamtext.split(':')
            candidates = [g for g in games if g.get('seasonGrade')==1 and e['start_date'].replace('-','')<=g['gameDate']<=e['end_date'].replace('-','') and all(any(t in (g[k] or '') for k in ['tnameH','tnameA']) for t in teams)]
            row['corroboration'] = [dict(date=g['gameDate'],time=g['gameStart'],venue=g['stadiumname'],home=g['tnameH'],away=g['tnameA'],ended=g['isEnded'],game_key=g['gmkey'],source_url=g['source_url']) for g in candidates]
            if len(candidates)==1:
                g=candidates[0]
                time=re.search(r'(\d{1,2})(?::(\d{2})|시)',e['detail_time_text'])
                tm=f'{int(time[1]):02}{time[2] or "00"}' if time else None
                venue={'잠실실내':'잠실실내체육관'}.get(g['stadiumname'],g['stadiumname'])
                matches=(e['start_date']==e['end_date'] and g['gameDate']==e['start_date'].replace('-','') and tm==g['gameStart'] and e['venue']==venue)
                row['status']='corroborated' if matches else 'discrepancy'
                row['actual_occurrence_verified']=matches and g['isEnded']==1
                if not matches:row['flags'].append('kbl_date_time_or_venue_mismatch')
            else:
                row['flags'].append('kbl_no_unique_match')
        if e['id'] in evidence:
            row['manual_review']=evidence[e['id']]
            row['status']=evidence[e['id']]['status']
            row['actual_occurrence_verified']=evidence[e['id']].get('actual_occurrence_verified',False)
        reviewed.append(row)
    matched={g['game_key'] for r in reviewed for g in r['corroboration']}
    missing=[dict(date=g['gameDate'],time=g['gameStart'],venue=g['stadiumname'],home=g['tnameH'],away=g['tnameA'],game_key=g['gmkey'],source_url=g['source_url']) for g in games if g.get('seasonGrade')==1 and g['stadiumname'] in ('잠실실내체육관','잠실실내') and g['gmkey'] not in matched]
    result=dict(review_date='2026-11-10',scope='2024-2025 non-baseball Seoul venue listings; source audit, not a complete event census',
        counts=dict(total=len(reviewed),by_status=dict(Counter(r['status'] for r in reviewed)),kbl_requests=24,
                    actual_occurrence_supported=sum(r['actual_occurrence_verified'] for r in reviewed),missing_indoor_basketball_records=len(missing)),
        records=reviewed,missing_indoor_basketball_records=missing,
        limits=['Source agreement is not proof of historical publication time.', 'Unconfirmed rows are quarantined, not converted to no-event days.',
                'No training inputs, model settings, targets or held-out evaluation were changed.', 'Listing IDs are not unique events; duplicates and ranges require resolution.'])
    save(ROOT/'reports/metrics/event_verification_2026-11-10.json',result)
    print(json.dumps(result['counts'],ensure_ascii=False))


if __name__=='__main__':main()
