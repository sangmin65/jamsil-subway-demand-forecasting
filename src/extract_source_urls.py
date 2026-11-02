"""Extract only KBO GameCenter URLs from workbook XML; ignore all status values."""
import argparse
import json
import re
import zipfile
import hashlib
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse, parse_qs


def extract(path):
    urls = set()
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith('xl/') or not name.endswith(('.xml', '.rels')):
                continue
            for element in ET.fromstring(archive.read(name)).iter():
                for value in [element.text or '', *element.attrib.values()]:
                    for candidate in re.findall(r'https?://[^\s<>"\']+', value):
                        parsed = urlparse(candidate)
                        if parsed.hostname == 'www.koreabaseball.com' and parsed.path == '/Schedule/GameCenter/Main.aspx':
                            urls.add(candidate)
    return sorted(urls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.workbook.resolve() == args.output.resolve():
        parser.error('Cannot overwrite workbook')
    urls = extract(args.workbook)
    years = Counter(parse_qs(urlparse(u).query).get('gameDate', ['unknown'])[0][:4] for u in urls)
    selected = [u for u in urls if parse_qs(urlparse(u).query).get('gameDate', [''])[0][:4] in ('2024', '2025')]
    result = {'source_file': args.workbook.name, 'sha256': hashlib.sha256(args.workbook.read_bytes()).hexdigest(),
              'purpose': 'URLs only. Workbook statuses and other data are not evidence.',
              'unique_urls_by_year': dict(years), 'in_scope_url_count': len(selected),
              'urls_2024_2025': selected, 'verification': 'URLs extracted, pages not individually verified'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'by_year':dict(years),'in_scope':len(selected)}))
