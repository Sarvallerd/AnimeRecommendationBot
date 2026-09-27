"""Run the installed production pipeline against a tiny verified offline registry."""
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

from recsys.build import build
from recsys.bundle import canonical_bytes, check_bundle
from recsys.export import export_bundle
from recsys.normalize import normalize
from recsys.obtain import obtain

root = Path(os.environ.get('ARB_FIXTURE_DATA_DIR', '/data'))
raw = root / 'raw'
raw.mkdir(parents=True, exist_ok=True)

def csv_data(header, rows):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode()

def descriptor(source_id, filename, content, **origin):
    return {'id': source_id, 'filename': filename, 'size': len(content),
            'sha256': hashlib.sha256(content).hexdigest(), **origin}

anime = csv_data(['MAL_ID', 'Name', 'English name', 'Japanese name', 'Genres', 'Score',
                  'Episodes', 'Aired', 'Premiered', 'Type', 'Duration'], [
    ['1', 'Космический рейс', 'Space Voyage', '宇宙', 'Sci-Fi, Adventure', '8.5', '12',
     'Apr 3, 1998', 'Spring 1998', 'TV', '24 min. per ep.'],
    ['2', 'Star Garden', 'Star Garden', '星の庭', 'Fantasy, Drama', '7.4', '1',
     '2001', 'Unknown', 'Movie', '90 min.'],
    ['3', 'Silent Sea', 'Silent Sea', '静かな海', 'Drama', '7.1', '6',
     '2004', 'Unknown', 'OVA', '30 min. per ep.'],
])
synopsis = csv_data(['MAL_ID', 'sypnopsis'], [
    ['1', 'A crew travels across distant stars.'],
    ['2', 'A garden appears under the stars.'],
    ['3', 'A traveler reaches a silent sea.'],
])
glove = ('space ' + ' '.join(['1'] + ['0'] * 299) + '\n' +
         'star ' + ' '.join(['0', '1'] + ['0'] * 298) + '\n' +
         'sea ' + ' '.join(['0', '0', '1'] + ['0'] * 297) + '\n').encode()
archive_stream = io.BytesIO()
with zipfile.ZipFile(archive_stream, 'w') as archive:
    archive.writestr('glove.6B.300d.txt', glove)
zip_bytes = archive_stream.getvalue()
for filename, payload in [('anime.csv', anime), ('anime_with_synopsis.csv', synopsis),
                          ('glove.6B.zip', zip_bytes)]:
    (raw / filename).write_bytes(payload)
registry = {'schema_version': 1, 'provenance': {
    'mal': {'commit': 'arb-019-fixture', 'repository': 'https://example.invalid/mal'},
    'glove': {'project': 'https://example.invalid/glove',
              'pretrained_vectors_license': 'fixture-only'},
}, 'sources': [
    descriptor('mal_anime', 'anime.csv', anime, url='https://example.invalid/anime.csv'),
    descriptor('mal_synopsis', 'anime_with_synopsis.csv', synopsis,
               url='https://example.invalid/anime_with_synopsis.csv'),
    descriptor('glove_zip', 'glove.6B.zip', zip_bytes,
               url='https://example.invalid/glove.6B.zip'),
    descriptor('glove_300d', 'glove.6B.300d.txt', glove,
               archive_id='glove_zip', member='glove.6B.300d.txt'),
]}
sha = hashlib.sha256(canonical_bytes(registry)).hexdigest()
obtain(raw, offline=True, registry=registry, registry_sha256=sha)
normalized = root / 'work' / 'normalized'
built = root / 'work' / 'build'
report = normalize(raw / 'anime.csv', raw / 'anime_with_synopsis.csv', normalized,
                   registry=registry, registry_sha256=sha)
build_report = build(normalized / 'catalog.json', report,
                     raw / 'glove.6B.300d.txt', built, registry=registry,
                     registry_sha256=sha)
arguments = (normalized / 'catalog.json', built / 'neighbors.json', build_report, report)
store = root / 'bundles'
first = export_bundle(*arguments, store, registry=registry, registry_sha256=sha)
identity, count = check_bundle(first)
assert count == 3 and first.name == identity.replace(':', '-')
contents = {name: (first / name).read_bytes() for name in
            ('catalog.json', 'neighbors.json', 'manifest.json')}
second = export_bundle(*arguments, store, registry=registry, registry_sha256=sha)
assert first == second and all((second / name).read_bytes() == contents[name] for name in contents)
# The exporter owns file bytes. Publish read permissions for the distinct runtime UID.
os.chmod(first, 0o755)
for name in contents:
    os.chmod(first / name, 0o644)
print(first)
