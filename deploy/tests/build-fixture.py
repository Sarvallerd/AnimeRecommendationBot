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
print(f'LEGACY_BUNDLE={first}')

# Exercise the named release with a complete small local lock. This file is mounted in
# the builder image, where repository test modules are intentionally unavailable.
import shutil
import platform
from recsys import __version__
from recsys.aliases import AliasSpec, CSV_COLUMNS as ALIAS_COLUMNS, _derive, normalize_aliases
from recsys.dataset import DatasetSpec, normalize_dataset, validate_dataset
from recsys.dataset_metadata import DETAIL_COLUMNS
from recsys.dataset_interactions import CSV_COLUMNS as RATING_COLUMNS
from recsys.release import compose_release
from recsys.release_build import build_release, export_release_bundle
from recsys.release_prepare import prepare_release

legacy_raw = raw / 'mal-2020-glove-v1'
legacy_raw.mkdir()
for filename in ('anime.csv', 'anime_with_synopsis.csv', 'glove.6B.zip',
                 'glove.6B.300d.txt', 'source-manifest.json'):
    shutil.copy2(raw / filename, legacy_raw / filename)
shared_glove = raw / 'glove-6b-300d-v1'
shared_glove.mkdir()
(shared_glove / 'glove.6B.300d.txt').write_bytes(glove)
legacy_again, legacy_identity, legacy_count = prepare_release(
    root, 'mal-2020-glove-v1', _legacy_registry=(registry, sha))
assert legacy_again == first and legacy_identity == identity and legacy_count == 3

for name in ('datasets', 'aliases', 'releases'):
    (root / name).mkdir()
source_cache = raw / 'neelagiri-2025-v1'
source_cache.mkdir()

def details(mal, title, date, status, year):
    row = {field: '' for field in DETAIL_COLUMNS}
    row.update(mal_id=str(mal), title=title, title_japanese=title + ' JP',
               start_date=date, status=status, year=str(year), genres="['Drama']",
               score='8.0', episodes='12.0', synopsis='A story.', type='TV')
    return [row[field] for field in DETAIL_COLUMNS]

base_details = csv_data(DETAIL_COLUMNS, [
    details(1, 'One', '2025-01-01', 'Finished Airing', 2025),
    details(2, 'Two', '2024-02-01', 'Currently Airing', 2024)])
ratings_rows = [
    ['A', '1', 'completed', '8', 'False', '1'],
    ['A', '1', 'completed', '8', 'False', '1'],
    ['A', '2', 'watching', '9', 'False', '1'],
    ['B', '2', 'dropped', '7', 'False', '1'],
    ['C', '2', 'plan_to_watch', '0', 'False', '1'],
    ['', '1', 'completed', '6', 'False', '1'],
    ['A', '2', 'plan_to_watch', '0', 'False', '1']]
ratings_raw = csv_data(RATING_COLUMNS, ratings_rows)
(source_cache / 'details.csv').write_bytes(base_details)
(source_cache / 'ratings.csv').write_bytes(ratings_raw)
base_sources = [descriptor('neelagiri_details_csv', 'details.csv', base_details,
                           url='https://example.invalid/details.csv'),
                descriptor('neelagiri_ratings_csv', 'ratings.csv', ratings_raw,
                           url='https://example.invalid/ratings.csv')]
base_registry = {'schema_version': 1,
                 'provenance': {'dataset': {'snapshot_id': 'neelagiri-2025-v1'}},
                 'sources': base_sources}
base_registry_bytes = canonical_bytes(base_registry)
base_source_manifest = {
    'schema_version': 1,
    'registry_sha256': hashlib.sha256(base_registry_bytes).hexdigest(),
    'provenance': base_registry['provenance'],
    'verified': [{key: source[key] for key in ('filename', 'size', 'sha256')}
                 for source in base_sources]}
(source_cache / 'source-manifest.json').write_bytes(canonical_bytes(base_source_manifest))
base_counts = {'metadata_rows': 2, 'catalog_items': 2, 'raw_rows': 7,
               'blank_user_rows': 1, 'zero_or_missing_rows': 2,
               'valid_explicit_rows': 4, 'raw_explicit_pairs': 3,
               'repeated_explicit_triples': 1, 'raw_mapped_pairs': 3,
               'accepted_pairs': 2, 'accepted_users': 2, 'accepted_rated_mal_ids': 2,
               'completed_only_pairs': 1, 'completed_only_users': 1,
               'recent_2024_plus': 2, 'recent_2025_plus': 1, 'mixed_unset_pairs': 1}
from importlib import resources
base_expected_template = json.loads(resources.files('recsys').joinpath(
    'neelagiri_2025_v1_expectations.json').read_bytes())
base_counts = {key: base_counts.get(key, 0) for key in base_expected_template['counts']}
base_expected = {'schema_version': 1, 'snapshot_id': 'neelagiri-2025-v1',
                 'source_sha256': {s['filename']: s['sha256'] for s in base_sources},
                 'counts': base_counts,
                 'distributions': {
                     'raw_status_histogram': {'completed': 3, 'watching': 1,
                                              'dropped': 1, 'plan_to_watch': 2},
                     'accepted_status_histogram': {'completed': 1, 'dropped': 1},
                     'accepted_score_histogram': {'8': 1, '7': 1},
                     'rated_items_per_user_bins': {'0': 2},
                     'positive_items_per_user_bins': {'0': 2}}}
base_spec = DatasetSpec('neelagiri-2025-v1', base_registry_bytes,
                        'neelagiri_2025_v1_expectations.json', canonical_bytes(base_expected))
base_temp = root / 'work' / 'fixture-base'
normalize_dataset('neelagiri-2025-v1', source_cache, base_temp,
                  scratch_dir=root / 'work' / 'fixture-scratch', _spec=base_spec)
base_id = validate_dataset(base_temp, _spec=base_spec).identity
base_dir = root / 'datasets' / base_id.replace(':', '-')
base_temp.rename(base_dir)
base_catalog = json.loads((base_dir / 'catalog.json').read_bytes())['anime']

alias_cache = raw / 'nafiul-2026-v1'
alias_cache.mkdir()
def alias_row(mal, main, english):
    row = {field: '' for field in ALIAS_COLUMNS}
    row.update(mal_id=str(mal), title=main, title_english=english,
               mal_url=f'https://myanimelist.net/anime/{mal}/slug')
    return [row[field] for field in ALIAS_COLUMNS]
alias_raw = csv_data(ALIAS_COLUMNS, [alias_row(1, 'Space One', 'Space'),
                                     alias_row(2, 'Star Two', 'Star')])
(alias_cache / 'anime_info.csv').write_bytes(alias_raw)
alias_source = descriptor('nafiul_anime_csv', 'anime_info.csv', alias_raw,
                          url='https://example.invalid/anime_info.csv')
alias_registry = {'schema_version': 1,
                  'provenance': {'dataset': {'snapshot_id': 'nafiul-2026-v1', 'version': 1}},
                  'sources': [alias_source]}
alias_registry_bytes = canonical_bytes(alias_registry)
(alias_cache / 'source-manifest.json').write_bytes(canonical_bytes({
    'schema_version': 1, 'registry_sha256': hashlib.sha256(alias_registry_bytes).hexdigest(),
    'provenance': alias_registry['provenance'],
    'verified': [{key: alias_source[key] for key in ('filename', 'size', 'sha256')}]}))
_, alias_qa = _derive(alias_raw, base_catalog)
alias_expected = {'schema_version': 1, 'snapshot_id': 'nafiul-2026-v1',
                  'source_sha256': alias_source['sha256'], 'base_identity': base_id,
                  'counts': alias_qa['counts'], 'fields': alias_qa['fields']}
alias_spec = AliasSpec('nafiul-2026-v1', alias_registry_bytes,
                       'nafiul_2026_v1_alias_expectations.json', canonical_bytes(alias_expected))
alias_temp = root / 'work' / 'fixture-aliases'
normalize_aliases(alias_cache, base_dir, alias_temp, snapshot='nafiul-2026-v1',
                  expected_base_identity=base_id, _spec=alias_spec, _base_spec=base_spec)
alias_id = 'sha256:' + hashlib.sha256((alias_temp / 'alias-manifest.json').read_bytes()).hexdigest()
alias_dir = root / 'aliases' / alias_id.replace(':', '-')
alias_temp.rename(alias_dir)
release_temp = root / 'work' / 'fixture-release'
release_manifest = compose_release(base_dir, alias_dir, alias_cache, release_temp,
                                   expected_base_identity=base_id,
                                   expected_alias_identity=alias_id,
                                   _base_spec=base_spec, _alias_spec=alias_spec)
release_id = 'sha256:' + hashlib.sha256(release_manifest.read_bytes()).hexdigest()
release_dir = root / 'releases' / release_id.replace(':', '-')
release_temp.rename(release_dir)
fixture_glove_registry = {'schema_version': 1,
    'provenance': {'glove': {'project': 'https://example.invalid/glove',
                             'pretrained_vectors_license': 'fixture-only'}},
    'sources': [descriptor('glove_zip', 'glove.zip', b'z',
                           url='https://example.invalid/glove.zip'),
                descriptor('glove_300d', 'glove.6B.300d.txt', glove,
                           archive_id='glove_zip', member='glove.6B.300d.txt')]}
fixture_glove_registry_bytes = canonical_bytes(fixture_glove_registry)
common = {'expected_release_identity': release_id, 'base_dir': base_dir,
          'aliases_dir': alias_dir, 'alias_data_dir': alias_cache,
          '_base_spec': base_spec, '_alias_spec': alias_spec,
          '_glove_registry_bytes': fixture_glove_registry_bytes}
fixture_report_path = build_release(release_dir, shared_glove / 'glove.6B.300d.txt',
                                    root / 'work' / 'fixture-reference-build',
                                    block_size=256, **common)
fixture_candidate = export_release_bundle(
    release_dir, fixture_report_path.parent / 'neighbors.json', fixture_report_path,
    root / 'work' / 'fixture-reference-store', **common)
fixture_report = json.loads(fixture_report_path.read_bytes())
fixture_catalog = json.loads((release_dir / 'catalog.json').read_bytes())['anime']
fixture_neighbors = json.loads((fixture_candidate / 'neighbors.json').read_bytes())['neighbors']
package_dir = Path(__import__('recsys').__file__).parent
fixture_lock = {
    'schema_version': 1, 'kind': 'arb-release-lock', 'release_id': 'fixture-v1',
    'inputs': {
        'base': {'snapshot_id': 'neelagiri-2025-v1', 'identity': base_id},
        'aliases': {'snapshot_id': 'nafiul-2026-v1', 'identity': alias_id},
        'catalog_release': {'identity': release_id,
                            'catalog_sha256': hashlib.sha256((release_dir / 'catalog.json').read_bytes()).hexdigest(),
                            'catalog_count': 2},
        'embeddings': {'snapshot_id': 'glove-6b-300d-v1',
                       'registry_sha256': hashlib.sha256(fixture_glove_registry_bytes).hexdigest(),
                       'text_size': len(glove), 'text_sha256': hashlib.sha256(glove).hexdigest()},
        'algorithm': fixture_report['algorithm']},
    'reference': {
        'block_size': 256,
        'build_report_sha256': hashlib.sha256(fixture_report_path.read_bytes()).hexdigest(),
        'bundle_identity': 'sha256:' + hashlib.sha256((fixture_candidate / 'manifest.json').read_bytes()).hexdigest(),
        'catalog_sha256': fixture_report['files']['catalog.json']['sha256'],
        'neighbors_sha256': fixture_report['files']['neighbors.json']['sha256'],
        'producer': {'source_commit': 'fixture', 'package_version': __version__,
                     'python_version': platform.python_version(), 'numpy_version': __import__('numpy').__version__,
                     'code_files_sha256': {name: hashlib.sha256((package_dir / name).read_bytes()).hexdigest()
                                           for name in ('release.py', 'build.py', 'release_build.py',
                                                        'export.py', 'bundle.py')}}},
    'semantic_expectations': {
        'catalog_count': 2, 'vector_status_counts': fixture_report['coverage']['vector_status_counts'],
        'zero_vector_ids': fixture_report['coverage']['zero_vector_ids'],
        'neighbor_count_histogram': fixture_report['coverage']['neighbor_count_histogram'],
        'smoke_seed': {'mal_id': 1, 'query': fixture_catalog['1']['title'],
                       'neighbor_ids': [item['mal_id'] for item in fixture_neighbors['1']]},
        'empty_seed': None}}
prepared, prepared_identity, prepared_count = prepare_release(
    root, _lock_bytes=canonical_bytes(fixture_lock), _base_spec=base_spec,
    _alias_spec=alias_spec, _glove_registry_bytes=fixture_glove_registry_bytes)
assert prepared_identity == fixture_lock['reference']['bundle_identity'] and prepared_count == 2
assert prepared == prepare_release(root, _lock_bytes=canonical_bytes(fixture_lock),
                                   _base_spec=base_spec, _alias_spec=alias_spec,
                                   _glove_registry_bytes=fixture_glove_registry_bytes)[0]
os.chmod(prepared, 0o755)
for filename in ('catalog.json', 'neighbors.json', 'manifest.json'):
    os.chmod(prepared / filename, 0o644)
print(f'CURRENT_BUNDLE={prepared}')
