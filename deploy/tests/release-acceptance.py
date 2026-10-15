#!/usr/bin/env python3
"""Private, opt-in full-input offline release compatibility and read-only verification."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NEW = 'neelagiri-2025-nafiul-2026-v1'
LABELS = ('old', 'new-A', 'new-B', 'new-A-return', 'old-restore')
CHECKS = ('reference', 'deterministic_ab', 'python_rust', 'semantic_expectations',
          'input_immutability', 'database_history', 'original_settings', 'restored_old_prepare')
REASON = {'PASS', 'INVALID_ARGUMENT', 'UNSAFE_PATH', 'UNSAFE_SETTINGS', 'SOURCE_CHANGED',
          'PROFILE_MISMATCH', 'INPUT_CHANGED', 'BUILD_FAILED', 'BUILD_MISMATCH',
          'RUNTIME_UNAVAILABLE', 'BOT_RUNNING', 'DATABASE_UNHEALTHY', 'DATABASE_HISTORY',
          'PREPARE_FAILED', 'PREPARE_TIMEOUT', 'PHASE_MISMATCH', 'RESTORE_FAILED',
          'RESTORE_PENDING', 'EVIDENCE_INVALID', 'TRANSPORT_TIMEOUT'}


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p = load('arb033_profiles', 'acceptance-profiles.py')
e = load('arb033_evidence', 'acceptance-evidence.py')


class ReleaseError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


def need(condition, code):
    if not condition:
        raise ReleaseError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def guarded(path, *, directory=False):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and
         all(not part.is_symlink() for part in (path, *path.parents)),
         'UNSAFE_PATH')
    need(path.is_dir() if directory else path.is_file(), 'UNSAFE_PATH')
    return path


def primary_root(checkout=ROOT):
    result = subprocess.run(['git', '-c', f'safe.directory={checkout}', 'rev-parse',
                             '--path-format=absolute', '--git-common-dir'], cwd=checkout,
                            capture_output=True, text=True, timeout=20, check=False)
    need(result.returncode == 0, 'SOURCE_CHANGED')
    common = guarded(Path(result.stdout.strip()), directory=True)
    need(common.name == '.git' and common.parent.is_dir(), 'SOURCE_CHANGED')
    return common.parent


def private_roots(checkout=ROOT):
    primary = primary_root(checkout)
    return (primary/'.arb', primary/'.tools'/'arb033-acceptance')


def project_lock_dir(checkout=ROOT):
    return primary_root(checkout)/'.arb'/'acceptance-locks'


def private_file(path, raw):
    need(not path.exists() and not path.is_symlink() and path.parent.is_dir() and
         stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0, 'UNSAFE_PATH')
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {'file': path.name, 'sha256': sha(raw)}


def save_json(path, value):
    return private_file(path, e.canonical(value))


def saved(evidence, descriptor):
    need(type(descriptor) is dict and set(descriptor) == {'file', 'sha256'} and
         type(descriptor['file']) is str and Path(descriptor['file']).name == descriptor['file'] and
         re.fullmatch('[0-9a-f]{64}', descriptor['sha256']) is not None, 'EVIDENCE_INVALID')
    path = guarded(evidence / descriptor['file'])
    raw = path.read_bytes()
    need(sha(raw) == descriptor['sha256'], 'EVIDENCE_INVALID')
    return raw


def saved_json(evidence, descriptor):
    return p.canonical(saved(evidence, descriptor))


def descriptor(path):
    path = guarded(path)
    return {'path': str(path), 'sha256': file_sha(path), 'bytes': path.stat().st_size}


def check_descriptor(item):
    need(type(item) is dict and set(item) == {'path','sha256','bytes'} and
         type(item['sha256']) is str and re.fullmatch('[0-9a-f]{64}', item['sha256']) and
         type(item['bytes']) is int and item['bytes'] >= 0, 'EVIDENCE_INVALID')
    path = guarded(Path(item['path']))
    need(path.stat().st_size == item['bytes'] and file_sha(path) == item['sha256'], 'INPUT_CHANGED')


def files_under(paths):
    result = {}
    for base in paths:
        base = guarded(base, directory=True)
        for path in sorted(base.rglob('*')):
            need(not path.is_symlink(), 'UNSAFE_PATH')
            if path.is_file():
                result[str(path)] = descriptor(path)
    return result


def source_head():
    root = subprocess.run(['git', '-c', f'safe.directory={ROOT}', 'rev-parse',
                           '--show-toplevel'], cwd=ROOT, capture_output=True,
                          text=True, timeout=20, check=False)
    need(root.returncode == 0 and root.stdout.strip() == str(ROOT), 'SOURCE_CHANGED')
    result = subprocess.run(['git', '-c', f'safe.directory={ROOT}', 'rev-parse', 'HEAD'],
                            cwd=ROOT, capture_output=True, text=True, timeout=20, check=False)
    need(result.returncode == 0 and re.fullmatch('[0-9a-f]{40}', result.stdout.strip()),
         'SOURCE_CHANGED')
    return result.stdout.strip()


def clean(head):
    need(source_head() == head, 'SOURCE_CHANGED')
    result = subprocess.run(['git', '-c', f'safe.directory={ROOT}', 'status', '--porcelain',
                             '--untracked-files=normal'], cwd=ROOT, capture_output=True,
                            text=True, timeout=20, check=False)
    need(result.returncode == 0 and not result.stdout, 'SOURCE_CHANGED')


def env_clean():
    return {k:v for k,v in os.environ.items() if not k.startswith(('ARB_', 'COMPOSE_'))}


def command(argv, *, timeout=300, input_text=None, both=False):
    try:
        result = subprocess.run(argv, cwd=ROOT, env=env_clean(), input=input_text,
                                capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ReleaseError('TRANSPORT_TIMEOUT') from exc
    need(result.returncode == 0, 'BUILD_FAILED' if timeout > 300 else 'RUNTIME_UNAVAILABLE')
    return result.stdout + result.stderr if both else result.stdout


class Transport:
    def __init__(self, project, settings):
        self.project = project
        self.settings = settings

    def compose(self, *args, timeout=300, input_text=None):
        return command(['docker','compose','--env-file',str(self.settings),'-p',self.project,
                        '-f',str(ROOT/'compose.yaml'),*args], timeout=timeout,
                       input_text=input_text)

    def docker(self, *args):
        return command(['docker',*args], both=bool(args and args[0] == 'logs'))

    def evidence(self):
        e.compose = lambda project, settings, *args, input_text=None: self.compose(*args, input_text=input_text)
        e.docker = lambda *args: self.docker(*args)


def postgres_health(t):
    pg = e.container(t.project,t.settings,'postgres')
    need(pg is not None and pg['running'], 'DATABASE_UNHEALTHY')
    status = t.docker('inspect','--format','{{.State.Health.Status}}',pg['id']).strip()
    need(status == 'healthy' and pg['database_volume'], 'DATABASE_UNHEALTHY')
    return pg


def quiescent(t, *, require_prepare_idle=True):
    need(e.container(t.project,t.settings,'bot') is None or
         not e.container(t.project,t.settings,'bot')['running'], 'BOT_RUNNING')
    prepare = e.container(t.project,t.settings,'prepare')
    need(not require_prepare_idle or prepare is None or not prepare['running'], 'PREPARE_FAILED')
    return postgres_health(t)


def nonempty_history(db):
    e.validate_database(db)
    need(any(db[name] for name in ('requests','delivered','anime_events',
                                   'recommendation_ratings','feedback','legacy_requests',
                                   'legacy_feedback')), 'DATABASE_HISTORY')


def change_selection(original, bundle):
    lines = original.decode().splitlines(keepends=True)
    need(sum(line.startswith('ARB_BUNDLE_DIR=') for line in lines) == 1, 'UNSAFE_SETTINGS')
    return ''.join('ARB_BUNDLE_DIR='+str(bundle)+'\n' if line.startswith('ARB_BUNDLE_DIR=')
                   else line for line in lines).encode()


def setup_paths(data, lock):
    base = data/'datasets'/lock['inputs']['base']['identity'].replace(':','-')
    alias = data/'aliases'/lock['inputs']['aliases']['identity'].replace(':','-')
    release = data/'releases'/lock['inputs']['catalog_release']['identity'].replace(':','-')
    alias_raw = data/'raw'/'nafiul-2026-v1'
    glove_dir = data/'raw'/'glove-6b-300d-v1'
    glove = glove_dir/'glove.6B.300d.txt'
    for path in (base,alias,release,alias_raw,glove_dir): guarded(path,directory=True)
    guarded(glove)
    return {'base':base,'aliases':alias,'release':release,'alias_raw':alias_raw,'glove':glove}


def builder_args(paths, lock):
    return ['--release-dir','/data/'+str(paths['release'].relative_to(paths['data'])),
            '--release-identity',lock['inputs']['catalog_release']['identity'],
            '--base-dir','/data/'+str(paths['base'].relative_to(paths['data'])),
            '--aliases-dir','/data/'+str(paths['aliases'].relative_to(paths['data'])),
            '--alias-data-dir','/data/'+str(paths['alias_raw'].relative_to(paths['data']))]


def probe_code():
    return """import hashlib,importlib.resources as r,json,platform,numpy,recsys
names=('build.py','bundle.py','export.py','release.py','release_build.py')
files={name:hashlib.sha256(r.files('recsys').joinpath(name).read_bytes()).hexdigest() for name in names}
lock=hashlib.sha256(r.files('recsys').joinpath('release_neelagiri_2025_nafiul_2026_v1.json').read_bytes()).hexdigest()
registry=hashlib.sha256(r.files('recsys').joinpath('source_registry_glove_6b_300d_v1.json').read_bytes()).hexdigest()
print(json.dumps({'python_version':platform.python_version(),'numpy_version':numpy.__version__,'package_version':recsys.__version__,'code_files_sha256':files,'lock_sha256':lock,'registry_sha256':registry},sort_keys=True))"""


def check_probe(probe, profile):
    ref = profile.reference['producer']
    need(type(probe) is dict and set(probe) == {'python_version','numpy_version',
         'package_version','code_files_sha256','lock_sha256','registry_sha256'}, 'BUILD_MISMATCH')
    need(all(probe[key] == ref[key] for key in
             ('python_version','numpy_version','package_version')) and
         probe['code_files_sha256'] == ref['code_files_sha256'] and
         probe['lock_sha256'] == profile.lock_sha256 and
         probe['registry_sha256'] == profile.inputs['embeddings']['registry_sha256'],
         'BUILD_MISMATCH')
    need(all(file_sha(ROOT/'src/recsys'/name) == digest for name,digest in
             ref['code_files_sha256'].items()) and
         file_sha(ROOT/'src/recsys/source_registry_glove_6b_300d_v1.json') ==
             profile.inputs['embeddings']['registry_sha256'], 'BUILD_MISMATCH')


def check_report(report, report_raw, summary, profile, block):
    need(report.get('schema_version') == 2 and report.get('kind') == 'arb-release-build' and
         report.get('catalog_release_identity') == profile.inputs['catalog_release']['identity'] and
         report.get('inputs',{}).get('glove.6B.300d.txt',{}).get('sha256') ==
           profile.inputs['embeddings']['text_sha256'] and
         report.get('files') == {'catalog.json':{'sha256':summary.hashes['catalog.json']},
                                 'neighbors.json':{'sha256':summary.hashes['neighbors.json']}},
         'BUILD_MISMATCH')
    ref = profile.reference['producer']
    producer = report.get('producer',{})
    need(all(producer.get(key) == ref[key] for key in
             ('python_version','numpy_version','package_version')) and
         producer.get('code_files') == {name:ref['code_files_sha256'][name] for name in
                                         ('build.py','release_build.py')}, 'BUILD_MISMATCH')
    execution = report.get('execution',{})
    need(type(execution.get('requested_block_rows')) is int and
         execution['requested_block_rows'] == block and
         execution.get('effective_block_rows') == block, 'BUILD_MISMATCH')
    semantic = profile.expectations
    coverage = report.get('coverage',{})
    need(coverage.get('neighbor_count_histogram') == semantic['neighbor_count_histogram'] and
         coverage.get('vector_status_counts') == semantic['vector_status_counts'] and
         coverage.get('zero_vector_ids') == semantic['zero_vector_ids'],
         'BUILD_MISMATCH')
    details = report.get('anime',{})
    need(type(details) is dict and set(details) == set(summary.neighbors) and
         all(type(item) is dict and item.get('vector_status') in
             ('no_tokens','oov_only','zero_vector','ready') for item in details.values()) and
         {name:sum(item['vector_status'] == name for item in details.values()) for name in
          ('ready','oov_only','no_tokens','zero_vector')} == semantic['vector_status_counts'] and
         sorted(int(key) for key,item in details.items() if item.get('vector_status') != 'ready') ==
           semantic['zero_vector_ids'] and
         all(details[key]['vector_status'] == 'ready' or not summary.neighbors[key]
             for key in details) and
         all(details[key]['neighbor_count'] == len(summary.neighbors[key]) for key in details),
         'BUILD_MISMATCH')
    return sha(report_raw)


def bundle_files(bundle):
    return {name: descriptor(bundle/name) for name in ('catalog.json','neighbors.json','manifest.json')}


def check_retained(retained, old_path):
    need(retained['input_files'] == files_under(retained['input_paths']) and
         retained['old_files'] == bundle_files(old_path), 'INPUT_CHANGED')


def phase_time(previous_capture, started_at, captured_at):
    need(e.utc(previous_capture) < e.utc(started_at) <= e.utc(captured_at),
         'PHASE_MISMATCH')


def snapshot_phase(t, evidence, label, settings, baseline, expected, previous_id,
                   previous_capture, bot_image, pg0):
    t.settings = settings
    quiescent(t)
    t.compose('up','-d','--no-deps','--force-recreate','prepare')
    current = None
    for _ in range(120):
        current = e.container(t.project, settings, 'prepare')
        if current is not None and current['status'] == 'exited':
            break
        time.sleep(2)
    need(current is not None and current['status'] == 'exited', 'PREPARE_TIMEOUT')
    need(current['exit_code'] == 0 and current['id'] != previous_id and
         current['image_id'] == bot_image and current['user'] == '10001:10001' and
         current['read_only_root'] is True and current['restart_count'] == 0 and
         current['artifact_mount'] == {'source': str(expected['path']), 'rw': False} and
         current['bundle_id'] == expected['identity'] and
         current['bundle_count'] == expected['count'], 'PHASE_MISMATCH')
    pg = postgres_health(t)
    need(all(pg[key] == pg0[key] for key in ('id','image_id','database_volume','started_at')),
         'DATABASE_HISTORY')
    shot = e.snapshot(t.project, settings)
    phase_time(previous_capture,current['started_at'],shot['captured_at'])
    need(shot['runtime']['bot'] is None or not shot['runtime']['bot']['running'], 'BOT_RUNNING')
    need(shot['database'] == baseline['database'], 'DATABASE_HISTORY')
    e.validate_database(shot['database'])
    evidence_file = save_json(evidence/f'snapshot-{label}.json', shot)
    return {'label':label,'bundle_identity':expected['identity'],
            'settings':{'file':settings.name,'sha256':file_sha(settings)},
            'snapshot':evidence_file}, current['id'], shot['captured_at']


def run_prepare_sequence(t, evidence, settings, original_state, head, baseline, selected,
                         settings_by_label, bot_image, pg0, retained, old_path, switch_state):
    sequence = []
    previous = e.container(t.project,settings,'prepare')
    previous_id = previous['id'] if previous else None
    previous_capture = baseline['captured_at']
    for label in LABELS:
        selection = selected['old' if label.startswith('old') else
                             'new-B' if label == 'new-B' else 'new-A']
        need((sha(settings.read_bytes()),settings.stat().st_uid,settings.stat().st_gid,
              stat.S_IMODE(settings.stat().st_mode)) == original_state, 'UNSAFE_SETTINGS')
        clean(head)
        check_retained(retained,old_path)
        switch_state['begun'] = True
        entry,previous_id,previous_capture = snapshot_phase(
            t,evidence,label,settings_by_label[label],baseline,selection,previous_id,
            previous_capture,bot_image,pg0)
        sequence.append(entry)
    return sequence


def restore_old(t, settings, original_state, old_settings, old_path, old, pg0, retained):
    try:
        t.settings = settings
        need((sha(settings.read_bytes()),settings.stat().st_uid,settings.stat().st_gid,
              stat.S_IMODE(settings.stat().st_mode)) == original_state,
             'RESTORE_PENDING')
        pg = quiescent(t)
        need(all(pg[key] == pg0[key] for key in
                 ('id','image_id','database_volume','started_at')), 'RESTORE_PENDING')
        check_retained(retained,old_path)
    except Exception:
        return 'PENDING'
    try:
        t.settings = old_settings
        t.compose('up','-d','--no-deps','--force-recreate','prepare')
        restored = None
        for _ in range(120):
            restored = e.container(t.project,old_settings,'prepare')
            if restored is not None and restored['status'] == 'exited': break
            time.sleep(2)
        need(restored is not None and restored['exit_code'] == 0 and
             restored['bundle_id'] == old.identity and
             restored['artifact_mount'] == {'source':str(old_path),'rw':False},
             'RESTORE_FAILED')
        return 'PASS'
    except Exception:
        return 'FAIL'


def verify_record(evidence):
    evidence = guarded(evidence, directory=True)
    report = saved_json(evidence, {'file':'preparation-report.json',
                                   'sha256':file_sha(evidence/'preparation-report.json')})
    need(set(report) == {'schema_version','kind','scope','status','reason','profile','source',
                         'original','inputs','builds','runtime','checks','live_acceptance',
                         'production_history_preservation','live_verifier','saved_evidence'},
         'EVIDENCE_INVALID')
    need(type(report['schema_version']) is int and report['schema_version'] == 2 and
         report['kind'] == 'arb-offline-release-acceptance' and
         report['scope'] == 'full-inputs-offline-compatibility' and
         report['status'] == 'PASS' and report['reason'] == 'PASS' and
         report['live_acceptance'] == 'PENDING' and
         report['production_history_preservation'] == 'PENDING' and
         report['live_verifier'] == 'REQUIRES_DATA06_V2', 'EVIDENCE_INVALID')
    records = report['saved_evidence']
    need(set(records) == {'profile','old_profile','lock','original_settings','probe','baseline',
                          'input_files','input_layout','bundle_files','snapshot_sql','tool','compose',
                          'builder_dockerfile','bot_dockerfile'}, 'EVIDENCE_INVALID')
    need(report['source'] == {'commit':report['source'].get('commit'),
         'tool_sha256':sha(saved(evidence,records['tool'])),
         'compose_sha256':sha(saved(evidence,records['compose'])),
         'dockerfile_sha256':{'builder':sha(saved(evidence,records['builder_dockerfile'])),
                              'bot':sha(saved(evidence,records['bot_dockerfile']))}} and
         re.fullmatch('[0-9a-f]{40}', report['source']['commit']), 'EVIDENCE_INVALID')
    profile_raw = saved(evidence, records['profile'])
    lock_raw = saved(evidence, records['lock'])
    profile = p.load_profile(NEW, _profile_bytes=profile_raw, _lock_bytes=lock_raw)
    need(profile.sha256 == p.TRUSTED_NEW_PROFILE_SHA and
         profile.lock_sha256 == p.TRUSTED_LOCK_SHA, 'EVIDENCE_INVALID')
    need(report['profile'] == {'id':profile.id,'sha256':profile.sha256,
         'release_id':NEW,'release_lock_sha256':profile.lock_sha256}, 'EVIDENCE_INVALID')
    original_raw = saved(evidence, records['original_settings'])
    original = report['original']
    need(sha(original_raw) == original['settings_sha256'] and
         type(original['settings_uid']) is int and type(original['settings_gid']) is int and
         type(original['settings_mode']) is int and
         (evidence/records['original_settings']['file']).stat().st_uid == original['settings_uid'] and
         (evidence/records['original_settings']['file']).stat().st_gid == original['settings_gid'] and
         stat.S_IMODE((evidence/records['original_settings']['file']).stat().st_mode) ==
             original['settings_mode'], 'EVIDENCE_INVALID')
    original_value = e.read_settings(evidence/records['original_settings']['file'])
    old_path = Path(original_value['ARB_BUNDLE_DIR'])
    need(str(old_path) == original['bundle_path'], 'EVIDENCE_INVALID')
    old_profile = p.load_profile(p.LEGACY_RELEASE,
                                 _profile_bytes=saved(evidence,records['old_profile']))
    need(old_profile.sha256 == p.TRUSTED_LEGACY_PROFILE_SHA, 'EVIDENCE_INVALID')
    old_summary = p.check_profile_bundle(old_path,old_profile,
                                         expected_identity=original['bundle_identity'])
    need(old_summary.hashes == original['bundle_file_sha256'], 'EVIDENCE_INVALID')
    probe = saved_json(evidence, records['probe'])
    # Historical verification uses the saved probe and lock; it never compares today's HEAD or installed image.
    ref = profile.reference['producer']
    need(all(probe.get(key) == ref[key] for key in
             ('python_version','numpy_version','package_version')) and
         probe.get('code_files_sha256') == ref['code_files_sha256'] and
         probe.get('lock_sha256') == profile.lock_sha256 and
         probe.get('registry_sha256') ==
             profile.inputs['embeddings']['registry_sha256'], 'EVIDENCE_INVALID')
    input_files = saved_json(evidence,records['input_files'])
    layout = saved_json(evidence,records['input_layout'])
    need(set(layout) == {'base','aliases','release','alias_raw','glove'}, 'EVIDENCE_INVALID')
    need(input_files == files_under([Path(layout[key]) for key in
         ('base','aliases','release','alias_raw')]+[Path(layout['glove']).parent]),
         'INPUT_CHANGED')
    for key,filename,identity in (
        ('base','dataset-manifest.json',profile.inputs['base']['identity']),
        ('aliases','alias-manifest.json',profile.inputs['aliases']['identity']),
        ('release','release-manifest.json',profile.inputs['catalog_release']['identity'])):
        target = Path(layout[key])/filename
        need(str(target) in input_files and 'sha256:'+file_sha(target) == identity,
             'EVIDENCE_INVALID')
    need(str(Path(layout['release'])/'catalog.json') in input_files and
         file_sha(Path(layout['release'])/'catalog.json') ==
         profile.inputs['catalog_release']['catalog_sha256'] and
         str(Path(layout['glove'])) in input_files and
         file_sha(Path(layout['glove'])) == profile.inputs['embeddings']['text_sha256'] and
         Path(layout['glove']).stat().st_size == profile.inputs['embeddings']['text_size'],
         'EVIDENCE_INVALID')
    need(type(input_files) is dict and bool(input_files), 'EVIDENCE_INVALID')
    for path,item in input_files.items():
        need(path == item['path'], 'EVIDENCE_INVALID')
        check_descriptor(item)
    for key in ('base_identity','alias_identity','catalog_release_identity','embeddings_sha256'):
        need(key in report['inputs'], 'EVIDENCE_INVALID')
    need(report['inputs']['base_identity'] == profile.inputs['base']['identity'] and
         report['inputs']['alias_identity'] == profile.inputs['aliases']['identity'] and
         report['inputs']['catalog_release_identity'] == profile.inputs['catalog_release']['identity'] and
         report['inputs']['embeddings_sha256'] == profile.inputs['embeddings']['text_sha256'],
         'EVIDENCE_INVALID')
    bundle_records = saved_json(evidence,records['bundle_files'])
    need(set(bundle_records) == {'old','A','B'}, 'EVIDENCE_INVALID')
    for label, files in bundle_records.items():
        need(set(files) == {'catalog.json','neighbors.json','manifest.json'}, 'EVIDENCE_INVALID')
        for item in files.values(): check_descriptor(item)
    need({name:item['sha256'] for name,item in bundle_records['old'].items()} ==
         old_summary.hashes, 'EVIDENCE_INVALID')
    need(all(item['path'] == str(old_path/name) for name,item in
             bundle_records['old'].items()), 'EVIDENCE_INVALID')
    builds = report['builds']
    need(set(builds) == {'A','B'}, 'EVIDENCE_INVALID')
    reports = {}
    bundles = {}
    for label, block in (('A',256),('B',128)):
        entry = builds[label]
        need(set(entry) == {'identity','path','count','report_sha256','file_sha256',
                            'producer','block_size','report'}, 'EVIDENCE_INVALID')
        summary = p.check_profile_bundle(Path(entry['path']),profile,
                                         expected_identity=entry['identity'])
        need(summary.count == entry['count'] and summary.hashes == entry['file_sha256'] and
             entry['block_size'] == block and
             {name:item['sha256'] for name,item in bundle_records[label].items()} ==
             summary.hashes and all(item['path'] == str(Path(entry['path'])/name)
                                    for name,item in bundle_records[label].items()),
             'EVIDENCE_INVALID')
        raw = saved(evidence,entry['report'])
        reports[label] = p.canonical(raw)
        need(check_report(reports[label],raw,summary,profile,block) == entry['report_sha256']
             and entry['producer'] == reports[label]['producer'], 'EVIDENCE_INVALID')
        manifest = p.canonical((Path(entry['path'])/'manifest.json').read_bytes())
        sources = {item['name']:item for item in manifest['sources']}
        need(sources['build_report']['sha256'] == entry['report_sha256'] and
             sources['build_report']['metadata']['producer'] == reports[label]['producer'] and
             sources['bundle_exporter']['sha256'] ==
                profile.reference['producer']['code_files_sha256']['release_build.py'] and
             sources['bundle_exporter']['metadata']['export_sha256'] ==
                profile.reference['producer']['code_files_sha256']['export.py'] and
             sources['bundle_exporter']['metadata']['bundle_sha256'] ==
                profile.reference['producer']['code_files_sha256']['bundle.py'],
             'EVIDENCE_INVALID')
        bundles[label] = summary
    need(builds['A']['identity'] == profile.reference['bundle_identity'] and
         builds['A']['report_sha256'] == profile.reference['build_report_sha256'] and
         bundles['A'].hashes['catalog.json'] == profile.reference['catalog_sha256'] and
         bundles['A'].hashes['neighbors.json'] == profile.reference['neighbors_sha256'] and
         builds['A']['identity'] != builds['B']['identity'] and
         bundles['A'].hashes['catalog.json'] == bundles['B'].hashes['catalog.json'] and
         bundles['A'].hashes['neighbors.json'] == bundles['B'].hashes['neighbors.json'],
         'EVIDENCE_INVALID')
    a,b = copy.deepcopy(reports['A']),copy.deepcopy(reports['B'])
    del a['execution']; del b['execution']
    need(a == b, 'EVIDENCE_INVALID')
    baseline = saved_json(evidence,records['baseline'])
    nonempty_history(baseline['database'])
    runtime = report['runtime']
    need(set(runtime) == {'project','builder_image_id','bot_image_id','postgres_image_id',
                          'database_volume','baseline_snapshot_sha256','prepare_sequence'},
         'EVIDENCE_INVALID')
    need(baseline['schema_version'] == 1 and baseline['kind'] == 'full' and
         baseline['snapshot_sql_sha256'] == sha(saved(evidence,records['snapshot_sql'])) and
         baseline['source_commit'] == report['source']['commit'] and
         baseline['deployment'] == {'project':runtime['project'],
                                    'compose_sha256':report['source']['compose_sha256'],
                                    'settings_sha256':original['settings_sha256']} and
         records['baseline']['sha256'] == runtime['baseline_snapshot_sha256'],
         'EVIDENCE_INVALID')
    pg0 = baseline['runtime']['postgres']
    need(pg0 is not None and pg0['running'] and
         pg0['image_id'] == runtime['postgres_image_id'] and
         pg0['database_volume'] == runtime['database_volume'] and
         (baseline['runtime']['bot'] is None or not baseline['runtime']['bot']['running']),
         'EVIDENCE_INVALID')
    sequence = runtime['prepare_sequence']
    need(type(sequence) is list and len(sequence) == 5 and
         [item['label'] for item in sequence] == list(LABELS), 'EVIDENCE_INVALID')
    last_id = None
    previous_capture = baseline['captured_at']
    for item in sequence:
        need(set(item) == {'label','bundle_identity','settings','snapshot'}, 'EVIDENCE_INVALID')
        label = item['label']
        expected = old_summary if label.startswith('old') else bundles['B' if label == 'new-B' else 'A']
        expected_path = old_path if label.startswith('old') else Path(builds['B' if label == 'new-B' else 'A']['path'])
        need(item['bundle_identity'] == expected.identity, 'EVIDENCE_INVALID')
        phase_raw = saved(evidence,item['settings'])
        need(phase_raw == change_selection(original_raw,expected_path), 'EVIDENCE_INVALID')
        e.read_settings(evidence/item['settings']['file'])
        shot = saved_json(evidence,item['snapshot'])
        need(shot['schema_version'] == 1 and shot['kind'] == 'full' and
             shot['source_commit'] == report['source']['commit'] and
             shot['snapshot_sql_sha256'] == baseline['snapshot_sql_sha256'] and
             shot['deployment'] == {'project':runtime['project'],
                                    'compose_sha256':report['source']['compose_sha256'],
                                    'settings_sha256':item['settings']['sha256']} and
             shot['database'] == baseline['database'], 'EVIDENCE_INVALID')
        e.validate_database(shot['database'])
        bot = shot['runtime']['bot']
        pg = shot['runtime']['postgres']
        prepare = shot['runtime']['prepare']
        phase_time(previous_capture,prepare['started_at'],shot['captured_at'])
        need((bot is None or not bot['running']) and pg is not None and pg['running'] and
             all(pg[key] == pg0[key] for key in ('id','image_id','database_volume','started_at')) and
             prepare is not None and prepare['id'] != last_id and
             prepare['image_id'] == runtime['bot_image_id'] and
             prepare['status'] == 'exited' and prepare['exit_code'] == 0 and
             prepare['user'] == '10001:10001' and prepare['read_only_root'] is True and
             prepare['restart_count'] == 0 and
             prepare['artifact_mount'] == {'source':str(expected_path),'rw':False} and
             prepare['bundle_id'] == expected.identity and
             prepare['bundle_count'] == expected.count, 'EVIDENCE_INVALID')
        last_id = prepare['id']
        previous_capture = shot['captured_at']
    need(report['checks'] == {key:'PASS' for key in CHECKS}, 'EVIDENCE_INVALID')
    return report


def run(args):
    import fcntl
    head = source_head()
    clean(head)
    data = guarded(args.data_dir,directory=True)
    settings = guarded(args.settings)
    evidence = Path(args.evidence_dir)
    allowed_private_roots = private_roots()
    need(evidence.is_absolute() and '..' not in evidence.parts and
         not evidence.exists() and
         not evidence.is_symlink() and evidence.parent.is_dir() and
         all(not item.is_symlink() for item in (evidence.parent,*evidence.parent.parents)) and
         stat.S_IMODE(evidence.parent.stat().st_mode) == 0o700 and
         '.worktrees' not in evidence.parts and '.worktrees' not in data.parts and
         not evidence.is_relative_to(data) and not data.is_relative_to(evidence),
         'UNSAFE_PATH')
    need(any(all(item.is_relative_to(private_root) for item in
                 (data,evidence,settings)) for private_root in allowed_private_roots), 'UNSAFE_PATH')
    need(e.PROJECT.fullmatch(args.project) is not None and
         e.IDENTITY.fullmatch(args.old_identity) is not None, 'INVALID_ARGUMENT')
    values = e.read_settings(settings)
    need(values['ARB_DATA_DIR'] == str(data) and
         settings.stat().st_uid == os.getuid() and data.stat().st_uid == os.getuid() and
         int(values['ARB_UID']) == os.getuid() and int(values['ARB_GID']) == os.getgid() and
         stat.S_IMODE(settings.stat().st_mode) & 0o077 == 0, 'UNSAFE_SETTINGS')
    for key in ('ARB_BOT_ENV','ARB_POSTGRES_ENV'):
        secret = guarded(Path(values[key]))
        need(secret.stat().st_uid == os.getuid() and
             stat.S_IMODE(secret.stat().st_mode) & 0o077 == 0, 'UNSAFE_SETTINGS')
    old_path = guarded(Path(values['ARB_BUNDLE_DIR']),directory=True)
    need(old_path.is_relative_to(data/'bundles'), 'UNSAFE_SETTINGS')
    profile = p.load_profile(NEW)
    old_profile = p.load_profile(p.LEGACY_RELEASE)
    old = p.check_profile_bundle(old_path,old_profile,expected_identity=args.old_identity)
    need(old.identity != profile.reference['bundle_identity'] and
         old.hashes['catalog.json'] != profile.reference['catalog_sha256'],
         'PROFILE_MISMATCH')
    paths = setup_paths(data,profile.lock)
    paths['data'] = data
    glove = paths['glove']
    need(file_sha(glove) == profile.inputs['embeddings']['text_sha256'] and
         glove.stat().st_size == profile.inputs['embeddings']['text_size'], 'INPUT_CHANGED')
    lock_dir = project_lock_dir()
    guarded(lock_dir.parent,directory=True)
    lock_dir.mkdir(mode=0o700,exist_ok=True)
    lock_dir_stat = guarded(lock_dir,directory=True).stat()
    need(stat.S_IMODE(lock_dir_stat.st_mode) == 0o700 and
         lock_dir_stat.st_uid == os.getuid(), 'UNSAFE_PATH')
    lock_path = lock_dir/(args.project+'.lock')
    lock_fd = os.open(lock_path,os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    try:
        lock_stat = os.fstat(lock_fd)
        need(stat.S_ISREG(lock_stat.st_mode) and lock_stat.st_uid == os.getuid() and
             stat.S_IMODE(lock_stat.st_mode) == 0o600, 'UNSAFE_PATH')
        try:
            fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ReleaseError('RUNTIME_UNAVAILABLE') from exc
        return _run_locked(args,head,data,settings,evidence,values,old_path,old,
                           profile,paths)
    finally:
        os.close(lock_fd)


def _run_locked(args,head,data,settings,evidence,values,old_path,old,profile,paths):
    original_raw = settings.read_bytes()
    original_stat = settings.stat()
    original_state = (sha(original_raw),original_stat.st_uid,original_stat.st_gid,
                      stat.S_IMODE(original_stat.st_mode))
    evidence.mkdir(mode=0o700)
    saved_records = {}
    saved_records['profile'] = private_file(evidence/'profile.json',
        p.regular_bytes(p.PROFILE_DIR/p.PROFILE_FILES[NEW]))
    saved_records['old_profile'] = private_file(evidence/'old-profile.json',
        p.regular_bytes(p.PROFILE_DIR/p.PROFILE_FILES[p.LEGACY_RELEASE]))
    saved_records['lock'] = private_file(evidence/'release-lock.json',
        p.regular_bytes(ROOT/'src/recsys'/p.LOCK_RESOURCE))
    saved_records['original_settings'] = private_file(evidence/'original-settings.env',original_raw)
    (evidence/'original-settings.env').chmod(original_state[3])
    for key,path in (('tool',Path(__file__)),('compose',ROOT/'compose.yaml'),
                     ('builder_dockerfile',ROOT/'deploy/Dockerfile.builder'),
                     ('bot_dockerfile',ROOT/'deploy/Dockerfile.bot'),
                     ('snapshot_sql',e.SQL)):
        saved_records[key] = private_file(evidence/(key+'.source'),guarded(path).read_bytes())
    input_paths = [paths[key] for key in ('base','aliases','release','alias_raw')]
    input_paths.append(paths['glove'].parent)
    before_files = files_under(input_paths)
    saved_records['input_files'] = save_json(evidence/'input-files.json',before_files)
    saved_records['input_layout'] = save_json(evidence/'input-layout.json',
        {key:str(paths[key]) for key in ('base','aliases','release','alias_raw','glove')})
    old_files = bundle_files(old_path)
    retained = {'input_files':before_files,'input_paths':input_paths,'old_files':old_files}
    t = Transport(args.project,settings)
    t.evidence()
    switch_state = {'begun':False}
    baseline = None
    old_settings = None
    try:
        quiescent(t)
        baseline = e.snapshot(args.project,settings)
        nonempty_history(baseline['database'])
        pg0 = baseline['runtime']['postgres']
        saved_records['baseline'] = save_json(evidence/'snapshot-baseline.json',baseline)
        t.compose('--profile','tools','build','builder','bot',timeout=7200)
        clean(head)
        builder_image = t.docker('image','inspect','--format','{{.Id}}',args.project+'-builder:local').strip()
        bot_image = t.docker('image','inspect','--format','{{.Id}}',args.project+'-bot:local').strip()
        need(builder_image and bot_image, 'BUILD_FAILED')
        probe_text = t.compose('--profile','tools','run','--rm','--no-deps',
                               '--entrypoint','python','builder','-c',probe_code(),timeout=600)
        try:
            probe = json.loads(probe_text.strip())
        except ValueError as exc:
            raise ReleaseError('BUILD_MISMATCH') from exc
        check_probe(probe,profile)
        saved_records['probe'] = save_json(evidence/'producer-probe.json',probe)
        common = builder_args(paths,profile.lock)
        t.compose('--profile','tools','run','--rm','--no-deps','builder','release','validate',
                  '/data/'+str(paths['release'].relative_to(data)),*common[2:],timeout=7200)
        clean(head)
        work = data/'work'/('arb033-'+evidence.name)
        work.parent.mkdir(mode=0o700,exist_ok=True)
        need(guarded(work.parent,directory=True).stat().st_uid == os.getuid() and
             not work.exists(), 'UNSAFE_PATH')
        work.mkdir(mode=0o700)
        store = work/'export-store'
        store.mkdir(mode=0o700)
        builds = {}
        summaries = {}
        report_values = {}
        for label,block in (('A',256),('B',128)):
            output = work/('build-'+label)
            build_out = '/data/'+str(output.relative_to(data))
            glove_in = '/data/'+str(paths['glove'].relative_to(data))
            t.compose('--profile','tools','run','--rm','--no-deps','builder','release','build',
                      *common,'--glove',glove_in,'--block-size',str(block),
                      '--output-dir',build_out,timeout=7200)
            clean(head)
            report_path = output/'build-report.json'
            report_raw = guarded(report_path).read_bytes()
            report_value = p.canonical(report_raw)
            t.compose('--profile','tools','run','--rm','--no-deps','builder','release','export',
                      *common,'--neighbors',build_out+'/neighbors.json',
                      '--build-report',build_out+'/build-report.json',
                      '--output-dir','/data/'+str(store.relative_to(data)),timeout=7200)
            manifest_candidates = [item for item in store.iterdir() if item.is_dir()]
            need(len(manifest_candidates) == 1, 'BUILD_MISMATCH')
            candidate = manifest_candidates[0]
            identity = 'sha256:'+candidate.name.removeprefix('sha256-')
            summary = p.check_profile_bundle(candidate,profile,expected_identity=identity)
            final = data/'bundles'/candidate.name
            need(not final.exists() and not final.is_symlink(), 'BUILD_MISMATCH')
            candidate.rename(final)
            final.chmod(0o755)
            for name in ('catalog.json','neighbors.json','manifest.json'):
                (final/name).chmod(0o644)
            summary = p.check_profile_bundle(final,profile,expected_identity=identity)
            report_hash = check_report(report_value,report_raw,summary,profile,block)
            report_copy = private_file(evidence/('build-report-'+label+'.json'),report_raw)
            builds[label] = {'identity':identity,'path':str(final),'count':summary.count,
                             'report_sha256':report_hash,'file_sha256':summary.hashes,
                             'producer':report_value['producer'],'block_size':block,
                             'report':report_copy}
            summaries[label] = summary
            report_values[label] = report_value
        a,b = copy.deepcopy(report_values['A']),copy.deepcopy(report_values['B'])
        del a['execution']; del b['execution']
        need(a == b and builds['A']['identity'] != builds['B']['identity'] and
             summaries['A'].hashes['catalog.json'] == summaries['B'].hashes['catalog.json'] and
             summaries['A'].hashes['neighbors.json'] == summaries['B'].hashes['neighbors.json'],
             'BUILD_MISMATCH')
        ref = profile.reference
        need(builds['A']['identity'] == ref['bundle_identity'] and
             builds['A']['report_sha256'] == ref['build_report_sha256'] and
             summaries['A'].hashes['catalog.json'] == ref['catalog_sha256'] and
             summaries['A'].hashes['neighbors.json'] == ref['neighbors_sha256'],
             'BUILD_MISMATCH')
        check_retained(retained,old_path)
        settings_by_label = {}
        selected = {'old':{'path':old_path,'identity':old.identity,'count':old.count},
                    'new-A':{'path':Path(builds['A']['path']),
                             'identity':builds['A']['identity'],'count':builds['A']['count']},
                    'new-B':{'path':Path(builds['B']['path']),
                             'identity':builds['B']['identity'],'count':builds['B']['count']}}
        for label in LABELS:
            selection = selected['old' if label.startswith('old') else
                                 'new-B' if label == 'new-B' else 'new-A']
            settings_by_label[label] = evidence/('settings-'+label+'.env')
            private_file(settings_by_label[label],change_selection(original_raw,selection['path']))
        old_settings = settings_by_label['old-restore']
        for label,selection in (('A',selected['new-A']),('B',selected['new-B'])):
            t.settings = settings_by_label['new-'+label]
            quiescent(t)
            rust = t.compose('run','--rm','--no-deps','--entrypoint',
                             '/usr/local/bin/check_bundle','prepare','/artifacts',timeout=600)
            need(rust.strip().splitlines()[-1] ==
                 selection['identity']+' records='+str(selection['count']), 'BUILD_MISMATCH')
        t.settings = settings
        sequence = run_prepare_sequence(t,evidence,settings,original_state,head,baseline,
                    selected,settings_by_label,bot_image,pg0,retained,old_path,switch_state)
        t.settings = settings
        need((sha(settings.read_bytes()),settings.stat().st_uid,settings.stat().st_gid,
              stat.S_IMODE(settings.stat().st_mode)) == original_state and
             before_files == files_under(input_paths) and old_files == bundle_files(old_path),
             'INPUT_CHANGED')
        quiescent(t)
        saved_records['bundle_files'] = save_json(evidence/'bundle-files.json',
            {'old':old_files,'A':bundle_files(Path(builds['A']['path'])),
             'B':bundle_files(Path(builds['B']['path']))})
        report = {'schema_version':2,'kind':'arb-offline-release-acceptance',
            'scope':'full-inputs-offline-compatibility','status':'PASS','reason':'PASS',
            'profile':{'id':profile.id,'sha256':profile.sha256,'release_id':NEW,
                       'release_lock_sha256':profile.lock_sha256},
            'source':{'commit':head,'tool_sha256':file_sha(Path(__file__)),
                      'compose_sha256':file_sha(ROOT/'compose.yaml'),
                      'dockerfile_sha256':{'builder':file_sha(ROOT/'deploy/Dockerfile.builder'),
                                           'bot':file_sha(ROOT/'deploy/Dockerfile.bot')}},
            'original':{'settings_sha256':original_state[0],'settings_uid':original_state[1],
                        'settings_gid':original_state[2],'settings_mode':original_state[3],
                        'bundle_identity':old.identity,'bundle_path':str(old_path),
                        'bundle_file_sha256':old.hashes},
            'inputs':{'base_identity':profile.inputs['base']['identity'],
                      'alias_identity':profile.inputs['aliases']['identity'],
                      'catalog_release_identity':profile.inputs['catalog_release']['identity'],
                      'embeddings_sha256':profile.inputs['embeddings']['text_sha256']},
            'builds':builds,
            'runtime':{'project':args.project,'builder_image_id':builder_image,
                       'bot_image_id':bot_image,'postgres_image_id':pg0['image_id'],
                       'database_volume':pg0['database_volume'],
                       'baseline_snapshot_sha256':saved_records['baseline']['sha256'],
                       'prepare_sequence':sequence},
            'checks':{key:'PASS' for key in CHECKS},'live_acceptance':'PENDING',
            'production_history_preservation':'PENDING',
            'live_verifier':'REQUIRES_DATA06_V2','saved_evidence':saved_records}
        save_json(evidence/'preparation-report.json',report)
        verify_record(evidence)
        clean(head)
        return report
    except BaseException as exc:
        reason = exc.code if isinstance(exc,(ReleaseError,p.ProfileError,e.EvidenceError)) else 'BUILD_FAILED'
        restoration = 'PENDING'
        if switch_state['begun'] and old_settings is not None:
            restoration = restore_old(t,settings,original_state,old_settings,old_path,old,
                                      pg0,retained)
        try:
            save_json(evidence/'failure-status.json',{'schema_version':1,'status':'FAIL',
                      'reason':reason if reason in REASON else 'BUILD_FAILED',
                      'restoration':restoration,'live_acceptance':'PENDING'})
        except Exception:
            pass
        raise ReleaseError(reason if reason in REASON else 'BUILD_FAILED') from exc


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest='command',required=True)
    go = commands.add_parser('run')
    go.add_argument('--project',required=True)
    go.add_argument('--settings',type=Path,required=True)
    go.add_argument('--data-dir',type=Path,required=True)
    go.add_argument('--evidence-dir',type=Path,required=True)
    go.add_argument('--old-identity',required=True)
    go.add_argument('--offline',action='store_true',required=True)
    verify = commands.add_parser('verify')
    verify.add_argument('--evidence-dir',type=Path,required=True)
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'run':
            run(args)
        else:
            verify_record(args.evidence_dir)
    except (ReleaseError,p.ProfileError,e.EvidenceError,OSError,ValueError,KeyError,TypeError,
            IndexError,subprocess.SubprocessError) as exc:
        code = exc.code if isinstance(exc,(ReleaseError,p.ProfileError,e.EvidenceError)) else 'EVIDENCE_INVALID'
        print('FAIL '+(code if code in REASON or code in ('UNKNOWN_RELEASE','LOCK_SHA_MISMATCH')
                        else 'EVIDENCE_INVALID'),file=sys.stderr)
        return 1
    print('PASS full-inputs-offline-compatibility; live PENDING; production history PENDING')
    return 0


if __name__ == '__main__':
    sys.exit(main())
