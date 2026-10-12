#!/usr/bin/env python3
"""Private ARB-034 live evidence capture and dispatch. No service transitions."""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v = _load('arb034_verifier', 'live_verifier_v2.py')
e, p, r = v.e, v.p, v.r
need, Error = e.require, e.EvidenceError
ROOT = Path(__file__).resolve().parents[2]


def command(argv, *, timeout=90, input_text=None, both=False, env=None):
    try:
        result = subprocess.run(argv, cwd=ROOT, input=input_text, capture_output=True,
                                text=True, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired as exc:
        raise Error('TRANSPORT_TIMEOUT') from exc
    need(result.returncode == 0, 'TRANSPORT_FAILED')
    return result.stdout + result.stderr if both else result.stdout


def _compose(project, settings, *args, input_text=None):
    need(e.PROJECT.fullmatch(project) is not None, 'INVALID_PROJECT')
    clean = {key: value for key, value in os.environ.items()
             if not key.startswith(('ARB_', 'COMPOSE_'))}
    return command(['docker', 'compose', '--env-file', str(settings), '-p', project,
                    '-f', str(ROOT / 'compose.yaml'), *args], input_text=input_text,
                   env=clean)


def _docker(*args):
    return command(['docker', *args], both=bool(args and args[0] == 'logs'))


@contextlib.contextmanager
def _bounded_transport():
    old_compose, old_docker = e.compose, e.docker
    e.compose, e.docker = _compose, _docker
    try:
        yield
    finally:
        e.compose, e.docker = old_compose, old_docker


def _regular_sha(path):
    need(path.is_absolute() and not path.is_symlink() and path.is_file(), 'UNSAFE_PRIVATE_FILE')
    need(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, 'UNSAFE_PRIVATE_FILE')
    return e.sha(path.read_bytes())


def _settings_guard(path):
    values = e.read_settings(path)
    mode = path.stat()
    need(stat.S_IMODE(mode.st_mode) & 0o077 == 0, 'UNSAFE_SETTINGS')
    credentials = {key: _regular_sha(Path(values[key])) for key in
                   ('ARB_BOT_ENV', 'ARB_POSTGRES_ENV')}
    return {'settings': {'sha256': e.sha(path.read_bytes()), 'uid': mode.st_uid,
                         'gid': mode.st_gid, 'mode': stat.S_IMODE(mode.st_mode)},
            'credential_sha256': credentials}


def _token(path):
    raw = path.read_text()
    matches = [line.split('=', 1)[1] for line in raw.splitlines()
               if line.startswith('TELOXIDE_TOKEN=')]
    need(len(matches) == 1 and bool(matches[0]), 'TOKEN_UNAVAILABLE')
    return matches[0]


def _inventory(token):
    identifiers = _docker('ps', '-q').split()
    matches = []
    docker_pids = set()
    for identifier in identifiers:
        data = json.loads(_docker('inspect', identifier))
        need(type(data) is list and len(data) == 1, 'INVENTORY_INCOMPLETE')
        item = data[0]
        env = item.get('Config', {}).get('Env') or []
        if 'TELOXIDE_TOKEN=' + token in env:
            matches.append(item['Id'])
            docker_pids.add(item.get('State', {}).get('Pid'))
    # A known local bot process whose environment cannot be inspected makes scope incomplete.
    local = 0
    proc = Path('/proc')
    complete = proc.is_dir()
    if proc.is_dir():
        for entry in proc.iterdir():
            if not entry.name.isdigit() or int(entry.name) in docker_pids:
                continue
            try:
                cmd = (entry / 'cmdline').read_bytes()
            except (OSError, PermissionError):
                complete = False
                continue
            executable = cmd.split(b'\0', 1)[0]
            if (executable not in (b'/usr/local/bin/bot', b'bot') and
                    not executable.endswith((b'/target/release/bot',
                                             b'/target/debug/bot'))):
                continue
            try:
                cgroup = (entry / 'cgroup').read_bytes()
                if any(identifier[:12].encode() in cgroup for identifier in matches):
                    continue
                env = (entry / 'environ').read_bytes()
                if b'TELOXIDE_TOKEN=' + token.encode() + b'\0' in env:
                    local += 1
            except (OSError, PermissionError):
                complete = False
    need(len(matches) + local <= 1, 'SECOND_POLLER')
    return {'matching': len(matches) + local, 'scope_complete': complete,
            'scope': 'docker-daemon-and-known-local-processes',
            'matching_container_ids_sha256': [e.sha(item.encode()) for item in matches]}


def _guards(project, settings):
    values = e.read_settings(settings)
    result = _settings_guard(settings)
    result['daemon_id'] = _docker('info', '--format', '{{.ID}}').strip()
    need(bool(result['daemon_id']), 'DAEMON_UNAVAILABLE')
    result['pollers'] = _inventory(_token(Path(values['ARB_BOT_ENV'])))
    pg = e.container(project, settings, 'postgres')
    result['postgres_identity'] = None if pg is None else {
        'id': pg['id'], 'image_id': pg['image_id'],
        'database_volume': pg['database_volume'], 'started_at': pg['started_at']}
    result['postgres_health'] = ('absent' if pg is None else
        _docker('inspect', '--format', '{{.State.Health.Status}}', pg['id']).strip())
    return result


def capture(project, settings, *, runtime_only=False):
    need(settings.is_absolute() and settings.is_file(), 'UNSAFE_SETTINGS')
    with _bounded_transport():
        before = _guards(project, settings)
        state = e.snapshot(project, settings, runtime_only)
        after = _guards(project, settings)
    need(before == after, 'UNSTABLE_CAPTURE')
    after['stable'] = True
    return {'schema_version': 2, 'kind': 'arb-live-capture',
            'state': state, 'guards': after}


def _read(path):
    need(path.is_absolute(), 'ABSOLUTE_PATH_REQUIRED')
    e.private_parent(path)
    need(stat.S_IMODE(path.stat().st_mode) & 0o077 == 0, 'UNSAFE_PRIVATE_FILE')
    return e.read_json(path)


def _save(path, value):
    need(path.is_absolute(), 'ABSOLUTE_PATH_REQUIRED')
    return e.write_new(path, value)


def _bytes_new(path, raw, original):
    e.private_parent(path)
    need(not path.exists() and not path.is_symlink(), 'OUTPUT_EXISTS')
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
                 stat.S_IMODE(original.stat().st_mode))
    with os.fdopen(fd, 'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chown(path, original.stat().st_uid, original.stat().st_gid)


def _catalog(path):
    return p.canonical((path / 'catalog.json').read_bytes())['anime']


def make_plan(project, settings, offline_evidence, baseline_path, target_bundle, ui_probes_path,
              output):
    need(all(path.is_absolute() for path in
             (settings, offline_evidence, baseline_path, target_bundle, ui_probes_path, output)),
         'ABSOLUTE_PATH_REQUIRED')
    report = r.verify_record(offline_evidence)
    report_sha = _regular_sha(offline_evidence / 'preparation-report.json')
    baseline = _read(baseline_path)
    need(baseline.get('kind') == 'arb-live-capture' and
         baseline['state']['kind'] == 'full', 'BASELINE_MISMATCH')
    original_settings = e.read_settings(settings)
    original_path = Path(original_settings['ARB_BUNDLE_DIR'])
    old_profile, target_profile = p.load_profile(p.LEGACY_RELEASE), p.load_profile(p.NEW_RELEASE)
    old = p.check_profile_bundle(original_path, old_profile,
                                 expected_identity=report['original']['bundle_identity'])
    target = p.check_profile_bundle(target_bundle, target_profile,
                                    expected_identity=report['builds']['A']['identity'])
    need(old.hashes == report['original']['bundle_file_sha256'] and
         target.hashes == report['builds']['A']['file_sha256'] and
         target.identity == target_profile.reference['bundle_identity'] and
         project != report['runtime']['project'], 'OFFLINE_MISMATCH')
    probes = _read(ui_probes_path)
    need(probes.get('schema_version') == 1 and probes.get('kind') == 'arb-live-ui-probes' and
         set(probes) == {'schema_version', 'kind', 'alias_search', 'description_edit'},
         'INVALID_UI_PROBES')
    catalog = _catalog(target_bundle)
    alias = probes['alias_search']
    description = probes['description_edit']
    entry = catalog.get(str(alias['mal_id']), {})
    need(alias['query'] in entry.get('aliases', []) and alias['query'] != entry.get('title') and
         str(description['seed_mal_id']) in catalog and
         description['neighbor_mal_id'] in [item['mal_id'] for item in
              target.neighbors[str(description['seed_mal_id'])]] and
         bool(catalog[str(description['neighbor_mal_id'])].get('synopsis')),
         'INVALID_UI_PROBES')
    source = {'commit': baseline['state']['source_commit'],
              'compose_sha256': e.sha((ROOT / 'compose.yaml').read_bytes()),
              'snapshot_sql_sha256': e.sha(e.SQL.read_bytes()),
              'verifier_sha256': e.sha(Path(v.__file__).read_bytes()),
              'helper_sha256': e.sha(Path(__file__).read_bytes()),
              'dispatch_sha256': e.sha(Path(__file__).with_name(
                  'acceptance-evidence.py').read_bytes())}
    guard = _settings_guard(settings)
    selected = baseline['state']['runtime']['bot']
    pg = baseline['state']['runtime']['postgres']
    need(selected is not None and pg is not None and pg['running'] and
         baseline['guards']['postgres_health'] == 'healthy' and
         baseline['guards']['pollers']['matching'] == 1 and
         baseline['guards']['pollers']['scope_complete'] is True and
         selected['bundle_id'] == old.identity and selected['bundle_count'] == old.count and
         selected['artifact_mount'] == {'source': str(original_path), 'rw': False},
         'BASELINE_MISMATCH')
    target_raw = r.change_selection(settings.read_bytes(), target_bundle)
    old_copy = output.parent / 'original-settings.env'
    target_copy = output.parent / 'target-settings.env'
    need(all(not path.exists() for path in (old_copy, target_copy, output)), 'OUTPUT_EXISTS')
    def bundle(role, summary, profile, path, image):
        return {'role': role, 'identity': summary.identity, 'path': str(path),
                'release_id': profile.release_id, 'file_sha256': summary.hashes,
                'count': summary.count, 'histogram': summary.histogram,
                'expectations': profile.expectations, 'image_id': image}
    plan = {'schema_version': 2, 'kind': 'arb-live-plan',
            'offline': {'evidence_dir': str(offline_evidence), 'report_sha256': report_sha,
                        'project': report['runtime']['project'],
                        'database_volume': report['runtime']['database_volume']},
            'profile': {'id': target_profile.id, 'release_id': target_profile.release_id,
                        'sha256': target_profile.sha256,
                        'lock_sha256': target_profile.lock_sha256},
            'source': source,
            'service': {'project': project, 'daemon_id': baseline['guards']['daemon_id'],
                        'postgres_id': pg['id'], 'postgres_image_id': pg['image_id'],
                        'database_volume': pg['database_volume']},
            'settings': {**guard['settings'],
                         'credential_sha256': guard['credential_sha256'],
                         'original_copy': str(old_copy), 'target_copy': str(target_copy)},
            'expected_settings_sha256': {'original': e.sha(settings.read_bytes()),
                                         'target': e.sha(target_raw)},
            'bundles': {'original': bundle('original', old, old_profile, original_path,
                                           selected['image_id']),
                        'target': bundle('target', target, target_profile, target_bundle,
                                         report['runtime']['bot_image_id'])},
            'ui_probes': probes, 'baseline_sha256': v.digest(baseline), 'baseline': baseline}
    _bytes_new(old_copy, settings.read_bytes(), settings)
    _bytes_new(target_copy, target_raw, settings)
    v.validate_plan(plan, verify_offline=False)
    _save(output, plan)
    return plan


def _cache(directory, value):
    path = directory / (v.digest(value) + '.json')
    if path.exists():
        need(not path.is_symlink() and _read(path) == value, 'INPUT_HASH_MISMATCH')
    else:
        _save(path, value)
    return v.digest(value)


def _cached(directory, digest):
    need(type(digest) is str and e.HEX.fullmatch(digest) is not None,
         'INPUT_HASH_MISMATCH')
    item = _read(directory / (digest + '.json'))
    need(v.digest(item) == digest, 'INPUT_HASH_MISMATCH')
    return item


FAILURES = (e.EvidenceError, r.ReleaseError, p.ProfileError, OSError,
            ValueError, TypeError, KeyError, IndexError)


def _code(exc):
    return exc.code if hasattr(exc, 'code') else 'INVALID_EVIDENCE'


def _failed_operation(plan, operation, inputs, code):
    source, target = ('target', 'original') if operation == 'rollback' else (
        'original', 'target')
    return {'schema_version': 2, 'kind': 'arb-live-operation',
            'plan_sha256': v.digest(plan), 'operation': operation,
            'from': source, 'to': target, 'status': 'FAIL', 'reason': code,
            'counts': [], 'started_capture_time': None,
            'input_sha256': {key: v.digest(value) for key, value in inputs.items()}}


def _failed_case(plan, context, case, inputs, operation, code):
    role = 'original' if case == 'rollback' else 'target'
    return {'schema_version': 2, 'kind': 'arb-live-case',
            'plan_sha256': v.digest(plan), 'context_sha256': v.digest(context),
            'case': case, 'status': 'FAIL', 'reason': code,
            'bundle_role': role, 'bundle_identity': plan['bundles'][role]['identity'],
            'counts': {}, 'input_sha256': {key: v.digest(value) for key, value in
                                         inputs.items()},
            'operation_sha256': v.digest(operation) if operation else None}


def _verify_saved_operation(plan, item, directory):
    need(item.get('kind') == 'arb-live-operation' and
         item.get('plan_sha256') == v.digest(plan), 'INVALID_OPERATION')
    inputs = {key: _cached(directory, value) for key, value in item['input_sha256'].items()}
    need(set(inputs) == {'before', 'stopped', 'prepared', 'started'}, 'INVALID_OPERATION')
    if item.get('status') == 'FAIL':
        try:
            v.verify_operation(plan, item['operation'], **inputs)
        except FAILURES as exc:
            need(item == _failed_operation(plan, item['operation'], inputs, _code(exc)),
                 'FORGED_OPERATION')
        else:
            need(False, 'FORGED_OPERATION')
    else:
        need(v.verify_operation(plan, item['operation'], **inputs) == item,
             'FORGED_OPERATION')
    return inputs


def _verify_saved_case(plan, context, item, directory, operations):
    need(item.get('kind') == 'arb-live-case' and
         item.get('context_sha256') == v.digest(context), 'INVALID_CASE')
    inputs = {key: _cached(directory, value) for key, value in item['input_sha256'].items()}
    op = operations.get('final-update' if item['case'] == 'catalog-stale' else item['case'])
    if item.get('status') == 'FAIL':
        try:
            v.verify_case(plan, context, item['case'], inputs['before'], inputs['after'],
                          during=inputs.get('during'), operation=op)
        except FAILURES as exc:
            need(item == _failed_case(plan, context, item['case'], inputs, op, _code(exc)),
                 'FORGED_CASE')
        else:
            need(False, 'FORGED_CASE')
    else:
        expected = v.verify_case(plan, context, item['case'], inputs['before'], inputs['after'],
                                 during=inputs.get('during'), operation=op)
        need(expected == item, 'FORGED_CASE')
    return inputs


def parser():
    top = argparse.ArgumentParser(description=__doc__)
    commands = top.add_subparsers(dest='command', required=True)
    cap = commands.add_parser('capture')
    for arg in ('project', 'settings', 'output'):
        cap.add_argument('--' + arg, required=True, type=Path if arg != 'project' else str)
    cap.add_argument('--plan', type=Path)
    cap.add_argument('--runtime-only', action='store_true')
    plan = commands.add_parser('plan')
    for arg in ('project', 'settings', 'offline-evidence', 'baseline', 'target-bundle',
                'ui-probes', 'output'):
        plan.add_argument('--' + arg, required=True, type=str if arg == 'project' else Path)
    op = commands.add_parser('verify-operation')
    op.add_argument('--operation', choices=v.OPERATIONS, required=True)
    for arg in ('plan', 'before', 'stopped', 'prepared', 'started', 'output'):
        op.add_argument('--' + arg, type=Path, required=True)
    bind = commands.add_parser('bind-actor')
    for arg in ('plan', 'initial-operation', 'after', 'output'):
        bind.add_argument('--' + arg, type=Path, required=True)
    case = commands.add_parser('verify')
    case.add_argument('--case', choices=v.CASES, required=True)
    for arg in ('context', 'before', 'after', 'output'):
        case.add_argument('--' + arg, type=Path, required=True)
    case.add_argument('--during', type=Path)
    case.add_argument('--operation', type=Path)
    summary = commands.add_parser('summarize')
    for arg in ('plan', 'operations-dir', 'results-dir', 'final', 'output', 'public-output'):
        summary.add_argument('--' + arg, type=Path, required=True)
    summary.add_argument('--context', type=Path)
    summary.add_argument('--observations', type=Path)
    return top


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == 'capture':
            if args.plan:
                plan = v.validate_plan(_read(args.plan), verify_offline=False)
                need(args.project == plan['service']['project'], 'WRONG_PROJECT')
            value = capture(args.project, args.settings, runtime_only=args.runtime_only)
            if args.plan: v._valid_capture(value, plan, full=not args.runtime_only)
            _save(args.output, value)
        elif args.command == 'plan':
            make_plan(args.project, args.settings, args.offline_evidence, args.baseline,
                      args.target_bundle, args.ui_probes, args.output)
        elif args.command == 'verify-operation':
            plan = v.validate_plan(_read(args.plan), verify_offline=False)
            values = {key: _read(getattr(args, key)) for key in
                      ('before', 'stopped', 'prepared', 'started')}
            try:
                result = v.verify_operation(plan, args.operation, **values)
            except FAILURES as exc:
                result = _failed_operation(plan, args.operation, values, _code(exc))
            directory = args.output.parent / 'inputs'
            need(directory.is_dir(), 'PRIVATE_DIRECTORY_REQUIRED')
            for value in values.values(): _cache(directory, value)
            _save(args.output, result)
            if result['status'] == 'FAIL':
                print('FAIL ' + result['reason'], file=sys.stderr)
                return 1
        elif args.command == 'bind-actor':
            plan = v.validate_plan(_read(args.plan), verify_offline=False)
            operation = _read(args.initial_operation)
            directory = args.initial_operation.parent / 'inputs'
            inputs = _verify_saved_operation(plan, operation, directory)
            after = _read(args.after)
            context = v.bind_context(plan, operation, inputs['started'], after)
            context['plan_path'] = str(args.plan)
            context_inputs = args.output.parent / 'inputs'
            need(context_inputs.is_dir(), 'PRIVATE_DIRECTORY_REQUIRED')
            _cache(context_inputs, inputs['started'])
            _cache(context_inputs, after)
            _save(args.output, context)
        elif args.command == 'verify':
            context = _read(args.context)
            plan_path = Path(context['plan_path'])
            plan = v.validate_plan(_read(plan_path), verify_offline=False)
            inputs = {'before': _read(args.before), 'after': _read(args.after)}
            if args.during: inputs['during'] = _read(args.during)
            operation = _read(args.operation) if args.operation else None
            if operation:
                _verify_saved_operation(plan, operation, args.operation.parent / 'inputs')
            try:
                result = v.verify_case(plan, context, args.case,
                                       inputs['before'], inputs['after'],
                                       during=inputs.get('during'), operation=operation)
            except FAILURES as exc:
                result = _failed_case(plan, context, args.case, inputs, operation,
                                      _code(exc))
            directory = args.output.parent / 'inputs'
            need(directory.is_dir(), 'PRIVATE_DIRECTORY_REQUIRED')
            for value in inputs.values(): _cache(directory, value)
            _save(args.output, result)
            if result['status'] == 'FAIL':
                print('FAIL ' + result['reason'], file=sys.stderr)
                return 1
        elif args.command == 'summarize':
            plan = v.validate_plan(_read(args.plan))
            context = _read(args.context) if args.context else None
            observations = _read(args.observations) if args.observations else None
            operations = {}
            operation_inputs = {}
            for path in args.operations_dir.glob('*.json'):
                item = _read(path)
                need(item['operation'] not in operations, 'DUPLICATE_OPERATION')
                operation_inputs[item['operation']] = _verify_saved_operation(
                    plan, item, args.operations_dir / 'inputs')
                operations[item['operation']] = item
            if context is not None:
                need(context.get('plan_path') == str(args.plan) and
                     context.get('initial_operation_sha256') ==
                     v.digest(operations['initial-update']), 'INVALID_CONTEXT')
                anchors = {key: _cached(args.context.parent / 'inputs', value) for key, value in
                           context['anchor_input_sha256'].items()}
                rebound = v.bind_context(plan, operations['initial-update'],
                                         anchors['before'], anchors['after'])
                rebound['plan_path'] = str(args.plan)
                need(rebound == context, 'FORGED_CONTEXT')
            results = {}
            case_inputs = {}
            for path in args.results_dir.glob('*.json'):
                item = _read(path)
                need(item['case'] not in results and context is not None, 'DUPLICATE_CASE')
                case_inputs[item['case']] = _verify_saved_case(
                    plan, context, item, args.results_dir / 'inputs', operations)
                results[item['case']] = item
            final = _read(args.final)
            value = v.summarize_evidence(plan, operations, results, final, context,
                                         observations, operation_inputs, case_inputs)
            _save(args.output, value)
            public = {key: value[key] for key in
                      ('schema_version', 'kind', 'offline_compatibility', 'operations',
                       'production_history_preservation', 'machine', 'live_ui_acceptance',
                       'live_acceptance', 'selected', 'poller_scope', 'limits')}
            _save(args.public_output, public)
            print(value['live_acceptance'])
            return 0 if value['live_acceptance'] == 'PASS' else 1
        print('PASS')
        return 0
    except (Error, r.ReleaseError, p.ProfileError, OSError, ValueError, TypeError,
            KeyError, IndexError, subprocess.SubprocessError) as exc:
        code = _code(exc)
        print('FAIL ' + code, file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
