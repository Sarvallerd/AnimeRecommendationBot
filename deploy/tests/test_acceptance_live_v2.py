"""Small synthetic ARB-034 verifier checks; no full inputs or Docker."""
import copy
import datetime as dt
import importlib.util
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('live_verifier_v2.py')
spec = importlib.util.spec_from_file_location('arb034_test_verifier', SCRIPT)
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
A = 'sha256:' + 'a' * 64
B = 'sha256:' + 'b' * 64
STAMP = dt.datetime(2026, 10, 12, tzinfo=dt.timezone.utc)


def time(minute):
    return (STAMP + dt.timedelta(minutes=minute)).isoformat()


def bot(identity, role, identifier, started):
    return {'id': identifier, 'image_id': 'image-' + role, 'project': 'real-project',
            'service': 'bot', 'user': '10001:10001', 'status': 'running',
            'running': True, 'exit_code': 0, 'started_at': time(started),
            'restart_count': 0, 'read_only_root': True,
            'artifact_mount': {'source': '/tmp/' + role, 'rw': False},
            'database_volume': None, 'bundle_id': identity,
            'bundle_count': 2 if role == 'original' else 3}


def database():
    return {'migrations': [{'version': 1, 'applied_at': time(0)}],
            'users': [{'tg_id': 42, 'created_at': time(0), 'updated_at': time(0)}],
            'requests': [{'id': 1, 'tg_id': 42, 'action_key': 'msg:42:1',
                          'raw_query': 'Old', 'seed_mal_id': 1, 'bundle_id': A,
                          'created_at': time(0), 'resolved_at': time(0)}],
            'delivered': [{'id': 1, 'request_id': 1, 'tg_id': 42, 'rank': 1,
                           'mal_id': 4, 'chat_id': 42, 'message_id': 10,
                           'delivered_at': time(0)}],
            'anime_events': [], 'anime_current': [], 'recommendation_ratings': [],
            'feedback': [], 'legacy_users': [{'old': 1}],
            'legacy_requests': [], 'legacy_feedback': []}


def plan():
    return {'schema_version': 2, 'kind': 'arb-live-plan',
            'service': {'project': 'real-project', 'daemon_id': 'daemon',
                        'postgres_id': 'pg', 'postgres_image_id': 'pg-image',
                        'database_volume': 'real-volume'},
            'source': {'commit': 'f' * 40, 'compose_sha256': 'c' * 64,
                       'snapshot_sql_sha256': 'd' * 64},
            'settings': {'uid': 1000, 'gid': 1000, 'mode': 0o600,
                         'credential_sha256': {'ARB_BOT_ENV': 'e' * 64,
                                               'ARB_POSTGRES_ENV': 'f' * 64}},
            'expected_settings_sha256': {'original': '1' * 64, 'target': '2' * 64},
            'bundles': {
                'original': {'identity': A, 'path': '/tmp/original', 'count': 2,
                             'image_id': 'image-original', 'expectations':
                             {'smoke_seed': {'mal_id': 1, 'query': 'Old',
                                             'neighbor_ids': [4]},
                              'empty_seed': None}, 'histogram': {'0': 0}},
                'target': {'identity': B, 'path': '/tmp/target', 'count': 3,
                           'image_id': 'image-target', 'expectations':
                           {'smoke_seed': {'mal_id': 8, 'query': 'New',
                                           'neighbor_ids': [9, 10]},
                            'empty_seed': None}, 'histogram': {'0': 0}}},
            'ui_probes': {'schema_version': 1, 'kind': 'arb-live-ui-probes',
                          'alias_search': {'query': 'Alias', 'mal_id': 8},
                          'description_edit': {'seed_mal_id': 8,
                                               'query': 'New',
                                               'neighbor_mal_id': 9}},
            'offline': {'report_sha256': '3' * 64},
            'profile': {'id': 'new', 'sha256': '4' * 64, 'lock_sha256': '5' * 64}}


def shot(minute, role='original', bot_item=None, prepare=None, settings=None, db=None,
         matching=None):
    settings = settings or ('1' * 64 if role == 'original' else '2' * 64)
    if bot_item is ...:
        bot_item = bot(A if role == 'original' else B, role,
                       'old-bot' if role == 'original' else 'new-bot', 0 if role == 'original' else 2.5)
    if matching is None: matching = 1 if bot_item is not None else 0
    return {'schema_version': 2, 'kind': 'arb-live-capture',
            'state': {'schema_version': 1, 'kind': 'full', 'captured_at': time(minute),
                      'source_commit': 'f' * 40, 'snapshot_sql_sha256': 'd' * 64,
                      'deployment': {'project': 'real-project', 'compose_sha256': 'c' * 64,
                                     'settings_sha256': settings},
                      'runtime': {'bot': bot_item, 'prepare': prepare,
                                  'postgres': {'id': 'pg', 'image_id': 'pg-image',
                                               'database_volume': 'real-volume',
                                               'running': True}},
                      'database': copy.deepcopy(db or database())},
            'guards': {'stable': True, 'daemon_id': 'daemon',
                       'settings': {'sha256': settings, 'uid': 1000, 'gid': 1000,
                                    'mode': 0o600},
                       'credential_sha256': {'ARB_BOT_ENV': 'e' * 64,
                                             'ARB_POSTGRES_ENV': 'f' * 64},
                       'postgres_health': 'healthy',
                       'postgres_identity': {'id': 'pg', 'image_id': 'pg-image',
                                             'database_volume': 'real-volume',
                                             'started_at': None},
                       'pollers': {'matching': matching, 'scope_complete': True,
                                   'scope': 'docker-daemon-and-known-local-processes'}}}


def sequence():
    before = shot(0, bot_item=bot(A, 'original', 'old-bot', -1))
    stopped = shot(1, bot_item=None, matching=0)
    prepared = shot(2, role='target', bot_item=None, matching=0)
    prep = bot(B, 'target', 'prepare-new', 1.5)
    prep.update(service='prepare', status='exited', running=False)
    prepared['state']['runtime']['prepare'] = prep
    started = shot(3, role='target', bot_item=bot(B, 'target', 'new-bot', 2.5),
                   prepare=copy.deepcopy(prep))
    return before, stopped, prepared, started


class LiveV2Tests(unittest.TestCase):
    def fail_code(self, code, fn):
        with self.assertRaises(v.Error) as caught: fn()
        self.assertEqual(caught.exception.code, code)

    def test_operation_recomputes_distinct_bundle_transition(self):
        values = sequence()
        result = v.verify_operation(plan(), 'initial-update', *values)
        self.assertEqual(result['counts'], [1, 0, 0, 1])
        self.assertEqual(result['to'], 'target')
        changed = copy.deepcopy(values)
        changed[2]['state']['runtime']['prepare']['exit_code'] = 1
        self.fail_code('PREPARE_GATE', lambda: v.verify_operation(plan(),
                                                             'initial-update', *changed))
        changed = copy.deepcopy(values)
        changed[1]['guards']['pollers']['matching'] = 2
        self.fail_code('SECOND_POLLER', lambda: v.verify_operation(plan(),
                                                               'initial-update', *changed))
        changed = copy.deepcopy(values)
        changed[1]['guards']['pollers']['scope_complete'] = False
        self.fail_code('POLLER_SCOPE_PENDING', lambda: v.verify_operation(plan(),
                                                                     'initial-update', *changed))
        changed = copy.deepcopy(values)
        changed[2]['guards']['postgres_identity']['id'] = 'replacement'
        self.fail_code('DATABASE_VOLUME_CHANGED', lambda: v.verify_operation(
            plan(), 'initial-update', *changed))
        changed = copy.deepcopy(values)
        changed[3]['state']['database']['migrations'][0]['applied_at'] = time(1)
        self.fail_code('DATABASE_HISTORY_CHANGED', lambda: v.verify_operation(plan(),
                                                                          'initial-update', *changed))
        changed = copy.deepcopy(values)
        changed[3]['state']['runtime']['bot']['started_at'] = time(1.5)
        self.fail_code('BOT_NOT_RECREATED', lambda: v.verify_operation(plan(),
                                                                  'initial-update', *changed))

    def test_real_flow_actor_order_and_private_chat(self):
        before, stopped, prepared, started = sequence()
        operation = v.verify_operation(plan(), 'initial-update', before, stopped,
                                       prepared, started)
        after = copy.deepcopy(started)
        after['state']['captured_at'] = time(4)
        after['state']['database']['requests'].append({
            'id': 2, 'tg_id': 42, 'action_key': 'msg:42:2', 'raw_query': 'New',
            'seed_mal_id': 8, 'bundle_id': B, 'created_at': time(4),
            'resolved_at': time(4)})
        for rank, target in enumerate((9, 10), 1):
            after['state']['database']['delivered'].append({
                'id': rank + 1, 'request_id': 2, 'tg_id': 42, 'rank': rank,
                'mal_id': target, 'chat_id': 42, 'message_id': rank + 10,
                'delivered_at': time(4)})
        bundles = {'original': {'id': A, 'neighbors': {'1': [{'mal_id': 4}]}},
                   'target': {'id': B, 'neighbors': {'8': [{'mal_id': 9},
                                                          {'mal_id': 10}]}}}
        with mock.patch.object(v, '_bundle', side_effect=lambda _plan, role:
                               (bundles[role], None, None)):
            context = v.bind_context(plan(), operation, started, after)
            self.assertEqual(context['actor']['target_anchor_request_id'], 2)
            stopped_anchor = copy.deepcopy(started)
            stopped_anchor['state']['runtime']['bot']['running'] = False
            self.fail_code('BOT_NOT_RUNNING', lambda: v.bind_context(
                plan(), operation, stopped_anchor, after))
            unpolled_anchor = copy.deepcopy(after)
            unpolled_anchor['guards']['pollers']['matching'] = 0
            self.fail_code('POLLER_COUNT', lambda: v.bind_context(
                plan(), operation, started, unpolled_anchor))
            result = v.verify_case(plan(), context, 'initial-update', before, after,
                                   operation=operation)
            self.assertEqual(result['counts']['delivered'], 2)
            wrong = copy.deepcopy(after)
            wrong['state']['database']['delivered'][-1]['mal_id'] = 9
            self.fail_code('DELIVERY_ORDER', lambda: v.bind_context(plan(), operation,
                                                                  started, wrong))
            wrong = copy.deepcopy(after)
            wrong['state']['database']['requests'][-1]['action_key'] = 'msg:55:2'
            self.fail_code('INVALID_ACTION_KEY', lambda: v.bind_context(plan(), operation,
                                                                     started, wrong))
            wrong = copy.deepcopy(after)
            wrong['state']['database']['requests'].append(copy.deepcopy(
                wrong['state']['database']['requests'][-1]))
            wrong['state']['database']['requests'][-1]['id'] = 3
            wrong['state']['database']['requests'][-1]['action_key'] = 'msg:42:3'
            self.fail_code('FLOW_AMBIGUOUS', lambda: v.bind_context(plan(), operation,
                                                                 started, wrong))
            empty_before = copy.deepcopy(after)
            empty_after = copy.deepcopy(after)
            empty_after['state']['captured_at'] = time(5)
            self.assertEqual(v.verify_case(plan(), context, 'empty', empty_before,
                                           empty_after)['status'], 'NOT_APPLICABLE')
            description = v.verify_case(plan(), context, 'description-edit',
                                        empty_before, empty_after)
            self.assertEqual(description['counts']['requests'], 0)
            alias_after = copy.deepcopy(after)
            alias_after['state']['captured_at'] = time(5)
            alias_after['state']['database']['requests'].append({
                'id': 3, 'tg_id': 42, 'action_key': 'msg:42:3', 'raw_query': 'Alias',
                'seed_mal_id': 8, 'bundle_id': B, 'created_at': time(5),
                'resolved_at': time(5)})
            for rank, target in enumerate((9, 10), 1):
                alias_after['state']['database']['delivered'].append({
                    'id': rank + 3, 'request_id': 3, 'tg_id': 42, 'rank': rank,
                    'mal_id': target, 'chat_id': 42, 'message_id': rank + 20,
                    'delivered_at': time(5)})
            self.assertEqual(v.verify_case(plan(), context, 'alias-search', after,
                                           alias_after)['status'], 'PASS')
            wrong_alias = copy.deepcopy(alias_after)
            wrong_alias['state']['database']['requests'][-1]['raw_query'] = 'New'
            self.fail_code('WRONG_SELECTION', lambda: v.verify_case(
                plan(), context, 'alias-search', after, wrong_alias))
            false_plan = plan()
            false_plan['bundles']['target']['histogram']['0'] = 1
            false_context = copy.deepcopy(context)
            false_context['plan_sha256'] = v.digest(false_plan)
            self.fail_code('FALSE_EMPTY_NA', lambda: v.verify_case(
                false_plan, false_context, 'empty', empty_before, empty_after))

    def test_case_windows_require_running_bot_and_one_poller(self):
        value = plan()
        context = {'schema_version': 2, 'kind': 'arb-live-context',
                   'plan_sha256': v.digest(value),
                   'actor': {'tg_id': 42, 'chat_id': 42}}
        live_bot = bot(B, 'target', 'target-bot', 2.5)
        before = shot(4, 'target', bot_item=live_bot)
        after = shot(6, 'target', bot_item=live_bot)
        during = copy.deepcopy(before)
        during['state']['kind'] = 'runtime_only'
        during['state']['captured_at'] = time(5)
        during['state']['database'] = None
        during['state']['runtime']['postgres']['running'] = False
        during['guards']['postgres_health'] = 'unhealthy'
        with mock.patch.object(v, '_bundle', return_value=(
                {'id': B, 'neighbors': {}}, None, None)):
            self.assertEqual(v.verify_case(value, context, 'navigation-search',
                                           before, after)['status'], 'PASS')
            self.assertEqual(v.verify_case(value, context, 'db-outage', before,
                                           after, during=during)['status'], 'PASS')
            for endpoint in ('before', 'after'):
                for field, replacement, code in (
                        ('running', False, 'BOT_NOT_RUNNING'),
                        ('matching', 0, 'POLLER_COUNT')):
                    changed = {'before': copy.deepcopy(before),
                               'after': copy.deepcopy(after)}
                    if field == 'running':
                        changed[endpoint]['state']['runtime']['bot']['running'] = replacement
                    else:
                        changed[endpoint]['guards']['pollers']['matching'] = replacement
                    self.fail_code(code, lambda: v.verify_case(
                        value, context, 'navigation-search', **changed))
            for field, replacement, code in (
                    ('running', False, 'BOT_NOT_RUNNING'),
                    ('matching', 0, 'POLLER_COUNT')):
                changed = copy.deepcopy(during)
                if field == 'running':
                    changed['state']['runtime']['bot']['running'] = replacement
                else:
                    changed['guards']['pollers']['matching'] = replacement
                self.fail_code(code, lambda: v.verify_case(
                    value, context, 'db-outage', before, after, during=changed))

    def test_description_coordinates_are_actor_owned_private_delivery(self):
        value = plan()
        context = {'schema_version': 2, 'kind': 'arb-live-context',
                   'plan_sha256': v.digest(value),
                   'actor': {'tg_id': 42, 'chat_id': 42}}
        db = database()
        db['users'].append({'tg_id': 55, 'created_at': time(0),
                            'updated_at': time(0)})
        db['requests'].append({'id': 2, 'tg_id': 55, 'action_key': 'msg:55:2',
                               'raw_query': 'New', 'seed_mal_id': 8,
                               'bundle_id': B, 'created_at': time(1),
                               'resolved_at': time(1)})
        db['delivered'].append({'id': 2, 'request_id': 2, 'tg_id': 55,
                                'rank': 1, 'mal_id': 9, 'chat_id': 55,
                                'message_id': 11, 'delivered_at': time(1)})
        live_bot = bot(B, 'target', 'target-bot', 2.5)
        def window(rows):
            return (shot(4, 'target', bot_item=live_bot, db=rows),
                    shot(5, 'target', bot_item=live_bot, db=rows))
        with mock.patch.object(v, '_bundle', return_value=(
                {'id': B, 'neighbors': {'8': [{'mal_id': 9}]}}, None, None)):
            before, after = window(db)
            self.fail_code('DESCRIPTION_COORDINATES', lambda: v.verify_case(
                value, context, 'description-edit', before, after))
            own = copy.deepcopy(db)
            own['requests'][-1].update(tg_id=42, action_key='msg:42:2')
            own['delivered'][-1].update(tg_id=42, chat_id=42)
            before, after = window(own)
            self.assertEqual(v.verify_case(value, context, 'description-edit',
                                           before, after)['status'], 'PASS')
            public_chat = copy.deepcopy(own)
            public_chat['delivered'][-1]['chat_id'] = 55
            before, after = window(public_chat)
            self.fail_code('DESCRIPTION_COORDINATES', lambda: v.verify_case(
                value, context, 'description-edit', before, after))
            foreign_delivery = copy.deepcopy(db)
            foreign_delivery['requests'].append({
                'id': 3, 'tg_id': 42, 'action_key': 'msg:42:3',
                'raw_query': 'New', 'seed_mal_id': 8, 'bundle_id': B,
                'created_at': time(1), 'resolved_at': time(1)})
            before, after = window(foreign_delivery)
            self.fail_code('DESCRIPTION_COORDINATES', lambda: v.verify_case(
                value, context, 'description-edit', before, after))

    def test_history_mutation_and_missing_ui_never_pass(self):
        old = database()
        later = copy.deepcopy(old)
        later['legacy_users'][0]['old'] = 2
        self.fail_code('LEGACY_CHANGED', lambda: v._preserved(old, later, 42))
        later = copy.deepcopy(old)
        later['requests'][0]['created_at'] = time(1)
        self.fail_code('OLD_REQUEST_CHANGED', lambda: v._preserved(old, later, 42))
        self.assertEqual(len(v.CASES), 21)
        final = shot(4, 'target', bot_item=bot(B, 'target', 'new-bot', 2.5))
        value = v.summarize_evidence(plan(), {}, {}, final)
        self.assertEqual(value['live_acceptance'], 'PENDING')
        self.assertEqual(value['production_history_preservation'], 'PENDING')

    def test_plan_binds_reference_a_and_separate_historical_source(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            report_path = parent / 'preparation-report.json'
            report_path.write_bytes(b'{}\n')
            original = parent / 'original.env'
            target = parent / 'target.env'
            prefix = (b'ARB_DATA_DIR=/tmp/data\nARB_UID=1000\nARB_GID=1000\n'
                      b'ARB_LOG=info\nARB_BOT_ENV=/tmp/bot.env\n'
                      b'ARB_POSTGRES_ENV=/tmp/postgres.env\n')
            original.write_bytes(prefix + b'ARB_BUNDLE_DIR=/tmp/original\n')
            target.write_bytes(prefix + b'ARB_BUNDLE_DIR=/tmp/target\n')
            original.chmod(0o600)
            target.chmod(0o600)
            value = plan()
            value['profile'] = {'id': v.p.NEW_RELEASE, 'release_id': v.p.NEW_RELEASE,
                                'sha256': v.p.TRUSTED_NEW_PROFILE_SHA,
                                'lock_sha256': v.p.TRUSTED_LOCK_SHA}
            value['offline'].update(evidence_dir=str(parent), project='offline-project',
                                    database_volume='offline-volume',
                                    report_sha256=v.sha(report_path.read_bytes()))
            value['source'].update(verifier_sha256=v.sha(SCRIPT.read_bytes()),
                                   helper_sha256=v.sha(SCRIPT.with_name(
                                       'live_evidence_v2.py').read_bytes()),
                                   dispatch_sha256=v.sha(SCRIPT.with_name(
                                       'acceptance-evidence.py').read_bytes()),
                                   compose_sha256=v.sha((v.e.ROOT / 'compose.yaml').read_bytes()),
                                   snapshot_sql_sha256=v.sha(v.e.SQL.read_bytes()))
            value['settings'].update(uid=os.getuid(), gid=os.getgid(), mode=0o600,
                                     original_copy=str(original), target_copy=str(target))
            value['expected_settings_sha256'] = {
                'original': v.sha(original.read_bytes()),
                'target': v.sha(target.read_bytes())}
            base = shot(0, bot_item=bot(A, 'original', 'old-bot', -1))
            base['guards']['settings'].update(uid=os.getuid(), gid=os.getgid(),
                                               sha256=value['expected_settings_sha256']['original'])
            base['state']['deployment']['settings_sha256'] = value['expected_settings_sha256']['original']
            base['state']['snapshot_sql_sha256'] = value['source']['snapshot_sql_sha256']
            base['state']['deployment']['compose_sha256'] = value['source']['compose_sha256']
            value['baseline'] = base
            value['baseline_sha256'] = v.digest(base)
            old_summary = SimpleNamespace(identity=A, hashes={'catalog.json': '1'}, count=2,
                                          histogram={'0': 0})
            target_summary = SimpleNamespace(identity=B,
                hashes={'catalog.json': '2', 'neighbors.json': '3'}, count=3,
                histogram={'0': 0})
            old_profile = SimpleNamespace(release_id=v.p.LEGACY_RELEASE)
            target_profile = SimpleNamespace(release_id=v.p.NEW_RELEASE,
                reference={'bundle_identity': B, 'catalog_sha256': '2',
                           'neighbors_sha256': '3'})
            report = {'source': {'commit': 'a' * 40},
                      'runtime': {'project': 'offline-project',
                                  'database_volume': 'offline-volume',
                                  'bot_image_id': 'image-target'},
                      'profile': {'sha256': v.p.TRUSTED_NEW_PROFILE_SHA,
                                  'release_lock_sha256': v.p.TRUSTED_LOCK_SHA},
                      'original': {'bundle_identity': A,
                                   'bundle_file_sha256': old_summary.hashes},
                      'builds': {'A': {'identity': B,
                                       'file_sha256': target_summary.hashes}}}
            def synthetic_bundle(_plan, role):
                summary = old_summary if role == 'original' else target_summary
                profile = old_profile if role == 'original' else target_profile
                return {'id': summary.identity, 'neighbors': {}}, summary, profile
            with mock.patch.object(v, '_bundle', side_effect=synthetic_bundle), \
                 mock.patch.object(v, '_check_probes'), \
                 mock.patch.object(v.r, 'verify_record', return_value=report):
                self.assertIs(v.validate_plan(value), value)
                self.assertNotEqual(report['source']['commit'], value['source']['commit'])
                alternate = copy.deepcopy(value)
                alternate['bundles']['target']['identity'] = 'sha256:' + 'c' * 64
                self.fail_code('REFERENCE_MISMATCH', lambda: v.validate_plan(alternate))
                alternate = copy.deepcopy(value)
                alternate['profile']['lock_sha256'] = '0' * 64
                self.fail_code('PROFILE_MISMATCH', lambda: v.validate_plan(alternate))
                alternate = copy.deepcopy(value)
                alternate['offline']['report_sha256'] = '0' * 64
                self.fail_code('OFFLINE_CHANGED', lambda: v.validate_plan(alternate))

    def test_three_window_history_and_final_runtime(self):
        initial = sequence()
        first_target = bot(B, 'target', 'new-bot', 2.5)
        rollback_before = shot(5, 'target', bot_item=first_target)
        rollback_stopped = shot(6, 'target', bot_item=None, matching=0)
        rollback_prepared = shot(7, 'original', bot_item=None, matching=0)
        old_prepare = bot(A, 'original', 'old-prepare-2', 6.5)
        old_prepare.update(service='prepare', status='exited', running=False)
        rollback_prepared['state']['runtime']['prepare'] = old_prepare
        second_old = bot(A, 'original', 'old-bot-2', 7.5)
        rollback_started = shot(8, 'original', bot_item=second_old,
                                prepare=copy.deepcopy(old_prepare))
        final_before = shot(9, 'original', bot_item=second_old)
        final_stopped = shot(10, 'original', bot_item=None, matching=0)
        final_prepared = shot(11, 'target', bot_item=None, matching=0)
        target_prepare = bot(B, 'target', 'target-prepare-3', 10.5)
        target_prepare.update(service='prepare', status='exited', running=False)
        final_prepared['state']['runtime']['prepare'] = target_prepare
        final_bot = bot(B, 'target', 'new-bot-3', 11.5)
        final_started = shot(12, 'target', bot_item=final_bot,
                             prepare=copy.deepcopy(target_prepare))
        inputs = {'initial-update': dict(zip(('before', 'stopped', 'prepared', 'started'),
                                             initial)),
                  'rollback': dict(zip(('before', 'stopped', 'prepared', 'started'),
                                       (rollback_before, rollback_stopped,
                                        rollback_prepared, rollback_started))),
                  'final-update': dict(zip(('before', 'stopped', 'prepared', 'started'),
                                           (final_before, final_stopped,
                                            final_prepared, final_started)))}
        value = plan()
        value['baseline'] = initial[0]
        ops = {name: v.verify_operation(value, name, **items) for name, items in
               inputs.items()}
        final = shot(13, 'target', bot_item=final_bot,
                     prepare=copy.deepcopy(target_prepare))
        context = {'actor': {'tg_id': 42}}
        summary = v.summarize_evidence(value, ops, {}, final, context,
                                       operation_inputs=inputs)
        self.assertEqual(summary['production_history_preservation'], 'PASS')
        self.assertEqual(summary['live_acceptance'], 'PENDING')
        changed = copy.deepcopy(final)
        changed['state']['database']['migrations'][0]['applied_at'] = time(1)
        self.fail_code('OLD_HISTORY_CHANGED', lambda: v.summarize_evidence(
            value, ops, {}, changed, context, operation_inputs=inputs))
        changed = copy.deepcopy(inputs)
        changed['final-update']['started']['state']['runtime']['bot']['id'] = 'new-bot'
        self.fail_code('OPERATION_CHAIN', lambda: v.summarize_evidence(
            value, ops, {}, final, context, operation_inputs=changed))

    def test_ui_pass_needs_observation_and_verified_case_reference(self):
        value = plan()
        context = {'actor': {'tg_id': 42}, 'plan_sha256': v.digest(value)}
        result = {'case': 'initial-update', 'status': 'PASS'}
        cases = {'initial-update': result}
        final = shot(4, 'target', bot_item=bot(B, 'target', 'new-bot', 2.5))
        entries = [{'id': identifier, 'status': 'PENDING', 'observed_at': None,
                    'observation': '', 'case_sha256': []} for identifier in v.UI_IDS]
        observations = {'schema_version': 2, 'kind': 'arb-live-observations',
                        'plan_sha256': v.digest(value),
                        'context_sha256': v.digest(context), 'scenarios': entries}
        summary = v.summarize_evidence(value, {}, cases, final, context, observations)
        self.assertEqual(summary['live_ui_acceptance'], 'PENDING')
        entries[0].update(status='PASS', observation='One target search was observed',
                          case_sha256=[v.digest(result)])
        self.fail_code('UNOBSERVED_UI_PASS', lambda: v.summarize_evidence(
            value, {}, cases, final, context, observations))
        entries[0]['observed_at'] = time(3)
        entries[0]['case_sha256'] = ['0' * 64]
        self.fail_code('INVALID_OBSERVATIONS', lambda: v.summarize_evidence(
            value, {}, cases, final, context, observations))
        entries[0]['case_sha256'] = [v.digest(result)]
        self.assertEqual(v.summarize_evidence(value, {}, cases, final, context,
                                               observations)['live_acceptance'], 'PENDING')
        entries[-1]['id'] = entries[0]['id']
        self.fail_code('INVALID_OBSERVATIONS', lambda: v.summarize_evidence(
            value, {}, cases, final, context, observations))

    def test_ui_probe_must_be_actual_alias_and_described_neighbor(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            (target / 'catalog.json').write_bytes(v.canonical({'anime': {
                '8': {'title': 'New', 'aliases': ['Alias'], 'synopsis': 'Seed'},
                '9': {'title': 'Neighbor', 'aliases': [], 'synopsis': 'Description'}}}))
            value = plan()
            value['bundles']['target']['path'] = str(target)
            bundle = {'neighbors': {'8': [{'mal_id': 9}]}}
            v._check_probes(value, bundle)
            value['ui_probes']['alias_search']['query'] = 'New'
            self.fail_code('INVALID_UI_PROBES', lambda: v._check_probes(value, bundle))
            value['ui_probes']['alias_search']['query'] = 'Alias'
            value['ui_probes']['description_edit']['neighbor_mal_id'] = 10
            self.fail_code('INVALID_UI_PROBES', lambda: v._check_probes(value, bundle))

    def test_applicable_empty_case_requires_real_request_and_no_delivery(self):
        value = plan()
        value['bundles']['target']['expectations']['empty_seed'] = {
            'mal_id': 12, 'query': 'Empty'}
        value['bundles']['target']['histogram']['0'] = 1
        context = {'schema_version': 2, 'kind': 'arb-live-context',
                   'plan_sha256': v.digest(value),
                   'actor': {'tg_id': 42, 'chat_id': 42}}
        live_bot = bot(B, 'target', 'target-bot', 2.5)
        before = shot(4, 'target', bot_item=live_bot)
        after = shot(5, 'target', bot_item=live_bot)
        after['state']['database']['requests'].append({
            'id': 2, 'tg_id': 42, 'action_key': 'msg:42:2', 'raw_query': 'Empty',
            'seed_mal_id': 12, 'bundle_id': B, 'created_at': time(5),
            'resolved_at': time(5)})
        with mock.patch.object(v, '_bundle', return_value=(
                {'id': B, 'neighbors': {'12': []}}, None, None)):
            receipt = v.verify_case(value, context, 'empty', before, after)
            self.assertEqual(receipt['status'], 'PASS')
            self.assertEqual(receipt['counts']['delivered'], 0)
            after['state']['database']['delivered'].append({
                'id': 2, 'request_id': 2, 'tg_id': 42, 'rank': 1,
                'mal_id': 99, 'chat_id': 42, 'message_id': 12,
                'delivered_at': time(5)})
            self.fail_code('DELIVERY_ORDER', lambda: v.verify_case(
                value, context, 'empty', before, after))


if __name__ == '__main__': unittest.main()
