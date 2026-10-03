"""Adversarial, secret-free fixtures for the live evidence verifier."""

import copy
import datetime as dt
import importlib.util
import json
import os
import pathlib
import stat
import tempfile
import unittest
from unittest import mock

SCRIPT = pathlib.Path(__file__).with_name('acceptance-evidence.py')
SPEC = importlib.util.spec_from_file_location('arb020_evidence', SCRIPT)
e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(e)
A = 'sha256:' + 'a' * 64
B = 'sha256:' + 'b' * 64
TIME = '2026-10-03T00:00:00+00:00'
TARGETS = [400, 709, 4088, 329, 1303]


def request(identifier, *, actor=42, key=None, query='Cowboy Bebop', seed=1, bundle=A):
    key = key or f'msg:{actor}:{identifier + 99}'
    return {'id': identifier, 'tg_id': actor, 'action_key': key, 'raw_query': query,
            'seed_mal_id': seed, 'bundle_id': bundle, 'resolved_at': TIME if seed else None,
            'created_at': TIME}


def delivery(identifier, request_id, rank, target, actor=42):
    return {'id': identifier, 'request_id': request_id, 'tg_id': actor, 'rank': rank,
            'mal_id': target, 'chat_id': actor, 'message_id': identifier + 100,
            'delivered_at': TIME}


def database():
    return {'migrations': [{'version': 1, 'applied_at': TIME}],
            'users': [{'tg_id': 42, 'language_code': 'ru', 'first_name': 'Тест', 'last_name': None,
                       'username': None, 'created_at': TIME, 'updated_at': TIME}],
            'requests': [request(1)],
            'delivered': [delivery(rank, 1, rank, target) for rank, target in enumerate(TARGETS, 1)],
            'anime_events': [], 'anime_current': [], 'recommendation_ratings': [],
            'feedback': [], 'legacy_users': [], 'legacy_requests': [], 'legacy_feedback': []}


def runtime(identity=A, *, started=TIME, running=True):
    return {'id': 'container-1', 'image_id': 'image-1', 'project': 'arb020-test', 'service': 'bot',
            'user': '10001:10001', 'status': 'running' if running else 'exited', 'running': running,
            'exit_code': 0, 'started_at': started, 'restart_count': 0, 'read_only_root': True,
            'artifact_mount': {'source': '/tmp/arb020-A' if identity == A else '/tmp/arb020-B',
                               'rw': False}, 'database_volume': None, 'bundle_id': identity,
            'bundle_count': 17562}


def shot(db=None, minute=0, identity=A):
    stamp = (dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=minute)).isoformat()
    return {'schema_version': 1, 'kind': 'full', 'snapshot_sql_sha256': e.sha(e.SQL.read_bytes()),
            'source_commit': 'f' * 40,
            'captured_at': stamp,
            'deployment': {'project': 'arb020-test', 'compose_sha256': 'c' * 64,
                           'settings_sha256': 'd' * 64},
            'runtime': {'bot': runtime(identity), 'prepare': None,
                        'postgres': {'id': 'pg-1', 'database_volume': 'arb020-test_postgres_data',
                                     'running': True}}, 'database': copy.deepcopy(db or database())}


def context():
    return {'schema_version': 1, 'project': 'arb020-test',
            'database_volume': 'arb020-test_postgres_data',
            'actor': {'tg_id': 42, 'chat_id': 42, 'anchor_request_id': 1},
            'bundles': {'A': {'identity': A, 'path': '/tmp/arb020-A'},
                        'B': {'identity': B, 'path': '/tmp/arb020-B'}}, 'anchor': {}}


def bundle(identity=A):
    return {'id': identity, 'count': 17562,
            'neighbors': {'1': [{'mal_id': target} for target in TARGETS], '40089': []}}


def check(case, before, after, *, identity=A, during=None):
    return e.verify_transition(case, context(), before, after, bundle(identity), during)


class EvidenceTests(unittest.TestCase):
    def assert_rejected(self, code, callback):
        with self.assertRaises(e.EvidenceError) as caught:
            callback()
        self.assertEqual(caught.exception.code, code)

    def recommendation_pair(self):
        before = shot(minute=0)
        after = shot(minute=1)
        after['database']['requests'].append(request(2))
        after['database']['delivered'].extend(delivery(5 + rank, 2, rank, target)
                                               for rank, target in enumerate(TARGETS, 1))
        return before, after

    def test_actor_bind_and_recommendation(self):
        before, after = self.recommendation_pair()
        self.assertEqual(check('recommendations', before, after)['counts']['delivered'], 5)
        report = {'bundles': context()['bundles'], 'project': 'arb020-test',
                  'source_commit': 'f' * 40, 'compose_sha256': 'c' * 64,
                  'settings_sha256': 'd' * 64, 'database_volume': 'arb020-test_postgres_data'}
        empty = shot({**database(), 'requests': [], 'delivered': []}, minute=0)
        bound = e.context_from_anchor(empty, shot(minute=1), report, 'a' * 64, 'b' * 64, 'c' * 64)
        self.assertEqual(bound['actor']['anchor_request_id'], 1)
        after['database']['requests'][-1]['tg_id'] = 43
        self.assert_rejected('INVALID_REQUEST', lambda: check('recommendations', before, after))

    def test_wrong_order_rank_seed_bundle_and_coordinates(self):
        before, after = self.recommendation_pair()
        self.assert_rejected('WRONG_BUNDLE', lambda: check('recommendations', before, after, identity=B))
        altered = copy.deepcopy(after)
        altered['database']['delivered'][-1]['mal_id'] = 999
        self.assert_rejected('DELIVERY_ORDER', lambda: check('recommendations', before, altered))
        altered = copy.deepcopy(after)
        altered['database']['delivered'][-1]['rank'] = 1
        self.assert_rejected('DUPLICATE_RANK', lambda: check('recommendations', before, altered))
        altered = copy.deepcopy(after)
        altered['database']['requests'][-1]['seed_mal_id'] = 40089
        self.assert_rejected('DELIVERY_ORDER', lambda: check('recommendations', before, altered))
        altered = copy.deepcopy(after)
        altered['database']['delivered'][-1]['chat_id'] = 77
        self.assert_rejected('DELIVERY_ORDER', lambda: check('recommendations', before, altered))

    def test_recommendation_scores_and_duplicate_effects(self):
        before = shot(minute=0)
        after = shot(minute=1)
        after['database']['recommendation_ratings'] = [
            {'position_id': 1, 'tg_id': 42, 'score': 0, 'created_at': TIME},
            {'position_id': 2, 'tg_id': 42, 'score': 5, 'created_at': TIME}]
        self.assertEqual(check('recommendation-scores', before, after)['counts']['recommendation_ratings'], 2)
        duplicate = copy.deepcopy(after)
        duplicate['database']['recommendation_ratings'].append(copy.deepcopy(duplicate['database']['recommendation_ratings'][0]))
        self.assert_rejected('DUPLICATE_KEY', lambda: check('recommendation-scores', before, duplicate))
        after['database']['recommendation_ratings'].pop()
        self.assert_rejected('RECOMMENDATION_SCORE_MISSING', lambda: check('recommendation-scores', before, after))

    def test_score_and_pointer_update_preserve_old_history(self):
        before = shot(minute=0)
        after = shot(minute=1)
        after['database']['requests'].append(request(2))
        after['database']['anime_events'].append({'id': 1, 'request_id': 2, 'tg_id': 42,
                                                  'mal_id': 1, 'score': 1, 'created_at': TIME})
        after['database']['anime_current'].append({'tg_id': 42, 'mal_id': 1, 'current_event_id': 1,
                                                   'created_at': TIME, 'updated_at': TIME})
        self.assertEqual(check('anime-score-1', before, after)['counts']['anime_events'], 1)
        later = copy.deepcopy(after)
        later['captured_at'] = (dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=2)).isoformat()
        later['database']['requests'].append(request(3))
        later['database']['anime_events'].append({'id': 2, 'request_id': 3, 'tg_id': 42,
                                                  'mal_id': 1, 'score': 10, 'created_at': TIME})
        later['database']['anime_current'][0]['current_event_id'] = 2
        later['database']['anime_current'][0]['updated_at'] = later['captured_at']
        self.assertEqual(check('anime-score-10', after, later)['counts']['anime_events'], 1)
        later['database']['anime_current'][0]['current_event_id'] = 1
        self.assert_rejected('CURRENT_POINTER', lambda: check('anime-score-10', after, later))

    def test_canary_exactness_and_old_rows(self):
        before = shot(minute=0)
        after = shot(minute=1)
        self.assertGreater(len(e.CANARIES['unicode']), 255)
        after['database']['feedback'].append({'id': 1, 'tg_id': 42, 'action_key': 'msg:42:200',
                                             'body': e.CANARIES['unicode'], 'created_at': TIME})
        self.assertEqual(check('feedback-unicode', before, after)['counts']['feedback'], 1)
        for body in (e.CANARIES['unicode'][:-1], e.CANARIES['unicode'].replace('e\u0301', 'é'),
                     e.CANARIES['unicode'] + ' '):
            mutated = copy.deepcopy(after)
            mutated['database']['feedback'][0]['body'] = body
            self.assert_rejected('FEEDBACK_BODY', lambda: check('feedback-unicode', before, mutated))
        prior = copy.deepcopy(before)
        prior['database']['feedback'].append({'id': 7, 'tg_id': 42, 'action_key': 'msg:42:7',
                                             'body': 'старый', 'created_at': TIME})
        new = copy.deepcopy(after)
        new['database']['feedback'].append({'id': 7, 'tg_id': 42, 'action_key': 'msg:42:7',
                                           'body': 'изменён', 'created_at': TIME})
        self.assert_rejected('OLD_HISTORY_CHANGED', lambda: check('feedback-unicode', prior, new))
        prior = shot(minute=0)
        prior['database']['legacy_feedback'] = [{'tg_id': 42, 'msg': 'legacy'}]
        self.assert_rejected('LEGACY_CHANGED', lambda: check('feedback-unicode', prior, after))
        prior = shot(minute=0)
        prior['database']['migrations'][0]['applied_at'] = '2026-10-02T00:00:00+00:00'
        self.assert_rejected('OLD_HISTORY_CHANGED', lambda: check('feedback-unicode', prior, after))

    def test_pending_resolution_only_expected_and_user_profile_update(self):
        before = shot(minute=0)
        pending = request(2, seed=None, bundle=None)
        before['database']['requests'].append(pending)
        after = copy.deepcopy(before)
        after['captured_at'] = (dt.datetime.fromisoformat(TIME) + dt.timedelta(minutes=1)).isoformat()
        after['database']['requests'][-1].update(seed_mal_id=40089, bundle_id=A, resolved_at=TIME,
                                                 raw_query='Pittanko!! Nekozakana')
        self.assert_rejected('OLD_REQUEST_CHANGED', lambda: check('empty', before, after))
        before['database']['requests'][-1]['raw_query'] = 'Pittanko!! Nekozakana'
        after['database']['users'][0]['first_name'] = 'Новое имя'
        after['database']['users'][0]['updated_at'] = after['captured_at']
        self.assertEqual(check('empty', before, after)['counts']['resolved_pending'], 1)
        after['database']['requests'][-1]['seed_mal_id'] = 1
        self.assert_rejected('WRONG_SELECTION', lambda: check('empty', before, after))

    def test_restart_outage_and_runtime_only_rejection(self):
        before = shot(minute=0)
        after = shot(minute=2)
        self.assert_rejected('RESTART_NOT_OBSERVED', lambda: check('restart', before, after))
        after['runtime']['bot']['started_at'] = '2026-10-03T00:01:00+00:00'
        self.assertEqual(check('restart', before, after)['status'], 'PASS')
        during = shot(minute=1)
        during['kind'] = 'runtime_only'
        during['database'] = None
        during['runtime']['postgres']['running'] = False
        self.assert_rejected('BOT_PROCESS_CHANGED', lambda: check('db-outage', before, after, during=during))
        after['runtime']['bot']['started_at'] = TIME
        self.assertEqual(check('db-outage', before, after, during=during)['status'], 'PASS')
        after['kind'] = 'runtime_only'
        self.assert_rejected('FULL_SNAPSHOT_REQUIRED', lambda: check('db-outage', before, after, during=during))

    def test_update_rollback_need_new_bundle_requests(self):
        before = shot(minute=0)
        after = shot(minute=1, identity=B)
        after['database']['requests'].append(request(2, bundle=B))
        after['database']['delivered'].extend(delivery(5 + rank, 2, rank, target)
                                               for rank, target in enumerate(TARGETS, 1))
        self.assertEqual(check('update', before, after, identity=B)['status'], 'PASS')
        old_only = shot(minute=1, identity=B)
        self.assert_rejected('EXPECTED_REQUEST_MISSING', lambda: check('update', before, old_only, identity=B))
        rollback = shot(copy.deepcopy(after['database']), minute=2)
        rollback['database']['requests'].append(request(3, bundle=A))
        rollback['database']['delivered'].extend(delivery(10 + rank, 3, rank, target)
                                                  for rank, target in enumerate(TARGETS, 1))
        self.assertEqual(check('rollback', after, rollback)['status'], 'PASS')

    def test_missing_events_ranges_and_early_outage_save(self):
        before = shot(minute=0)
        after = shot(minute=1)
        after['database']['requests'].append(request(2))
        self.assert_rejected('ANIME_SCORE_MISSING', lambda: check('anime-score-1', before, after))
        after['database']['anime_events'].append({'id': 1, 'request_id': 2, 'tg_id': 42,
                                                  'mal_id': 1, 'score': 11, 'created_at': TIME})
        after['database']['anime_current'].append({'tg_id': 42, 'mal_id': 1,
                                                   'current_event_id': 1, 'created_at': TIME,
                                                   'updated_at': TIME})
        self.assert_rejected('INVALID_ANIME_EVENT', lambda: check('anime-score-1', before, after))
        after = shot(minute=1)
        after['database']['recommendation_ratings'].append(
            {'position_id': 1, 'tg_id': 42, 'score': 6, 'created_at': TIME})
        self.assert_rejected('INVALID_RECOMMENDATION_SCORE',
                             lambda: check('recommendation-scores', before, after))
        after = shot(minute=2)
        after['database']['feedback'].append({'id': 1, 'tg_id': 42, 'action_key': 'msg:42:200',
                                             'body': e.CANARIES['retry'], 'created_at': TIME})
        during = shot(minute=1)
        during['kind'] = 'runtime_only'; during['database'] = None
        during['runtime']['postgres']['running'] = False
        self.assert_rejected('UNEXPECTED_EFFECTS', lambda: check('db-outage', before, after, during=during))

    def test_unrelated_resolution_and_summary_duplicate(self):
        before = shot(minute=0)
        after = shot(minute=1)
        other = request(2, key='msg:42:301', seed=None, bundle=None)
        before['database']['requests'].append(other)
        after['database']['requests'].append({**other, 'seed_mal_id': 1,
                                              'bundle_id': A, 'resolved_at': TIME})
        self.assert_rejected('UNEXPECTED_REQUEST', lambda: check('recommendation-repeat', before, after))
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            results = root/'results'; results.mkdir(mode=0o700)
            inputs = results/'inputs'; inputs.mkdir(mode=0o700)
            body = e.canonical({'schema_version': 1})
            digest = e.sha(body)
            (inputs/f'{digest}.json').write_bytes(body)
            item = {'schema_version': 1, 'case': 'recommendations', 'phase': None,
                    'status': 'FAIL', 'reason': 'INVALID_EVIDENCE', 'counts': {}, 'bundle_id': None,
                    'context_sha256': e.sha(e.canonical(context())),
                    'input_sha256': {'before': digest, 'after': digest}}
            (results/'one.json').write_bytes(e.canonical(item))
            (results/'two.json').write_bytes(e.canonical(item))
            observations = {'schema_version': 1, 'context_sha256': item['context_sha256'],
                            'scenarios': [{'id': name, 'status': 'PENDING', 'observed_at': None,
                                           'observation': ''} for name in e.UI_IDS]}
            self.assert_rejected('DUPLICATE_RESULT',
                                 lambda: e.summarize(context(), results, observations))

    def test_embedded_harness_python_compiles(self):
        lines = SCRIPT.with_name('full-catalog-acceptance.sh').read_text().splitlines()
        starts = [index for index, line in enumerate(lines) if "<<'PY'" in line]
        self.assertEqual(len(starts), 4)
        for start in starts:
            end = next(i for i in range(start + 1, len(lines)) if lines[i] == 'PY')
            compile('\n'.join(lines[start + 1:end]), f'harness-{start}', 'exec')

    def test_private_output_and_ui_pending(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            private = root/'private'; private.mkdir(mode=0o700)
            target = private/'one.json'
            e.write_new(target, {'value': 1})
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o600)
            self.assert_rejected('OUTPUT_EXISTS', lambda: e.write_new(target, {'value': 2}))
            link = private/'link.json'; link.symlink_to(target)
            self.assert_rejected('OUTPUT_EXISTS', lambda: e.write_new(link, {'value': 3}))
            public = root/'public'; public.mkdir(mode=0o755)
            self.assert_rejected('PRIVATE_DIRECTORY_REQUIRED', lambda: e.write_new(public/'x.json', {}))
            results = private/'results'; results.mkdir(mode=0o700)
            observations = {'schema_version': 1, 'context_sha256': e.sha(e.canonical(context())),
                            'scenarios': [{'id': name, 'status': 'PENDING', 'observed_at': None,
                                           'observation': ''} for name in e.UI_IDS]}
            self.assertEqual(e.summarize(context(), results, observations)['live_acceptance'], 'PENDING')
            observations['scenarios'][0]['status'] = 'FAIL'
            self.assertEqual(e.summarize(context(), results, observations)['live_acceptance'], 'FAIL')


if __name__ == '__main__':
    unittest.main()
