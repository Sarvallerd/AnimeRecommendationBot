"""Capture import and guard behavior without Docker, settings, or real rows."""
import copy
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).with_name('live_evidence_v2.py')
spec = importlib.util.spec_from_file_location('arb034_test_capture', SCRIPT)
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CaptureTests(unittest.TestCase):
    def test_import_does_not_call_transport_or_modify_files(self):
        with mock.patch('subprocess.run', side_effect=AssertionError('transport called')):
            again = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(again)
        self.assertTrue(callable(again.capture))

    def test_guards_must_remain_stable_and_transport_is_restored(self):
        guard = {'daemon_id': 'synthetic', 'pollers': {'matching': 1}}
        state = {'schema_version': 1, 'kind': 'full'}
        old_compose, old_docker = c.e.compose, c.e.docker
        with mock.patch.object(c, '_guards', side_effect=[copy.deepcopy(guard),
                                                           copy.deepcopy(guard)]), \
             mock.patch.object(c.e, 'snapshot', return_value=state), \
             mock.patch.object(c.Path, 'is_file', return_value=True):
            result = c.capture('synthetic-project', Path('/tmp/synthetic.env'))
        self.assertTrue(result['guards']['stable'])
        self.assertIs(c.e.compose, old_compose)
        self.assertIs(c.e.docker, old_docker)
        changed = copy.deepcopy(guard)
        changed['daemon_id'] = 'different'
        with mock.patch.object(c, '_guards', side_effect=[guard, changed]), \
             mock.patch.object(c.e, 'snapshot', return_value=state), \
             mock.patch.object(c.Path, 'is_file', return_value=True):
            with self.assertRaises(c.Error) as caught:
                c.capture('synthetic-project', Path('/tmp/synthetic.env'))
        self.assertEqual(caught.exception.code, 'UNSTABLE_CAPTURE')
        self.assertIs(c.e.compose, old_compose)
        self.assertIs(c.e.docker, old_docker)

    def test_fixed_error_never_prints_private_exception(self):
        secret = 'private-token-canary'
        with mock.patch.object(c, 'make_plan', side_effect=ValueError(secret)), \
             mock.patch('sys.stderr', new_callable=io.StringIO) as stderr:
            code = c.main(['plan', '--project', 'project', '--settings', '/tmp/a',
                           '--offline-evidence', '/tmp/b', '--baseline', '/tmp/c',
                           '--target-bundle', '/tmp/d', '--ui-probes', '/tmp/e',
                           '--output', '/tmp/f'])
        self.assertEqual(code, 1)
        self.assertEqual(stderr.getvalue(), 'FAIL INVALID_EVIDENCE\n')
        self.assertNotIn(secret, stderr.getvalue())

    def test_bounded_compose_ignores_shell_selection_overrides(self):
        with mock.patch.dict(c.os.environ, {'ARB_BUNDLE_DIR': '/wrong',
                                             'COMPOSE_PROJECT_NAME': 'wrong'}, clear=False), \
             mock.patch.object(c, 'command', return_value='ok') as command:
            self.assertEqual(c._compose('good-project', Path('/tmp/fixture.env'), 'ps'), 'ok')
        self.assertNotIn('ARB_BUNDLE_DIR', command.call_args.kwargs['env'])
        self.assertNotIn('COMPOSE_PROJECT_NAME', command.call_args.kwargs['env'])
        self.assertEqual(command.call_args.args[0][0:4],
                         ['docker', 'compose', '--env-file', '/tmp/fixture.env'])

    def test_failed_operation_receipt_is_recomputed_from_private_inputs(self):
        from deploy.tests.test_acceptance_live_v2 import plan, sequence
        values = dict(zip(('before', 'stopped', 'prepared', 'started'), sequence()))
        values['prepared']['state']['runtime']['prepare']['exit_code'] = 1
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for value in values.values(): c._cache(parent, value)
            receipt = c._failed_operation(plan(), 'initial-update', values, 'PREPARE_GATE')
            self.assertEqual(c._verify_saved_operation(plan(), receipt, parent), values)
            forged = dict(receipt, reason='OK')
            with self.assertRaises(c.Error) as caught:
                c._verify_saved_operation(plan(), forged, parent)
            self.assertEqual(caught.exception.code, 'FORGED_OPERATION')

    def test_failed_case_receipt_does_not_turn_into_pass(self):
        from deploy.tests.test_acceptance_live_v2 import plan, sequence
        before, _, _, after = sequence()
        inputs = {'before': before, 'after': after}
        context = {'schema_version': 1, 'kind': 'old-context'}
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            for value in inputs.values(): c._cache(parent, value)
            receipt = c._failed_case(plan(), context, 'description-edit', inputs,
                                     None, 'INVALID_CONTEXT')
            self.assertEqual(c._verify_saved_case(plan(), context, receipt, parent, {}),
                             inputs)
            forged = dict(receipt, reason='OK')
            with self.assertRaises(c.Error) as caught:
                c._verify_saved_case(plan(), context, forged, parent, {})
            self.assertEqual(caught.exception.code, 'FORGED_CASE')


if __name__ == '__main__': unittest.main()
