import contextlib
import io
import unittest

from recsys.cli import build_parser, main


class ResearchCliTests(unittest.TestCase):
    def test_commands_exist_without_eager_research_import(self):
        parser = build_parser()
        for command in ("prepare", "validate", "evaluate", "compare", "finalize", "test-queries"):
            help_text = parser.format_help()
            self.assertIn("research", help_text)
            self.assertIn(command, parser._subparsers._group_actions[0].choices["research"].
                          _subparsers._group_actions[0].choices)

    def test_safe_error_code(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            status = main(["research", "validate", "/private/secret/input", "--expected-identity",
                           "sha256:" + "0" * 64])
        self.assertEqual(status, 1)
        self.assertIn("research_unsafe_path", err.getvalue())
        self.assertNotIn("/private/secret/input", err.getvalue())
