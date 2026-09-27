"""Exercise acquisition with small ZIP and HTTP fixtures."""

import hashlib
import io
import json
import stat
import tempfile
import threading
import unittest
import warnings
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from recsys.obtain import obtain
from recsys.sources import SourceError


def descriptor(source_id, filename, content, **origin):
    return {
        "id": source_id, "filename": filename, "size": len(content),
        "sha256": hashlib.sha256(content).hexdigest(), **origin,
    }


def make_zip(items):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, content in items:
                archive.writestr(name, content)
    return output.getvalue()


class LocalServer:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                outer.requests.append(self.path)
                response = outer.responses.get(self.path)
                if response is None:
                    self.send_error(404)
                    return
                status, body = response
                self.send_response(status)
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *_args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def url(self, path):
        return f"http://127.0.0.1:{self.server.server_port}/{path}"


class ObtainTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "raw"
        self.csv = b"MAL_ID,Name\n1,Example\n"
        self.txt = b"anime 0.1 0.2 0.3\n"
        self.zip_bytes = make_zip([("glove.6B.300d.txt", self.txt), ("../escape", b"bad")])

    def registry(self, server, *, csv=None, zip_bytes=None, txt=None):
        csv = self.csv if csv is None else csv
        zip_bytes = self.zip_bytes if zip_bytes is None else zip_bytes
        txt = self.txt if txt is None else txt
        return {
            "schema_version": 1,
            "provenance": {"fixture": "local HTTP"},
            "sources": [
                descriptor("mal", "anime.csv", csv, url=server.url("anime.csv")),
                descriptor("zip", "glove.6B.zip", zip_bytes, url=server.url("glove.6B.zip")),
                descriptor("text", "glove.6B.300d.txt", txt,
                           archive_id="zip", member="glove.6B.300d.txt"),
            ],
        }

    def assert_no_temps(self):
        self.assertEqual(list(self.data.glob(".*.tmp")), [])

    def test_download_extract_manifest_and_repeat_cache(self):
        with LocalServer({"/anime.csv": (200, self.csv), "/glove.6B.zip": (200, self.zip_bytes)}) as server:
            registry = self.registry(server)
            manifest = obtain(self.data, registry=registry)
            first = manifest.read_bytes()
            payload = json.loads(first)
            self.assertEqual(payload["schema_version"], 1)
            self.assertEqual(len(payload["verified"]), 3)
            self.assertEqual((self.data / "glove.6B.300d.txt").read_bytes(), self.txt)
            self.assertFalse((self.data.parent / "escape").exists())
            self.assertEqual(server.requests, ["/anime.csv", "/glove.6B.zip"])
            obtain(self.data, registry=registry)
            self.assertEqual(manifest.read_bytes(), first)
            self.assertEqual(len(server.requests), 2)
            self.assert_no_temps()

    def test_offline_uses_archive_and_missing_fails_without_network(self):
        with LocalServer({"/anime.csv": (200, self.csv), "/glove.6B.zip": (200, self.zip_bytes)}) as server:
            registry = self.registry(server)
            obtain(self.data, registry=registry)
            (self.data / "glove.6B.300d.txt").unlink()
            obtain(self.data, registry=registry, offline=True)
            self.assertEqual(len(server.requests), 2)
            (self.data / "anime.csv").unlink()
            with self.assertRaisesRegex(SourceError, "missing source"):
                obtain(self.data, registry=registry, offline=True)
            self.assertFalse((self.data / "source-manifest.json").exists())
            self.assertEqual(len(server.requests), 2)

    def test_existing_corruption_fails_and_invalidates_manifest(self):
        with LocalServer({"/anime.csv": (200, self.csv), "/glove.6B.zip": (200, self.zip_bytes)}) as server:
            registry = self.registry(server)
            obtain(self.data, registry=registry)
            (self.data / "anime.csv").write_bytes(b"X" + self.csv[1:])
            with self.assertRaisesRegex(SourceError, "actual size.*SHA256"):
                obtain(self.data, registry=registry)
            self.assertFalse((self.data / "source-manifest.json").exists())
            self.assertEqual(len(server.requests), 2)

    def test_bad_downloads_leave_no_final_or_temp(self):
        for status, body, expected in (
            (200, b"X" + self.csv[1:], "invalid source"),
            (200, self.csv[:-1], "invalid source"),
            (200, self.csv + b"x", "oversized source"),
            (500, b"error", "download failed"),
        ):
            with self.subTest(status=status, body=body):
                with LocalServer({"/anime.csv": (status, body)}) as server:
                    registry = self.registry(server)
                    with self.assertRaisesRegex(SourceError, expected):
                        obtain(self.data, registry=registry)
                    self.assertFalse((self.data / "anime.csv").exists())
                    self.assertFalse((self.data / "source-manifest.json").exists())
                    self.assert_no_temps()

    def test_missing_duplicate_and_wrong_archive_members(self):
        cases = (
            (make_zip([("other.txt", self.txt)]), "exactly one"),
            (make_zip([("glove.6B.300d.txt", self.txt)] * 2), "exactly one"),
            (make_zip([("glove.6B.300d.txt", b"X" + self.txt[1:])]), "invalid source"),
        )
        for zip_bytes, expected in cases:
            with self.subTest(expected=expected):
                with LocalServer({"/anime.csv": (200, self.csv), "/glove.6B.zip": (200, zip_bytes)}) as server:
                    registry = self.registry(server, zip_bytes=zip_bytes)
                    with self.assertRaisesRegex(SourceError, expected):
                        obtain(self.data, registry=registry)
                    self.assertFalse((self.data / "glove.6B.300d.txt").exists())
                    self.assertFalse((self.data / "source-manifest.json").exists())
                    self.assert_no_temps()
                for path in self.data.iterdir():
                    path.unlink()

    def test_rejects_directory_mode_without_trailing_slash(self):
        member = zipfile.ZipInfo("glove.6B.300d.txt")
        member.create_system = 3
        member.external_attr = (stat.S_IFDIR | 0o755) << 16
        zip_bytes = make_zip([(member, self.txt)])
        with LocalServer({"/anime.csv": (200, self.csv), "/glove.6B.zip": (200, zip_bytes)}) as server:
            registry = self.registry(server, zip_bytes=zip_bytes)
            with self.assertRaisesRegex(SourceError, "unsafe or incorrect member"):
                obtain(self.data, registry=registry)
        self.assertFalse((self.data / "glove.6B.300d.txt").exists())
        self.assertFalse((self.data / "source-manifest.json").exists())
        self.assert_no_temps()

    def test_rejects_symlink_target(self):
        self.data.mkdir()
        outside = Path(self.temp.name) / "outside"
        outside.write_bytes(self.csv)
        (self.data / "anime.csv").symlink_to(outside)
        with LocalServer({}) as server:
            with self.assertRaisesRegex(SourceError, "unsafe source target"):
                obtain(self.data, registry=self.registry(server), offline=True)
        self.assertEqual(outside.read_bytes(), self.csv)


if __name__ == "__main__":
    unittest.main()
