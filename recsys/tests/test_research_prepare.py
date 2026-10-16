import tempfile
import unittest
from pathlib import Path

from research_support import ResearchFixture

from recsys.research.artifacts import identity, read_json, stream_jsonl, stream_tsv
from recsys.research.prepare import validate_preparation
from recsys.research.protocol import ResearchError, split_user


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fixture = ResearchFixture(self.tmp.name)

    def test_determinism_projection_and_heldout_fit_separation(self):
        first = self.fixture.prepare("a")
        second = self.fixture.prepare("b")
        for path in first.rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())
        manifest = validate_preparation(
            first, expected_identity=identity(first / "prepare-manifest.json")
        )
        files = manifest["files"]
        catalog, _ = read_json(first / "model-catalog.json")
        self.assertEqual(set(catalog["anime"]), set(map(str, self.fixture.items)))
        self.assertTrue(all(item["score"] is None for item in catalog["anime"].values()))
        train = list(
            stream_tsv(
                first / "fit/train-interactions.tsv.gz", files["fit/train-interactions.tsv.gz"]
            )
        )
        self.assertTrue(
            all(split_user(row.split(b"\t")[0].decode()) == "train" for row in train[1:])
        )
        self.assertTrue(all(b"\tcompleted\n" in row for row in train[1:]))
        validation = list(
            stream_jsonl(
                first / "evaluator/validation-labels.jsonl.gz",
                files["evaluator/validation-labels.jsonl.gz"],
            )
        )
        test = list(
            stream_jsonl(
                first / "evaluator/test-labels.jsonl.gz", files["evaluator/test-labels.jsonl.gz"]
            )
        )
        self.assertTrue(
            {row["user_key"] for row in validation}.isdisjoint({row["user_key"] for row in test})
        )
        self.assertTrue(
            all(
                set(row["target_ids"]).isdisjoint(row["reserved_seed_ids"])
                for row in validation + test
            )
        )
        self.assertTrue(
            all(
                row["target_ids"] == validation[0]["target_ids"]
                for row in validation
                if row["user_key"] == validation[0]["user_key"]
            )
        )
        queries = list(
            stream_jsonl(
                first / "development/queries.jsonl.gz", files["development/queries.jsonl.gz"]
            )
        )
        self.assertEqual(len(queries), len(validation))
        self.assertEqual(len(queries), 8 * 41)
        self.assertEqual(manifest["counts"]["sampled_users"], 15)

    def test_short_cohort_query_bounds(self):
        short_user = next(user for user in self.fixture.users if split_user(user) == "validation")
        # Build a second fixture in a different directory with one 3-positive user.
        another = Path(self.tmp.name) / "short"
        another.mkdir()
        fixture = ResearchFixture(another, short_user=short_user)
        prepared = fixture.prepare()
        manifest = validate_preparation(
            prepared, expected_identity=identity(prepared / "prepare-manifest.json")
        )
        labels = list(
            stream_jsonl(
                prepared / "evaluator/validation-labels.jsonl.gz",
                manifest["files"]["evaluator/validation-labels.jsonl.gz"],
            )
        )
        short = [row for row in labels if row["user_key"] == short_user]
        self.assertEqual(len(short), 9)  # N=0 and four orders at N=1,2.
        self.assertEqual(len(short[0]["target_ids"]), 1)
        self.assertEqual(max(row["n"] for row in short), 2)

    def test_heldout_rating_change_does_not_change_train_statistics(self):
        first = self.fixture.prepare("first")
        train_before = (first / "fit/train-stats.json").read_bytes()
        heldout = next(user for user in self.fixture.users if split_user(user) == "validation")
        source = self.fixture.dataset / "interactions.tsv.gz"
        raw = list(stream_tsv(source))
        source.unlink()
        from recsys.research.artifacts import StreamWriter

        writer = StreamWriter(source)
        changed = False
        for row in raw:
            if row.startswith(f"{heldout}\t1\t".encode()):
                row = (
                    f"{heldout}\t1\t8\tcompleted\n".encode()
                    if b"\t8\t" not in row
                    else f"{heldout}\t1\t9\tcompleted\n".encode()
                )
                changed = True
            writer.write(row)
        writer.close()
        self.assertTrue(changed)
        second = self.fixture.prepare("second")
        self.assertEqual(train_before, (second / "fit/train-stats.json").read_bytes())

    def test_tamper_and_unsafe_output_fail_closed(self):
        prepared = self.fixture.prepare()
        expected = identity(prepared / "prepare-manifest.json")
        (prepared / "fit/train-stats.json").write_bytes(b"{}\n")
        with self.assertRaises(ResearchError):
            validate_preparation(prepared, expected_identity=expected)
        with self.assertRaises(ResearchError):
            self.fixture.prepare("prepared")
        link = Path(self.tmp.name) / "unsafe"
        link.symlink_to(self.fixture.release, target_is_directory=True)
        with self.assertRaises(ResearchError):
            self.fixture.prepare("unsafe")
