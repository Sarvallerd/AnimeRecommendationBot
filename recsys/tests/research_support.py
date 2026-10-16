"""Small deterministic ML-01 fixture; no generated user data enters Git artifacts."""

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from recsys.contracts.bundle import canonical_bytes
from recsys.research.artifacts import StreamWriter, identity, read_json, stream_jsonl, write_json
from recsys.research.prepare import prepare_research
from recsys.research.protocol import sha


class ResearchFixture:
    def __init__(self, root, *, short_user=None):
        self.root = Path(root)
        self.dataset = self.root / "dataset"
        self.release = self.root / "release"
        self.dataset.mkdir()
        self.release.mkdir()
        self.items = list(range(1, 22))
        self.catalog = {
            "schema_version": 1,
            "anime": {
                str(item): {
                    "title": f"Title {item}",
                    "aliases": [],
                    "genres": ["Drama" if item % 2 else "Action"],
                    "score": 8.1,
                    "year": 2025,
                    "type": "TV",
                    "episodes": 12,
                    "synopsis": "A story.",
                }
                for item in self.items
            },
        }
        self.catalog_raw = canonical_bytes(self.catalog)
        (self.release / "catalog.json").write_bytes(self.catalog_raw)
        self.users = sorted(
            "u1_" + hashlib.sha256(f"user-{i}".encode()).hexdigest() for i in range(80)
        )
        interactions = StreamWriter(self.dataset / "interactions.tsv.gz")
        interactions.write(b"dataset_user_key\tmal_id\trating\twatch_status\n")
        for user in self.users:
            for item in self.items[:3] if user == short_user else self.items[:17]:
                interactions.write(f"{user}\t{item}\t{8 + (item % 3)}\tcompleted\n".encode())
            interactions.write(f"{user}\t18\t7\twatching\n".encode())
        interactions.close()
        base = canonical_bytes({"fixture": "base"})
        (self.dataset / "dataset-manifest.json").write_bytes(base)
        (self.dataset / "item-facts.json").write_bytes(
            canonical_bytes(
                {
                    "schema_version": 1,
                    "anime": {
                        str(item): {
                            "first_release_date": "2025-01-01",
                            "airing_status": "finished_airing",
                            "seasonal_year": 2025,
                        }
                        for item in self.items
                    },
                }
            )
        )
        release_manifest = {
            "kind": "arb-catalog-release",
            "policy": "ordered-alias-overlay-v1",
            "inputs": {
                "base": {"identity": "sha256:" + sha(base)},
                "aliases": {"identity": "sha256:" + "b" * 64},
            },
            "counts": {"catalog_count": len(self.items)},
            "files": {
                "catalog.json": {"bytes": len(self.catalog_raw), "sha256": sha(self.catalog_raw)}
            },
        }
        release_raw = canonical_bytes(release_manifest)
        (self.release / "release-manifest.json").write_bytes(release_raw)
        self.lock = {
            "release_id": "neelagiri-2025-nafiul-2026-v1",
            "inputs": {
                "base": {"identity": "sha256:" + sha(base)},
                "aliases": {"identity": "sha256:" + "b" * 64},
                "catalog_release": {
                    "identity": "sha256:" + sha(release_raw),
                    "catalog_sha256": sha(self.catalog_raw),
                    "catalog_count": len(self.items),
                },
            },
        }

    def prepare(self, name="prepared"):
        output = self.root / name
        with (
            patch(
                "recsys.research.prepare._release", return_value=(self.lock, self.catalog, {}, b"")
            ),
            patch(
                "recsys.research.prepare.validate_dataset",
                return_value=SimpleNamespace(identity=self.lock["inputs"]["base"]["identity"]),
            ),
        ):
            prepare_research(self.dataset, self.release, output, code_revision="a" * 40)
        return output

    def predictions(
        self, prepared, name, *, split="validation", queries_path=None, model_id="fixture"
    ):
        output = self.root / name
        output.mkdir()
        prep, _ = read_json(prepared / "prepare-manifest.json")
        fit, _ = read_json(prepared / "fit/manifest.json")
        stats, _ = read_json(prepared / "fit/train-stats.json")
        if queries_path is None:
            queries_path = prepared / "development/queries.jsonl.gz"
            query_desc = prep["files"]["development/queries.jsonl.gz"]
        else:
            manifest, _ = read_json(queries_path.parent / "test-query-manifest.json")
            query_desc = manifest["queries"]
        score_writer = StreamWriter(output / "scores.jsonl.gz")
        for query in stream_jsonl(queries_path, query_desc):
            scores = [
                [
                    item,
                    stats["items"][str(item)]["positive_users"] if query["n"] == 0 else 100 - item,
                ]
                for item in self.items
            ]
            score_writer.write_json({"query_id": query["query_id"], "scores": scores})
        score_desc = score_writer.close()
        write_json(
            output / "prediction-manifest.json",
            {
                "schema_version": 1,
                "kind": "arb-ml-predictions",
                "prepared_identity": identity(prepared / "prepare-manifest.json"),
                "query_descriptor": query_desc,
                "catalog_descriptor": prep["files"]["model-catalog.json"],
                "split": split,
                "model_id": model_id,
                "config_sha256": "c" * 64,
                "fit_identity": identity(prepared / "fit/manifest.json"),
                "train_interactions": prep["files"]["fit/train-interactions.tsv.gz"],
                "train_stats": prep["files"]["fit/train-stats.json"],
                "code_revision": "a" * 40,
                "zero_seed_policy": "popularity-train-positive-user-counts",
                "scores": score_desc,
            },
        )
        return output
