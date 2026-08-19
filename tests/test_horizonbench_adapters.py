"""Regression tests for the HorizonBench public/held-out JSONL adapter."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.adapters import (  # noqa: E402
    BenchmarkTask,
    BenchmarkTaskSet,
    JsonlTaskAdapter,
)


def _write_bytes(directory: str, name: str, payload: bytes) -> Path:
    path = Path(directory) / name
    path.write_bytes(payload)
    return path


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")


class HorizonBenchAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = JsonlTaskAdapter("horizonbench-public", "fixture-v1")

    def test_load_heldout_selects_split_and_preserves_metadata(self) -> None:
        records = [
            {
                "id": "pub-1",
                "category": "goal_retention",
                "goal": "Keep the public goal",
                "constraint": "Do not leak the held-out set",
                "split": "public",
                "difficulty": "easy",
            },
            {
                "id": "hold-1",
                "category": "constraint_retention",
                "goal": "Retain the original constraint",
                "constraint": "Never export private fields",
                "split": "heldout",
                "tags": ["held", "strict"],
                "nested": {"origin": "suite-a"},
                "count": 3,
                "enabled": True,
                "note": None,
            },
            {
                "id": "hold-2",
                "category": "fault_recovery",
                "goal": "Resume from the last checkpoint",
                "constraint": "Reuse the stable operation ID",
                "split": "heldout",
                "metadata": {"suite": "recovery"},
                "weight": 1.5,
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_bytes(temporary, "tasks.jsonl", _jsonl(records))
            loaded = self.adapter.load_heldout(path)

        self.assertIsInstance(loaded, BenchmarkTaskSet)
        self.assertEqual(loaded.source_name, "horizonbench-public")
        self.assertEqual(loaded.source_revision, "fixture-v1")
        self.assertEqual(loaded.split, "heldout")
        self.assertEqual(loaded.task_ids, ("hold-1", "hold-2"))
        self.assertEqual(len(loaded.tasks), 2)
        self.assertTrue(all(isinstance(task, BenchmarkTask) for task in loaded.tasks))
        self.assertEqual(
            loaded.tasks[0].as_dict()["metadata"],
            {
                "tags": ["held", "strict"],
                "nested": {"origin": "suite-a"},
                "count": 3,
                "enabled": True,
                "note": None,
            },
        )
        self.assertEqual(loaded.tasks[1].as_dict()["metadata"], {"suite": "recovery", "weight": 1.5})
        identity = loaded.identity()
        self.assertEqual(identity["task_ids"], ["hold-1", "hold-2"])
        self.assertEqual(identity["source_sha256"], loaded.source_sha256)
        self.assertEqual(identity["task_set_sha256"], loaded.task_set_sha256)
        self.assertNotIn("tasks", identity)
        self.assertEqual(loaded.as_dict()["tasks"][0]["task_id"], "hold-1")

    def test_task_set_hash_is_independent_of_formatting_and_order(self) -> None:
        compact = (
            b'{"split":"public","constraint":"C-public","goal":"G-public","category":"goal_retention","id":"pub-1"}\n'
            b'{"id":"hold-2","category":"fault_recovery","goal":"G2","constraint":"C2","split":"heldout","score":2}\n'
            b'{"id":"hold-1","category":"goal_retention","goal":"G1","constraint":"C1","split":"heldout","score":1}\n'
        )
        reordered = (
            b'{  "score": 1,  "split": "heldout",  "constraint": "C1",  "goal": "G1",  "category": "goal_retention",  "id": "hold-1" }\n'
            b'{"id":"pub-1","category":"goal_retention","goal":"G-public","constraint":"C-public","split":"public"}\n'
            b'{"split": "heldout", "score": 2, "constraint": "C2", "goal": "G2", "category": "fault_recovery", "id": "hold-2"}\n'
        )
        with tempfile.TemporaryDirectory() as temporary:
            compact_path = _write_bytes(temporary, "compact.jsonl", compact)
            pretty_path = _write_bytes(temporary, "pretty.jsonl", reordered)
            first = self.adapter.load(compact_path, split="heldout")
            second = self.adapter.load(pretty_path, split="heldout")

        self.assertNotEqual(compact, reordered)
        self.assertNotEqual(first.source_sha256, second.source_sha256)
        self.assertEqual(first.source_sha256, hashlib.sha256(compact).hexdigest())
        self.assertEqual(second.source_sha256, hashlib.sha256(reordered).hexdigest())
        self.assertEqual(first.task_set_sha256, second.task_set_sha256)
        self.assertEqual(set(first.task_ids), {"hold-1", "hold-2"})
        self.assertEqual(set(second.task_ids), {"hold-1", "hold-2"})

    def test_duplicate_ids_across_splits_are_rejected(self) -> None:
        records = [
            {
                "id": "shared-1",
                "category": "goal_retention",
                "goal": "Public copy",
                "constraint": "Keep the public wording",
                "split": "public",
            },
            {
                "id": "shared-1",
                "category": "goal_retention",
                "goal": "Held-out copy",
                "constraint": "Keep the held-out wording",
                "split": "heldout",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_bytes(temporary, "duplicates.jsonl", _jsonl(records))
            with self.assertRaises(ValueError) as ctx:
                self.adapter.load(path, split="heldout")
        message = str(ctx.exception)
        self.assertIn("duplicate", message)
        self.assertIn("shared-1", message)
        self.assertIn("duplicates.jsonl", message)

    def test_invalid_and_missing_required_data_are_rejected(self) -> None:
        valid = {
            "id": "ok-1",
            "category": "goal_retention",
            "goal": "A valid goal",
            "constraint": "A valid constraint",
            "split": "heldout",
        }
        cases = {
            "missing_split": {key: value for key, value in valid.items() if key != "split"},
            "blank_id": dict(valid, id="   "),
            "non_string_goal": dict(valid, goal=12),
            "invalid_json": "not-json",
            "blank_line": "",
            "non_object": ["not", "an", "object"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            for name, payload in cases.items():
                if isinstance(payload, str):
                    raw = (payload + "\n").encode("utf-8")
                elif isinstance(payload, list):
                    raw = (json.dumps(payload) + "\n").encode("utf-8")
                else:
                    raw = _jsonl([payload])
                path = _write_bytes(temporary, name + ".jsonl", raw)
                with self.subTest(name=name):
                    with self.assertRaises(ValueError) as ctx:
                        self.adapter.load(path, split="heldout")
                    self.assertIn(name + ".jsonl", str(ctx.exception))

            missing = Path(temporary) / "missing.jsonl"
            with self.assertRaises(ValueError) as ctx:
                self.adapter.load(missing, split="heldout")
            self.assertIn("missing.jsonl", str(ctx.exception))

            directory = Path(temporary) / "not-a-file"
            directory.mkdir()
            with self.assertRaises(ValueError) as ctx:
                self.adapter.load(directory, split="heldout")
            self.assertIn("not-a-file", str(ctx.exception))

    def test_empty_requested_split_is_rejected(self) -> None:
        records = [
            {
                "id": "pub-1",
                "category": "goal_retention",
                "goal": "Only a public task",
                "constraint": "Do not invent a held-out row",
                "split": "public",
            }
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = _write_bytes(temporary, "public-only.jsonl", _jsonl(records))
            with self.assertRaises(ValueError) as ctx:
                self.adapter.load(path, split="heldout")
        message = str(ctx.exception)
        self.assertIn("public-only.jsonl", message)
        self.assertIn("heldout", message)

    def test_metadata_values_and_unsupported_shapes_are_validated(self) -> None:
        supported = {
            "id": "meta-1",
            "category": "long_context_degradation",
            "goal": "Keep nested evidence",
            "constraint": "Preserve scalar and structured metadata",
            "split": "heldout",
            "label": "keep",
            "score": 0.25,
            "retries": 2,
            "ok": False,
            "empty": None,
            "flags": ["a", {"b": 1}],
            "detail": {"level": 2, "names": ["x", "y"]},
        }
        with tempfile.TemporaryDirectory() as temporary:
            supported_path = _write_bytes(temporary, "supported.jsonl", _jsonl([supported]))
            loaded = self.adapter.load(supported_path, split="heldout")
            self.assertEqual(
                loaded.tasks[0].as_dict()["metadata"],
                {
                    "label": "keep",
                    "score": 0.25,
                    "retries": 2,
                    "ok": False,
                    "empty": None,
                    "flags": ["a", {"b": 1}],
                    "detail": {"level": 2, "names": ["x", "y"]},
                },
            )

            for name, raw in {
                "metadata_list": _jsonl([dict(supported, metadata=["not-an-object"])]),
                "metadata_string": _jsonl([dict(supported, metadata="nope")]),
                "nonfinite": (
                    json.dumps({key: value for key, value in supported.items() if key != "score"}, ensure_ascii=False)[:-1]
                    + ', "score": NaN}\n'
                ).encode("utf-8"),
            }.items():
                path = _write_bytes(temporary, name + ".jsonl", raw)
                with self.subTest(name=name):
                    with self.assertRaises(ValueError) as ctx:
                        self.adapter.load(path, split="heldout")
                    self.assertIn(name + ".jsonl", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
