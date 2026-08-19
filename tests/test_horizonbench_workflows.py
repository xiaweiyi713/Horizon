"""Regression tests for the HorizonBench cross-domain workflow loader."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType
from typing import Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.horizonbench.workflows import (  # noqa: E402
    DEFAULT_CROSS_DOMAIN_TASKS,
    CrossDomainWorkflowSuite,
    WorkflowDefinition,
    WorkflowTask,
    WorkflowValidationError,
    load_cross_domain_workflows,
)


EXPECTED_DOMAINS = (
    "data_analysis",
    "operations",
    "research_synthesis",
    "software_engineering",
)


def _write_bytes(directory: str, name: str, payload: bytes) -> Path:
    path = Path(directory) / name
    path.write_bytes(payload)
    return path


def _jsonl(records: list[dict[str, object]]) -> bytes:
    return "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records).encode("utf-8")


def _valid_task(
    task_id: str,
    workflow_id: str,
    domain: str,
    step_index: int,
    *,
    depends_on: Optional[list[str]] = None,
    recovery_boundary: bool = False,
    evidence_required: bool = True,
    expected_boundaries: int = 12,
    workflow_title: Optional[str] = None,
    **overrides: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "id": task_id,
        "category": domain,
        "goal": "Complete step {} of {} with the original long-horizon goal intact.".format(step_index, workflow_id),
        "constraint": "Keep the approved {} contract and do not invent a later dependency.".format(domain),
        "split": "heldout",
        "workflow_id": workflow_id,
        "workflow_title": workflow_title or "{} workflow".format(workflow_id),
        "domain": domain,
        "step_index": step_index,
        "expected_boundaries": expected_boundaries,
        "depends_on": list(depends_on or []),
        "recovery_boundary": recovery_boundary,
        "evidence_required": evidence_required,
    }
    record.update(overrides)
    return record


def _valid_workflow(
    workflow_id: str,
    domain: str,
    *,
    prefix: str | None = None,
    recovery_step: int = 3,
    expected_boundaries: int = 12,
) -> list[dict[str, object]]:
    stem = prefix or workflow_id
    first = "{}-01".format(stem)
    second = "{}-02".format(stem)
    third = "{}-03".format(stem)
    return [
        _valid_task(first, workflow_id, domain, 1, expected_boundaries=expected_boundaries),
        _valid_task(
            second,
            workflow_id,
            domain,
            2,
            depends_on=[first],
            expected_boundaries=expected_boundaries,
        ),
        _valid_task(
            third,
            workflow_id,
            domain,
            3,
            depends_on=[first, second],
            recovery_boundary=recovery_step == 3,
            expected_boundaries=expected_boundaries,
        ),
    ]


class HorizonBenchWorkflowTests(unittest.TestCase):
    def test_default_fixture_structure_domains_and_immutability(self) -> None:
        suite = load_cross_domain_workflows()
        self.assertIsInstance(suite, CrossDomainWorkflowSuite)
        self.assertEqual(DEFAULT_CROSS_DOMAIN_TASKS.name, "cross_domain_tasks.jsonl")
        self.assertTrue(DEFAULT_CROSS_DOMAIN_TASKS.is_file())
        self.assertEqual(len(suite.workflow_tasks), 12)
        self.assertEqual(len(suite.workflows), 4)
        self.assertEqual(len(suite.task_set.tasks), 12)
        self.assertEqual(suite.domains, EXPECTED_DOMAINS)
        self.assertEqual(set(suite.domains), set(EXPECTED_DOMAINS))
        self.assertTrue(all(task.split == "heldout" for task in suite.task_set.tasks))
        self.assertEqual(len(suite.task_by_id), 12)
        self.assertEqual(len(suite.workflow_by_id), 4)
        self.assertIsInstance(suite.task_by_id, MappingProxyType)
        self.assertIsInstance(suite.workflow_by_id, MappingProxyType)

        seen_domains = []
        for workflow in suite.workflows:
            self.assertIsInstance(workflow, WorkflowDefinition)
            self.assertEqual(len(workflow.tasks), 3)
            self.assertGreaterEqual(workflow.expected_boundaries, 12)
            self.assertGreaterEqual(workflow.expected_boundaries, len(workflow.tasks))
            self.assertTrue(any(task.recovery_boundary for task in workflow.tasks))
            self.assertEqual([task.step_index for task in workflow.tasks], [1, 2, 3])
            self.assertEqual({task.domain for task in workflow.tasks}, {workflow.domain})
            self.assertEqual({task.workflow_title for task in workflow.tasks}, {workflow.workflow_title})
            self.assertEqual({task.expected_boundaries for task in workflow.tasks}, {workflow.expected_boundaries})
            seen_domains.append(workflow.domain)
            previous_ids: list[str] = []
            for task in workflow.tasks:
                self.assertIsInstance(task, WorkflowTask)
                self.assertEqual(task.task_id, task.task.task_id)
                self.assertEqual(task.workflow_id, workflow.workflow_id)
                self.assertTrue(task.task.goal)
                self.assertTrue(task.task.constraint)
                self.assertIsInstance(task.depends_on, tuple)
                self.assertTrue(all(dependency in previous_ids for dependency in task.depends_on))
                self.assertIsInstance(task.recovery_boundary, bool)
                self.assertIsInstance(task.evidence_required, bool)
                previous_ids.append(task.task_id)
        self.assertEqual(sorted(seen_domains), list(EXPECTED_DOMAINS))
        self.assertEqual(len(set(seen_domains)), 4)

        with self.assertRaises(TypeError):
            suite.task_by_id["new-id"] = suite.workflow_tasks[0]  # type: ignore[index]
        with self.assertRaises(TypeError):
            suite.workflow_by_id["new-id"] = suite.workflows[0]  # type: ignore[index]
        with self.assertRaises(AttributeError):
            suite.workflows = ()  # type: ignore[misc]
        with self.assertRaises(TypeError):
            suite.workflow_tasks[0].depends_on[0] = "mutated"  # type: ignore[index]
        with self.assertRaises(TypeError):
            suite.workflow_tasks[0].task.metadata["workflow_id"] = "mutated"  # type: ignore[index]

    def test_task_source_identity_and_json_safe_output(self) -> None:
        suite = load_cross_domain_workflows()
        again = load_cross_domain_workflows(DEFAULT_CROSS_DOMAIN_TASKS)
        self.assertEqual(suite.task_set.source_name, "horizon-cross-domain-fixture")
        self.assertEqual(suite.task_set.source_revision, "v1")
        self.assertEqual(suite.task_set.split, "heldout")
        self.assertEqual(suite.task_set.source_sha256, again.task_set.source_sha256)
        self.assertEqual(suite.task_set.task_set_sha256, again.task_set.task_set_sha256)
        self.assertEqual(suite.identity()["source_sha256"], suite.task_set.source_sha256)
        self.assertEqual(suite.identity()["task_set_sha256"], suite.task_set.task_set_sha256)
        self.assertEqual(suite.identity()["task_ids"], list(suite.task_set.task_ids))
        self.assertEqual(suite.identity()["workflow_ids"], [workflow.workflow_id for workflow in suite.workflows])
        self.assertEqual(suite.identity()["domains"], list(EXPECTED_DOMAINS))
        self.assertNotIn("tasks", suite.identity())
        self.assertNotIn("workflows", suite.identity())

        identity_text = json.dumps(suite.identity(), allow_nan=False)
        as_dict_text = json.dumps(suite.as_dict(), allow_nan=False)
        self.assertIsInstance(json.loads(identity_text), dict)
        payload = json.loads(as_dict_text)
        self.assertEqual(payload["source_name"], "horizon-cross-domain-fixture")
        self.assertEqual(len(payload["workflow_tasks"]), 12)
        self.assertEqual(len(payload["workflows"]), 4)
        self.assertIsInstance(payload["task_ids"], list)
        self.assertIsInstance(payload["domains"], list)
        self.assertIsInstance(payload["workflows"][0]["task_ids"], list)
        self.assertIsInstance(payload["workflow_tasks"][0]["depends_on"], list)
        self.assertEqual(suite.as_dict()["task_set"]["task_ids"], list(suite.task_set.task_ids))

    def test_malformed_and_missing_metadata_are_rejected(self) -> None:
        valid = _valid_workflow("wf-alpha", "software_engineering")

        with tempfile.TemporaryDirectory() as temporary:
            missing_field = {key: value for key, value in valid[0].items() if key != "workflow_id"}
            missing_field["id"] = "wf-alpha-missing"
            path = _write_bytes(temporary, "missing_workflow_id.jsonl", _jsonl([missing_field, valid[1], valid[2]]))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            message = str(ctx.exception)
            self.assertIn("wf-alpha-missing", message)
            self.assertIn("workflow_id", message)

            blank = [dict(valid[0], id="wf-alpha-blank", workflow_title="   "), valid[1], valid[2]]
            path = _write_bytes(temporary, "blank_title.jsonl", _jsonl(blank))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("workflow_title", str(ctx.exception))
            self.assertIn("wf-alpha-blank", str(ctx.exception))

            zero_step = [dict(valid[0], step_index=0), valid[1], valid[2]]
            path = _write_bytes(temporary, "zero_step.jsonl", _jsonl(zero_step))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("step_index", str(ctx.exception))

            string_step = [dict(valid[0], step_index="1"), valid[1], valid[2]]
            path = _write_bytes(temporary, "string_step.jsonl", _jsonl(string_step))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("step_index", str(ctx.exception))

            negative = [dict(valid[0], expected_boundaries=-1), valid[1], valid[2]]
            path = _write_bytes(temporary, "negative_boundaries.jsonl", _jsonl(negative))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("expected_boundaries", str(ctx.exception))

            depends_on_string = [valid[0], dict(valid[1], depends_on="wf-alpha-01"), valid[2]]
            path = _write_bytes(temporary, "depends_on_string.jsonl", _jsonl(depends_on_string))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("depends_on", str(ctx.exception))
            self.assertIn("wf-alpha-02", str(ctx.exception))

            duplicate_depends = [valid[0], valid[1], dict(valid[2], depends_on=["wf-alpha-01", "wf-alpha-01"])]
            path = _write_bytes(temporary, "duplicate_depends_on.jsonl", _jsonl(duplicate_depends))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("depends_on", str(ctx.exception))
            self.assertIn("wf-alpha-03", str(ctx.exception))

            recovery_string = [valid[0], valid[1], dict(valid[2], recovery_boundary="true")]
            path = _write_bytes(temporary, "recovery_string.jsonl", _jsonl(recovery_string))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("recovery_boundary", str(ctx.exception))

            evidence_int = [valid[0], valid[1], dict(valid[2], evidence_required=1)]
            path = _write_bytes(temporary, "evidence_int.jsonl", _jsonl(evidence_int))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("evidence_required", str(ctx.exception))

            self.assertTrue(issubclass(WorkflowValidationError, ValueError))

    def test_invalid_ordering_cross_workflow_dependency_and_missing_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            gapped = _valid_workflow("wf-gap", "data_analysis")
            gapped[2] = dict(gapped[2], step_index=4)
            path = _write_bytes(temporary, "gapped_steps.jsonl", _jsonl(gapped))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            self.assertIn("wf-gap", str(ctx.exception))
            self.assertIn("step_index", str(ctx.exception))

            later_dep = _valid_workflow("wf-order", "research_synthesis")
            later_dep[1] = dict(later_dep[1], depends_on=["wf-order-03"])
            path = _write_bytes(temporary, "later_dependency.jsonl", _jsonl(later_dep))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            message = str(ctx.exception)
            self.assertIn("wf-order", message)
            self.assertIn("depends_on", message)
            self.assertIn("wf-order-02", message)

            first = _valid_workflow("wf-one", "operations")
            second = _valid_workflow("wf-two", "software_engineering")
            second[2] = dict(second[2], depends_on=["wf-two-01", "wf-one-02"])
            path = _write_bytes(temporary, "cross_workflow.jsonl", _jsonl(first + second))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            message = str(ctx.exception)
            self.assertIn("wf-two", message)
            self.assertIn("depends_on", message)
            self.assertIn("wf-one-02", message)

            no_recovery = _valid_workflow("wf-norecovery", "data_analysis")
            no_recovery[2] = dict(no_recovery[2], recovery_boundary=False)
            path = _write_bytes(temporary, "no_recovery.jsonl", _jsonl(no_recovery))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            message = str(ctx.exception)
            self.assertIn("wf-norecovery", message)
            self.assertIn("recovery_boundary", message)

            mismatched = _valid_workflow("wf-mismatch", "operations")
            mismatched[1] = dict(mismatched[1], domain="data_analysis", workflow_title="other title", expected_boundaries=20)
            path = _write_bytes(temporary, "inconsistent.jsonl", _jsonl(mismatched))
            with self.assertRaises(WorkflowValidationError) as ctx:
                load_cross_domain_workflows(path)
            message = str(ctx.exception)
            self.assertIn("wf-mismatch", message)
            self.assertTrue("domain" in message or "workflow_title" in message or "expected_boundaries" in message)

    def test_task_workflow_ordering_and_direct_script_import(self) -> None:
        suite = load_cross_domain_workflows()
        self.assertEqual(
            [task.task_id for task in suite.workflow_tasks],
            list(suite.task_set.task_ids),
        )
        self.assertEqual(
            [task.task_id for task in suite.workflow_tasks],
            [task.task_id for task in suite.task_set.tasks],
        )
        first_seen: list[str] = []
        for task in suite.workflow_tasks:
            if task.workflow_id not in first_seen:
                first_seen.append(task.workflow_id)
        self.assertEqual([workflow.workflow_id for workflow in suite.workflows], first_seen)
        for workflow in suite.workflows:
            self.assertIs(suite.workflow_by_id[workflow.workflow_id], workflow)
            self.assertEqual(
                [task.task_id for task in workflow.tasks],
                [task.task_id for task in sorted(workflow.tasks, key=lambda item: item.step_index)],
            )
            for task in workflow.tasks:
                self.assertIs(suite.task_by_id[task.task_id], task)
                self.assertEqual(task.domain, workflow.domain)

        spec = importlib.util.spec_from_file_location(
            "horizonbench_workflows_direct",
            ROOT / "benchmarks" / "horizonbench" / "workflows.py",
        )
        self.assertIsNotNone(spec)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
            imported = module.load_cross_domain_workflows()
        finally:
            sys.modules.pop(spec.name, None)
        self.assertEqual(imported.task_set.task_ids, suite.task_set.task_ids)
        self.assertEqual(imported.task_set.source_sha256, suite.task_set.source_sha256)
        self.assertEqual(imported.domains, suite.domains)
        self.assertTrue(issubclass(module.WorkflowValidationError, ValueError))


if __name__ == "__main__":
    unittest.main()
