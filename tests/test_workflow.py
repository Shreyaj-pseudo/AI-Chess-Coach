"""Workflow guards; all task execution uses temporary repositories and no APIs."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("workflow", REPO / "scripts" / "workflow.py")
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def task(task_id, deps=None, owner="agent"):
    return {
        "id": task_id, "title": "A real task", "track": "mvp", "owner": owner,
        "hours": "2-3", "depends_on": deps or [], "steps": ["Implement"],
        "acceptance": ["Verify"], "evidence": f"artifacts/{task_id}.md",
        **({"test_file": "test_example.py"} if owner == "agent" else {}),
    }


class ManifestTests(unittest.TestCase):
    def test_actual_manifest_is_complete_and_acyclic(self):
        tasks = json.loads((REPO / "docs" / "tasks.json").read_text(encoding="utf-8"))["tasks"]
        workflow.validate(tasks)
        self.assertEqual(tasks[0]["id"], "T01")
        self.assertEqual(len([item for item in tasks if item["id"].startswith("L")]), 20)

    def test_bad_graphs_and_incomplete_tasks_are_rejected(self):
        invalid = [[], [task("A"), task("A")], [task("A", ["B"])],
                   [task("A", ["B"]), task("B", ["A"])], [task("A", ["A"])]]
        broken = task("A")
        del broken["acceptance"]
        invalid.append([broken])
        for tasks in invalid:
            with self.subTest(tasks=tasks), self.assertRaises(ValueError):
                workflow.validate(tasks)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.manifest = self.root / "docs" / "tasks.json"
        self.manifest.parent.mkdir()
        self.manifest.write_text(json.dumps({"version": 1, "tasks": [
            task("A"), task("B", ["A"]), task("H", ["A"], "human")
        ]}), encoding="utf-8")
        self.patches = patch.multiple(workflow, ROOT=self.root, MANIFEST=self.manifest,
                                     STATE=self.root / ".workflow" / "state.json")
        self.patches.start()
        self.addCleanup(self.patches.stop)
        (self.root / "tests").mkdir()
        self.test_file = self.root / "tests" / "test_example.py"
        self.test_file.write_text(
            "import unittest\nclass Example(unittest.TestCase):\n"
            "    def test_works(self):\n        self.assertEqual(2 + 2, 4)\n", encoding="utf-8")
        (self.root / "artifacts").mkdir()
        self.proof = self.root / "artifacts" / "A.md"
        self.proof.write_text("Observed acceptance results", encoding="utf-8")

    def run_command(self, *args):
        output = io.StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            result = workflow.main(list(args))
        return result, output.getvalue()

    def read_state(self):
        return workflow.read_json(workflow.STATE, {})

    def verify(self):
        return self.run_command("verify", "A", "--evidence", "artifacts/A.md")

    def test_readonly_commands_do_not_create_progress(self):
        for command in ("validate", "status", "next", "prompt"):
            result, output = self.run_command(command)
            self.assertEqual(result, 0, output)
        self.assertFalse(workflow.STATE.exists())

    def test_dependencies_and_single_active_task_are_enforced(self):
        manifest = workflow.read_json(self.manifest)
        manifest["tasks"].append(task("C"))
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self.run_command("start", "B")[0], 1)
        self.assertEqual(self.run_command("start", "A")[0], 0)
        self.assertEqual(self.run_command("start", "B")[0], 1)
        self.assertEqual(self.run_command("start", "C")[0], 1)
        self.assertEqual(self.read_state()["A"]["status"], "in_progress")

    def test_completion_needs_real_fresh_checks(self):
        self.run_command("start", "A")
        self.assertEqual(self.run_command("complete", "A")[0], 1)
        result, output = self.verify()
        self.assertEqual(result, 0, output)
        self.assertTrue((workflow.STATE.parent / "A-tests.log").is_file())
        self.assertEqual(self.run_command("complete", "A")[0], 0)
        self.assertEqual(self.read_state()["A"]["status"], "done")

    def test_source_change_invalidates_verification(self):
        self.run_command("start", "A")
        self.assertEqual(self.verify()[0], 0)
        (self.root / "app.py").write_text("changed = True\n", encoding="utf-8")
        self.assertEqual(self.run_command("complete", "A")[0], 1)

    def test_source_change_during_checks_rejects_the_pass(self):
        self.run_command("start", "A")

        def change_source(pattern):
            (self.root / "app.py").write_text(pattern, encoding="utf-8")
            return True, "Simulated test pass for this guard test\n"

        with patch.object(workflow, "run_tests", side_effect=change_source):
            self.assertEqual(self.verify()[0], 1)
        self.assertNotIn("verification", self.read_state()["A"])

    def test_check_timeout_cannot_leave_a_previous_pass(self):
        self.run_command("start", "A")
        self.assertEqual(self.verify()[0], 0)
        with patch.object(workflow, "run_tests", side_effect=workflow.subprocess.TimeoutExpired("tests", 600)):
            self.assertEqual(self.verify()[0], 1)
        self.assertNotIn("verification", self.read_state()["A"])

    def test_evidence_change_invalidates_verification(self):
        self.run_command("start", "A")
        self.assertEqual(self.verify()[0], 0)
        self.proof.write_text("Changed results", encoding="utf-8")
        self.assertEqual(self.run_command("complete", "A")[0], 1)

    def test_failed_verification_clears_old_pass(self):
        self.run_command("start", "A")
        self.assertEqual(self.verify()[0], 0)
        self.test_file.write_text(
            "import unittest\nclass Broken(unittest.TestCase):\n"
            "    def test_failure(self):\n        self.fail('regression')\n", encoding="utf-8")
        self.assertEqual(self.verify()[0], 1)
        self.assertNotIn("verification", self.read_state()["A"])
        self.assertEqual(self.run_command("complete", "A")[0], 1)

    def test_missing_empty_and_skipped_test_suites_do_not_pass(self):
        self.run_command("start", "A")
        self.test_file.unlink()
        self.assertEqual(self.verify()[0], 1)
        self.test_file.write_text("# zero tests\n", encoding="utf-8")
        self.assertEqual(self.verify()[0], 1)
        self.test_file.write_text(
            "import unittest\nclass Skipped(unittest.TestCase):\n"
            "    @unittest.skip('not ready')\n    def test_skip(self):\n        pass\n", encoding="utf-8")
        self.assertEqual(self.verify()[0], 1)

    def test_regression_failure_blocks_otherwise_passing_task(self):
        self.run_command("start", "A")
        (self.root / "tests" / "test_regression.py").write_text(
            "import unittest\nclass Regression(unittest.TestCase):\n"
            "    def test_failure(self):\n        self.fail('old behavior broke')\n", encoding="utf-8")
        self.assertEqual(self.verify()[0], 1)
        self.assertNotIn("verification", self.read_state()["A"])

    def test_human_gate_cannot_use_agent_completion(self):
        self.assertEqual(self.run_command("approve", "H", "--reviewer", "User",
                                          "--evidence", "artifacts/A.md")[0], 1)
        self.run_command("start", "A")
        self.verify()
        self.run_command("complete", "A")
        self.assertEqual(self.run_command("start", "H")[0], 1)
        self.assertEqual(self.run_command("complete", "H")[0], 1)
        self.assertEqual(self.run_command("approve", "H")[0], 1)
        self.assertEqual(self.run_command("approve", "H", "--reviewer", "User",
                                          "--evidence", "artifacts/A.md")[0], 0)
        self.assertEqual(self.read_state()["H"]["status"], "done")

    def test_dataset_approval_requires_successful_actual_batch_audit(self):
        manifest = workflow.read_json(self.manifest)
        manifest["tasks"].append({**task("L01", owner="human"), "batch": {"partition": "train"}})
        self.manifest.write_text(json.dumps(manifest), encoding="utf-8")
        with patch.object(workflow.subprocess, "run", return_value=workflow.subprocess.CompletedProcess([], 1, "", "bad labels")):
            self.assertEqual(self.run_command("approve", "L01", "--reviewer", "User",
                                              "--evidence", "artifacts/A.md")[0], 1)
        self.assertNotIn("L01", self.read_state())

    def test_blocks_can_be_resumed_without_faking_completion(self):
        self.run_command("start", "A")
        self.assertEqual(self.run_command("block", "A")[0], 1)
        self.assertEqual(self.run_command("block", "A", "--reason", "Missing credentials")[0], 0)
        self.assertEqual(self.read_state()["A"]["status"], "blocked")
        self.assertEqual(self.run_command("start", "A")[0], 0)
        self.assertNotIn("verification", self.read_state()["A"])

    def test_evidence_and_mutations_require_explicit_inputs(self):
        self.assertEqual(self.run_command("start")[0], 1)
        self.assertEqual(self.run_command("start", "unknown")[0], 1)
        self.run_command("start", "A")
        self.assertEqual(self.run_command("verify", "A")[0], 1)
        self.assertEqual(self.run_command("verify", "A", "--evidence", "missing.md")[0], 1)
        self.assertEqual(self.run_command("approve", "A", "--reviewer", "User", "--evidence", "artifacts/A.md")[0], 1)


if __name__ == "__main__":
    unittest.main()
