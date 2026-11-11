"""Track one local coding task at a time; completion requires fresh test evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "tasks.json"
STATE = ROOT / ".workflow" / "state.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_state(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(STATE)


def validate(tasks):
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Task manifest must contain tasks")
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs")
    for task in tasks:
        required = ("id", "title", "track", "owner", "hours", "depends_on", "steps", "acceptance", "evidence")
        if any(not task.get(key) for key in required if key != "depends_on"):
            raise ValueError(f"Incomplete task: {task.get('id')}")
        if task["owner"] not in ("agent", "human"):
            raise ValueError(f"Invalid owner: {task['id']}")
        if task["owner"] == "agent" and not re.fullmatch(r"test_[A-Za-z0-9_]+\.py", task.get("test_file", "")):
            raise ValueError(f"Missing task-specific test file: {task['id']}")
        if task["id"] in task["depends_on"] or any(dep not in ids for dep in task["depends_on"]):
            raise ValueError(f"Invalid dependency: {task['id']}")
    remaining = {task["id"]: set(task["depends_on"]) for task in tasks}
    while remaining:
        ready = {key for key, deps in remaining.items() if not deps}
        if not ready:
            raise ValueError("Dependency cycle")
        remaining = {key: deps - ready for key, deps in remaining.items() if key not in ready}


def fingerprint():
    """Ignore secrets, source corpora and runtime output, but include tests/fixtures."""
    digest = hashlib.sha256()
    excluded = {".git", ".workflow", ".venv", "venv", "env", "__pycache__", "node_modules", "stockfish", "artifacts", ".pytest_cache"}
    for directory, names, files in os.walk(ROOT):
        names[:] = sorted(name for name in names if name not in excluded)
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_symlink() or name.startswith(".env") or path.suffix in (".pyc", ".gguf", ".safetensors"):
                continue
            if path.relative_to(ROOT).parts[0] == "data" and path.suffix not in (".py", ".md", ".toml"):
                continue
            digest.update(path.relative_to(ROOT).as_posix().encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def evidence_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("Evidence must be a nonempty file inside this repository")
    return path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ready(task, state):
    return all(state.get(dep, {}).get("status") == "done" for dep in task["depends_on"])


def show(task):
    print(f"{task['id']}: {task['title']}\nOwner: {task['owner']} | Track: {task['track']} | Estimate: {task['hours']} hours")
    print("Dependencies: " + (", ".join(task["depends_on"]) or "none"))
    for heading, key in (("Implementation", "steps"), ("Acceptance checks", "acceptance")):
        print(f"\n{heading}:")
        for item in task[key]:
            print(f"- {item}")
    print("\nEvidence: " + task["evidence"])
    if task.get("test_file"):
        print("Task test: tests/" + task["test_file"])


def run_tests(pattern):
    if not list((ROOT / "tests").glob(pattern)):
        raise ValueError(f"Missing tests: {pattern}")
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", pattern, "-v"],
        cwd=ROOT, capture_output=True, text=True, timeout=600,
    )
    output = result.stdout + result.stderr
    print(output, end="")
    match = re.search(r"Ran (\d+) tests? in", output)
    # A successful exit with zero tests or skipped checks is not a task pass.
    passed = result.returncode == 0 and match is not None and int(match[1]) > 0
    passed = passed and not re.search(r"\b(skipped=|expected failures=|unexpected successes=)", output)
    return passed, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "status", "next", "show", "prompt", "start", "verify", "complete", "block", "approve"))
    parser.add_argument("task", nargs="?")
    parser.add_argument("--track", choices=("mvp", "dataset", "model", "publish", "business"))
    parser.add_argument("--evidence", help="Repository-relative file with actual results")
    parser.add_argument("--reviewer", help="Human reviewer name, for approve only")
    parser.add_argument("--reason", help="Concrete blocker, for block only")
    args = parser.parse_args(argv)
    try:
        if args.command in ("start", "verify", "complete", "block", "approve") and not args.task:
            raise ValueError("A task ID is required for changes")
        manifest = read_json(MANIFEST)
        if not manifest or manifest.get("version") != 1:
            raise ValueError("Missing or unsupported task manifest")
        tasks = manifest["tasks"]
        validate(tasks)
        state = read_json(STATE, {})
        if args.command == "validate":
            print(f"Validated {len(tasks)} tasks and their dependency graph")
            return 0
        if args.command == "status":
            for task in tasks:
                if not args.track or task["track"] == args.track:
                    item = state.get(task["id"], {})
                    status = item.get("status", "pending")
                    print(f"{task['id']:4} {status:11} {task['owner']:5} {task['title']}")
                    if item.get("reason"):
                        print("     Blocker: " + item["reason"])
            return 0
        active = [task for task in tasks if state.get(task["id"], {}).get("status") == "in_progress"]
        if args.task:
            task = next((item for item in tasks if item["id"] == args.task), None)
            if task is None:
                raise ValueError("Unknown task ID")
        else:
            task = next((item for item in active if not args.track or item["track"] == args.track), None)
            if task is None:
                task = next((item for item in tasks if (not args.track or item["track"] == args.track)
                             and state.get(item["id"], {}).get("status", "pending") == "pending"
                             and ready(item, state)), None)
        if task is None:
            print("No eligible task. Check status for human gates, blocked tasks, or unfinished dependencies.")
            return 0
        task_id = task["id"]
        item = state.get(task_id, {})
        if args.command in ("show", "next", "prompt"):
            if args.command == "prompt":
                print("Read docs/DAILY_WORKFLOW.md and docs/PLAN.md. Work only on the task below.\n"
                      "Inspect its callers and existing tests first. Preserve unrelated user changes.\n"
                      "Use the smallest complete change; do not expand the roadmap.\n"
                      "An agent must never run approve or invent human review, credentials, model results or test passes.\n"
                      "For an agent task: start it, implement its acceptance checks, add meaningful task tests,\n"
                      "write actual results to its evidence file, run verify, then complete.\n"
                      "Stop at a human/external gate and report precisely what is needed.\n"
                      "Finish at most ONE implementation task per session; return a short handoff.\n")
            show(task)
            return 0
        if args.command in ("start", "approve"):
            if item.get("status") == "done":
                raise ValueError("Task is already complete")
            if not ready(task, state):
                raise ValueError("Dependencies are not complete")
            if active and active[0]["id"] != task_id:
                raise ValueError(f"Finish or block {active[0]['id']} first")
        if args.command == "start":
            if task["owner"] != "agent":
                raise ValueError("Human task: collect real review evidence and use approve")
            state[task_id] = {"status": "in_progress", "started_at": now()}
        elif args.command == "block":
            if item.get("status") != "in_progress" or not args.reason:
                raise ValueError("Block requires an active task and --reason")
            state[task_id] = {**item, "status": "blocked", "reason": args.reason, "blocked_at": now()}
            state[task_id].pop("verification", None)
        elif args.command == "verify":
            if task["owner"] != "agent" or item.get("status") != "in_progress":
                raise ValueError("Start an agent task before verification")
            item.pop("verification", None)
            state[task_id] = item
            write_state(state)
            if not args.evidence:
                raise ValueError("Verification requires --evidence with real acceptance-check results")
            proof = evidence_path(args.evidence)
            before = fingerprint()
            targeted, log = run_tests(task["test_file"])
            regression, regression_log = run_tests("test*.py") if targeted else (False, "Regression not run\n")
            log_path = STATE.parent / f"{task_id}-tests.log"
            log_path.write_text(log + regression_log, encoding="utf-8")
            if not targeted or not regression or before != fingerprint():
                raise ValueError("Verification failed, skipped tests, or source changed during checks; task stays active")
            state[task_id]["verification"] = {
                "at": now(), "fingerprint": before, "evidence": str(proof.relative_to(ROOT)),
                "evidence_hash": file_hash(proof), "test_log": str(log_path.relative_to(ROOT)),
            }
        elif args.command == "complete":
            verification = item.get("verification", {})
            if task["owner"] != "agent" or item.get("status") != "in_progress" or not verification:
                raise ValueError("Fresh passing verification is required")
            proof = evidence_path(verification["evidence"])
            if fingerprint() != verification["fingerprint"] or file_hash(proof) != verification["evidence_hash"]:
                raise ValueError("Source or evidence changed; run verify again")
            state[task_id] = {**item, "status": "done", "completed_at": now()}
        elif args.command == "approve":
            if task["owner"] != "human" or not args.reviewer or not args.evidence:
                raise ValueError("Human approval requires a human task, --reviewer and --evidence")
            proof = evidence_path(args.evidence)
            if task.get("batch"):
                batch = task["batch"]
                result = subprocess.run(
                    [sys.executable, "-m", "coach.dataset", "audit-batch", "--batch", task_id,
                     "--expected-count", "25", "--partition", batch["partition"]],
                    cwd=ROOT, capture_output=True, text=True, timeout=600,
                )
                print(result.stdout + result.stderr, end="")
                if result.returncode != 0:
                    raise ValueError("Dataset batch audit failed; human approval was not recorded")
            state[task_id] = {"status": "done", "reviewer": args.reviewer, "evidence": str(proof.relative_to(ROOT)),
                              "evidence_hash": file_hash(proof), "completed_at": now()}
        write_state(state)
        print(f"{task_id}: {state[task_id]['status']}" + (" (checks passed)" if args.command == "verify" else ""))
        return 0
    except (ValueError, KeyError, OSError, subprocess.TimeoutExpired) as error:
        print(f"Workflow error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
