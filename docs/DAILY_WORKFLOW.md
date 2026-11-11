# Daily agent workflow

This breaks [the audited plan](PLAN.md) into **56 tasks: 28 coding tasks and 28 human tasks**, including twenty review batches of twenty-five labels. [tasks.json](tasks.json) defines every task's detailed steps, prerequisites, acceptance criteria, test filename and evidence path. Use `show ID` to read a complete task card.

"Daily" means one focused work session, not an automatic daily launch. Most coding tasks take 2-4 focused hours; a 25-label review can take 4-8 hours and span several days. Dependencies determine readiness. The runner tracks work and runs checks; your coding agent implements the tasks. No background scheduler, model request, training run or deployment has been started.

## Start today

The workflow needs only Python's standard library:

```powershell
python scripts/workflow.py validate
python -m unittest discover -s tests -p test_workflow.py -v
python scripts/workflow.py next --track mvp
python scripts/workflow.py prompt --track mvp
```

Give a coding agent this kickoff instruction:

> Read docs/DAILY_WORKFLOW.md and docs/PLAN.md. Run python scripts/workflow.py prompt --track mvp. Complete exactly that next eligible agent task, following its acceptance checks and test gate. Preserve unrelated changes. Write actual results to its evidence file. Stop at human or external dependencies and report what is needed. Do not run human approvals or invent test/benchmark results. End with the task ID, changed behavior, test results and next eligible task.

Use `--track dataset`, `--track model`, `--track publish` or `--track business` in later sessions. Filtering never bypasses dependencies. Default selection considers the entire backlog. `show T05` displays a task even if its prerequisites are unfinished.

## Daily loop and completion gate

1. Inspect `git status --short`, task status and prerequisites. Resume an interrupted active task before claiming another.
2. Read affected code, callers and tests; run the existing offline suite before editing. Record unrelated failures rather than concealing them.
3. Start the explicit task. Implement its steps and its named behavioral test file, covering the stated edge cases.
4. Write an evidence file with actual acceptance results, commands/exit statuses, test counts, manual/live checks and limitations. Label simulated output clearly.
5. Run verification, repair failures, and complete only after a fresh pass. Finish at most one implementation task per agent session; a large task may resume tomorrow.
6. Hand off the task ID, delivered behavior, verification result and next eligible task. Stop at human/external dependencies with a concrete blocker.

Example, after implementing the first task:

```powershell
python scripts/workflow.py start T01
# Implement T01 and tests/test_T01_environment.py.
# Write artifacts/tasks/T01.md with observed results.
python scripts/workflow.py verify T01 --evidence artifacts/tasks/T01.md
python scripts/workflow.py complete T01
python scripts/workflow.py next --track mvp
```

`verify` runs the task's named test file, then **all offline tests**. Missing/zero-test suites, failures, skipped tests and expected failures do not qualify. Keep credential-dependent smoke checks in separately invoked scripts, not silently skipped tests. A test that merely checks that a file exists is insufficient for behavioral acceptance.

A pass records the source fingerprint, evidence hash and test log. Source/tests/docs/evidence changes before `complete` invalidate that pass. Failed verification removes an earlier pass. One agent task may be active at a time. Do not hide missing prerequisites in tests or rewrite tests to bless a defect.

```powershell
python scripts/workflow.py block T17 --reason "Real acceptance run needs configured API credentials"
python scripts/workflow.py status
# After resolving the blocker:
python scripts/workflow.py start T17
```

Logs/state are in gitignored `.workflow/`. Generated evidence/data/models belong in gitignored `artifacts/`. Back up those folders deliberately: they are not committed project history. Creating this workflow marks no chess task complete. The fingerprint excludes secrets, raw `data/`, binaries and generated `artifacts/`; each task's tests and evidence must validate those external inputs/results separately.

## Session map

| Suggested sequence | Work | Required exit evidence |
| --- | --- | --- |
| Coding sessions 1-3: T01-T03 | Environment, PGN replay, records/scores | Fresh setup and legal/invalid input checks |
| Sessions 4-6: T04-T06 | Engine reuse, candidate/played lines, board facts | Real engine smoke plus behavioral tests |
| Sessions 7-9: T07-T09 | Openings, tablebases, critical moments | API-failure and deterministic selection cases |
| Sessions 10-12: T10-T12 | Model/OpenRouter adapter, grounded cards, integration | Ten-position suite; live/simulated results distinguished |
| Sessions 13-16: T13-T16 | Jobs, upload, report board, feedback | HTTP tests and actual browser checks |
| Session 17: T17 | Local acceptance/performance | Twenty real game runs, actual timings/failures |
| Human T18 | Learner/demo review | Real feedback and a transfer exercise |
| Sessions 18-19: T19-T20 | Annotation, corpus/splits and batch audit | Approved-only exports and leakage checks |
| Human T21 | Draft-source permission | Exact model/provider/terms and permitted use |
| Session 20: T22 | Resumable draft generation | Budget, interruption and provenance checks |
| Human T23 | Thirty-example pilot | Chess review, stronger-player sample and measured review time |
| Session 21: T24 | Evaluation harness | Blind forms, error taxonomy and leakage checks |
| Session 22: T25 | Portfolio package | Runnable release and honest measured claims |
| Human L01-L20 | Five hundred reviewed labels | Twenty approved, audited batches |
| Session 23: T26 | Untuned baselines and conversion smoke | Actual validation outputs, memory and speed |
| Human T27 | Compute/training decision | Real permitted compute and any spending approval |
| Session 24: T28 | QLoRA run | Actual checkpoint, logs and split isolation |
| Session 25: T29 | Locked test evaluation | Held-out results and blind rating forms |
| Human T30 | Blind model-quality review | Actual ratings and model release decision |
| Session 26: T31 | Ollama export and re-evaluation | Real local quantized-model results |
| Session 27: T32 | Open-source release candidate | License inventory and fresh-install checks |
| Human T33 | Beta destination/publish approval | Real target and explicit authorization |
| Session 28: T34 | Hosted beta | Actual deployment, quota and access checks |
| Human T35-T36 | Four-week feedback; optional business research | Real usage/interviews and an evidence-based decision |

This is not a promise of 28 calendar days. Human labeling is likely the largest investment. T25 is a complete portfolio milestone even if training/public hosting is blocked. T32 can follow T25 without waiting for the model experiments.

## Human review

An agent cannot mark a human task done through `complete`. After the review actually happens, the human records its outcome:

```powershell
python scripts/workflow.py show T18
python scripts/workflow.py approve T18 --reviewer "Shreyaj" --evidence artifacts/tasks/T18.md
```

This is a local trust convention, not authenticated identity verification. The runner cannot prove who typed the name, whether a reviewer really participated, or whether a prose explanation is good. Agents must not invoke `approve` for the user or invent observations. Inspect the tests themselves: a passing suite does not prove the assertions cover the requirement.

Missing credentials, unavailable GPUs, unresolved permissions and failed external services leave the affected live/training/deployment task blocked. Continue independent eligible work if useful. Do not report a mocked result as an actual run.

Use this evidence structure:

```text
Task ID:
Acceptance criteria: each criterion and observed result
Commands and exit statuses:
Task-specific and regression tests: counts and results
Live/manual checks: actual engine/model/browser details
Changed behavior:
Limits/blockers: include any simulated or untested behavior
```

## Generating training and evaluation examples

Software tests verify program behavior. Held-out chess examples evaluate model quality. Keep these separate.

The coding agent can generate legal position candidates, run engine analysis, compute features, create deterministic software fixtures, propose explanation drafts, check schemas, detect leakage and prepare blinded comparisons. You approve each trusted explanation. A stronger chess reviewer checks difficult samples. Legal moves alone do not establish correct strategic reasoning.

The thirty-example pilot is outside the final 500. Source games/positions are partitioned before drafts are written. The split is 400 training, 50 validation and 50 held-out test records; never feed reference answers to the evaluated model. Select models/checkpoints on validation only. Repeatedly tuning after reading test results requires a new independent test set.

| Batches | Contents |
| --- | --- |
| L01-L04 | 100 opening training examples |
| L05-L08 | 100 tactical training examples |
| L09-L12 | 100 positional training examples |
| L13-L16 | 100 endgame training examples |
| L17-L18 | 50 validation examples across themes |
| L19-L20 | 50 held-out examples across themes |

Each batch has twenty-five new records and its own review file. Approving a label batch invokes the actual dataset audit implemented in T20:

```powershell
python -m coach.dataset audit-batch --batch L01 --expected-count 25 --partition train
python scripts/workflow.py approve L01 --reviewer "Shreyaj" --evidence artifacts/labels/L01-review.md
```

The `coach.dataset` command is a future T20 deliverable and **does not exist yet**. It must return nonzero for incorrect counts, unapproved/empty labels, wrong themes/partitions, duplicates, missing provenance or unresolved critical claims. A failed audit blocks approval. Creating twenty-five rows is insufficient; real approval and factual review are required.

## OpenRouter free-model option

As checked on 2026-11-10, free variants allow 20 requests/minute and 50/day without purchased credits; failures consume the allowance. Capacity and available models can change. [Free-model limits](https://openrouter.ai/collections/free-models), [pricing](https://openrouter.ai/pricing)

T22 uses a conservative default of twenty attempts per session, forty per rolling 24 hours and at least four seconds between calls. Repairs and failures count. Checkpoint after every request, stop on quota exhaustion, and never fall back to a paid route. Twenty-five drafts can span sessions; manual approval is slower than generation anyway.

The twenty-game live benchmark can also span several days under a free quota. Preserve partial measurements, count failed requests, and complete T17 only after the actual runs finish; do not extrapolate a small sample into twenty completed games.

Select an explicit free model/provider, verify zero pricing, disable fallback and record the actual serving provider/returned model. Avoid `openrouter/free` for benchmark/data runs because it can change the model between requests. If a route disappears, stop and explicitly version a replacement rather than silently mixing sources.

Free price does not establish training rights. Check the underlying provider/model terms for output reuse separately from its use of your prompts. T21 records the permitted draft source; if unclear, write original human labels or use an explicitly permitted local model. [OpenRouter terms](https://openrouter.ai/terms)

No API key is required to use the workflow. Configure it privately when implementing the model task; never put it in task evidence or commit it. This change made no live generation calls and created no verified chess labels.
