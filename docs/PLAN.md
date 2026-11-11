# Chess AI Coach: Audited Implementation Plan

Updated: 2026-11-10. This replaces the conversation's proposed plan for implementation. Rationale, grades, market context, and limitations are in [PLAN_AUDIT.md](PLAN_AUDIT.md).

Status: implementation started; use `python scripts/workflow.py status` for verified task completion. Primary objective: a strong, demonstrable portfolio project. Secondary paths: an open-source release and, only after user validation, a possible business.

Execution backlog: [DAILY_WORKFLOW.md](DAILY_WORKFLOW.md) explains the daily agent loop and test gates; [tasks.json](tasks.json) defines the ordered detailed tasks. OpenRouter is an optional hosted draft source, subject to provenance and permission checks.

## 1. Product contract

An improving club player pastes or uploads one standard PGN, chooses White or Black, and receives detailed coaching about instructive decisions for that side.

- Accept one UTF-8 PGN up to 256 KB and 200 plies. Reject malformed moves, parse errors, multiple games, unsupported variants, and invalid initial positions with clear messages. Ignore comments/side variations for analysis; allow valid standard-chess setup FENs.
- Derive positions from the game instead of accepting an independent FEN that may contradict it. Preserve move history for repetition and draw handling.
- Show the three most useful moments first, with up to fifteen available. Do not invent moments to meet a minimum count. Explain when fewer are found.
- Each card includes a verdict, what the move accomplishes, a supported recommendation or playable alternative, a concrete comparison, replayable supporting lines, and a transferable lesson. Start with a short summary; expand for detail.
- Explain when several candidates are close. Do not invent the player's intention or treat an engine's first choice as uniquely correct.
- Initial audience: approximately 800-1800 online players. English only. No accounts, game-URL imports, live play assistance, open chat, or generated whole-game learning summary in v1.

The first deliverable is a local website. Neither 500 labels nor a fine-tuned model is required to demonstrate it.

## 2. Analysis and explanation design

### Data flow

```text
PGN + learner color
  -> strict parse and legal replay
  -> bounded engine scan / opening and endgame evidence
  -> rank teaching moments
  -> deeper candidate analysis + played-move continuation
  -> immutable AnalysisRecord
  -> replaceable explanation model
  -> structural/factual checks + uncertainty handling
  -> CoachingCards and feedback
```

Use python-chess as the shared rules/PGN layer. Consolidate the phase prompt modules only after their callers use the replacement. A single worker owns one persistent Stockfish process; searches remain sequential. Missing binaries and engine exits must produce visible failures and clean process shutdown.

Use a configurable engine path with environment override and PATH discovery. Pin tested dependencies; separate training dependencies from the runtime. Keep existing data and draft labels intact; exclude unapproved records from exports. Remove unused packages only after confirming they have no callers.

### Engine evidence

- Scan the learner's pre-move position and its played continuation. Normalize scores to the learner. Record scores as centipawn or mate values with an explicit point of view; do not map mate scores into ordinary teaching centipawns.
- Deep-analyze selected positions with up to three engine candidates. If the played move is absent, additionally search that move as a restricted root. Retain the opponent's strongest continuation after it.
- Use the best candidate's score and PV from the same result. Record separate search metadata for forced-move comparisons. Materially inconsistent search results trigger reanalysis or a qualified verdict.
- Start with one engine thread, 128 MB hash, a 0.15-second shallow search, a 1.5-second deep search, and a 1.5-second restricted played-move search when necessary. These are tuning defaults, not evidence of sufficient depth. Record achieved depth, nodes, engine version and elapsed time.
- A one-pawn-or-larger preliminary loss is eligible for priority review, not automatically a proven mistake. Treat ordinary candidate gaps below 0.3 pawns conservatively. Reconfirm consequential claims in the deep pass; handle mates and draw outcomes separately. Do not manufacture a positive loss from inconsistent results.
- Preserve the full legal PV internally. Supply short lines initially, extending through the claimed tactical payoff when needed. If it cannot fit the model's context, narrow the explanation rather than silently truncate its evidence.

`python-chess` supports MultiPV, root-move restrictions, viewpoint normalization, and move-stack transmission. [Engine API](https://python-chess.readthedocs.io/en/latest/engine.html)

### Openings and endgames

- Use current documented Lichess endpoints, not the repository's assumed legacy URLs. Cache exact-position results; serialize external requests, set timeouts, and respect rate-limit responses. [Lichess API](https://lichess.org/api), [API usage guidance](https://lichess.org/page/api-tips)
- Combine Masters and Lichess game statistics, showing source, filters and sample counts. Use rapid/classical Lichess games in the 1200-1800 range as the initial practical sample.
- Query opening context only within the first twenty full moves for the initial release; beyond that, retain the last identified opening as historical context. A sample of fewer than twenty games is marked sparse, not theory evidence. Both cutoffs are initial heuristics recorded in the configuration.
- A named opening and a common move do not establish soundness. Engine-free cards may describe common play and curated plans. Claims of superiority or tactical errors require engine or curated analysis. If opening lookup fails, retain engine analysis without pretending it is book theory.
- Use tablebase results only for supported eligible positions, with castling, fifty-move and repetition context checked. Distinguish ordinary wins/draws from rule-dependent outcomes. Do not treat distance-to-zeroing as mate distance. Fall back visibly to engine analysis when unavailable.

### Facts, claims, and model contract

Version these Pydantic records:

| Record | Required contents |
| --- | --- |
| `AnalysisRecord` | Source hash, ply, board and history reference, learner, played SAN/UCI, candidate lines, played-move line, normalized scores, relevant features, evidence provenance, search metadata and uncertainty flags. |
| `CoachingCard` | Position reference, verdict, summary, move explanation, recommendation, candidate comparison, supporting-line references, lesson, and any uncertainty note. |
| `TrainingExample` | Analysis reference, immutable approved output revision, authorship/permission provenance, reviewer status, schema/prompt versions and dataset split. |

Derive inexpensive facts first: material, legal checks/captures, attacks/defenses, pins, open files, passed/isolated/doubled pawns, and feature changes along candidate lines. Label heuristics honestly. Attack counts alone must not establish hanging pieces or a tactical threat.

Represent concrete claims with evidence references to an exact line and ply. Validate referenced moves and board facts deterministically. Strategic prose still needs human evaluation. When support is insufficient, omit or qualify the claim.

Only pass the game prefix available at the reviewed move to the model. Treat PGN headers/comments and feedback as untrusted content, never prompt instructions. Escape all rendered text. Replay supplied lines into precomputed board states; free variation entry is deferred.

Use a small `generate_card(analysis_record)` interface for hosted and Ollama inference. Count tokens with each model's tokenizer and reserve room for its output. Return structured JSON; permit one repair attempt for invalid responses. A failed factual check suppresses the card and records the failure. Do not present invalid output as coaching.

Gemini can be evaluated as the local hosted baseline, subject to its terms. Training-label reuse is not authorized by this plan; default to original human labels or an explicitly permitted source. Public provider eligibility must be checked again before deployment. [Provider terms](https://aistudio.google.com/docs/terms)

## 3. Local application and first release gates

Use FastAPI, Jinja templates, lightweight JavaScript, a pinned board component, and SQLite. The board should support keyboard navigation and visible text notation. Keep the reviewer interface local-only. Choose and verify board-code and piece-asset licenses before bundling.

Minimum routes:

- `POST /reviews`: PGN and learner color; return job ID and private capability token.
- `GET /reviews/{id}`: token-protected progress, status and final result.
- `POST /reviews/{id}/feedback`: helpful/not-helpful, issue category and optional comment.
- Local reviewer routes: inspect evidence, edit card fields, approve/reject revisions and export approved examples.

One in-process bounded queue, one worker, and one web process are sufficient locally. Keep status in SQLite. Mark interrupted work failed on restart; do not retry jobs indefinitely. Timeouts, cancellation, engine errors and model quotas must release worker resources. Preserve the PGN in the browser for resubmission, with a clear reset control.

Retain ordinary local reports for 24 hours by default. Keep deliberate annotation projects until manually removed. Do not log raw PGNs or API credentials. Remove identifying headers before remote requests, without claiming that move sequences are fully anonymous. Opted-in feedback remains untrusted until reviewed.

### Build in this order

1. **One trustworthy card:** repair analysis, compare a best move and the actual played move, generate a card, and replay both lines. Use ten curated positions. This is an internal milestone, not a change to the full-game goal.
2. **Whole-game local demo:** add PGN validation, moment ranking, progress, report navigation, detailed cards, and feedback. Rank deep-confirmed evaluation losses first, then forcing-choice/strong-move examples and opening decisions. Deduplicate adjacent moments teaching the same line; select at most fifteen. Quiet positional selection remains heuristic and is tested separately.
3. **Pilot evaluation and annotation:** review thirty diverse examples, revise the schema/rubric, and demonstrate that users understand the cards. Obtain an independent stronger-player review of at least ten difficult examples before calling labels trusted.
4. **Portfolio release:** document setup, limitations, benchmark results, a short demonstration and failure cases. A working local application plus evidence of quality is a valid completed portfolio milestone.

Target median review time is 1-3 minutes on the measured local setup; report p50 and p95 over twenty representative games, engine/model versions, cache state and failure counts. Until measured, make no latency guarantee. Show progress and allow cancellation if slower. Track engine, generation, network and queue latency separately.

## 4. Dataset and custom-model experiment

Proceed after the thirty-example pilot and permissions/compute checks pass.

- Stream public PGNs and preserve provenance. Existing Lichess exports can seed the project; Lichess releases its database exports under CC0. Do not infer that arbitrary annotations or video transcripts share that permission. [Lichess database](https://database.lichess.org/)
- Use independently written answers by default. Permission-cleared model drafts are optional. The user reviews every accepted label; use a stronger chess reviewer on difficult claims and a sample of the completed set.
- Target 500 approved examples, roughly 125 each for opening decisions, tactics, positional middlegames and endgames. These are coverage targets, not reasons to accept weak examples. Add motif and example-type tags, including strong choices, equivalent candidates, and uncertain cases.
- Deduplicate exact positions across splits and keep source games together. If games transpose to the same labeled position, assign them to the same split or omit the collision. Track related opening-family challenge positions separately.
- Split approximately 400/50/50 into train/validation/test. Lock test data before model selection. Keep failed/rejected drafts and editor history separate; never export them as targets.
- Benchmark untuned OLMo 2 7B Instruct and Qwen3 4B plus the hosted baseline on validation positions. Verify actual 4-bit memory, tokenizer behavior, structured output, and Ollama conversion on a small sample before committing. Choose the model with fewer severe errors; break ties by blind usefulness score, then local speed.
- Train only in an environment permitting the workload. Colab availability and its free-tier restriction on chess training require clarification for this use; it is not a guaranteed resource. Persist checkpoints and budget for preemption. [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- For a permitted GPU experiment, begin with 4-bit QLoRA, batch size one, gradient accumulation eight, gradient checkpointing, LoRA rank sixteen and learning rate 0.0001. Use at most two initial epochs and select on validation quality. Start at 2048 tokens; increase only if the full input/output fits available memory. Never silently truncate approved examples.
- Pin all model, tokenizer, training-library and conversion revisions. Export the selected adapter, training manifest and logs. Merge/convert to a supported GGUF and evaluate the quantized Ollama artifact again; fitting 16 GB RAM does not establish speed or quality.

Freeze a human rubric with separate 1-5 scores for factual correctness, evidence support, clarity and usefulness. Define a critical error as an illegal cited line, invented material/check/mate, wrong-side evaluation, or unsupported concrete tactic changing the lesson. Report raw error counts and reviewer disagreement.

Pilot success requires better blind usefulness than the untuned base without increasing critical errors, and zero known critical errors left unfixed in release examples. Report small-test-set uncertainty. Formatting success and comparisons with Gemini are secondary metrics; the original 98%/5%/0.5-point thresholds are not evidence-backed guarantees. If fine-tuning fails to help, publish the result and retain the stronger baseline.

## 5. Verification, publication, and next decisions

Required checks:

- Legal PGN replay; malformed/multi-game rejection; SAN/UCI conversion; setup FENs; castling, en passant, promotions and terminal positions.
- Score viewpoint, mates, comparable search provenance, the played move outside MultiPV, equivalent candidates, and unstable analysis.
- Opening transpositions, sparse statistics, opening lookup after the reviewed move, unavailable APIs, tablebase rule context and repetition.
- False prose claims despite legal lines, prompt injection in comments/feedback, safe rendering, oversized inputs, invalid JSON and repair failure.
- No leakage across dataset splits, approved-only exports, correct token budgets, interrupted jobs, engine crashes and worker cleanup.
- Browser smoke checks for upload, orientation, keyboard use, line replay, progress, feedback and error recovery.
- Blind review of a separate acceptance set containing tactical and quiet positional examples, with actual readers completing a small transfer exercise.

Publish only after a fresh install works, the acceptance set is reviewed, and measured results are documented. Keep accounts/chat and infrastructure expansion deferred. Add a glossary and architecture decisions when the relevant design is implemented, not as proof of progress by themselves.

For a later hosted beta, recheck free-provider availability and terms at deployment time. Use an unlisted URL as chosen, but add payload limits, a small bounded queue, per-client and global quotas, token-protected reviews, expiry, training opt-in, and an operator kill switch. Public reviewer/admin routes stay disabled. A free tier may be unavailable; the zero-cost fallback is continuing local testing and releasing reproducible source, not promising continuous service.

For open source, document licenses and provenance for code, models, dependencies and datasets. Before pursuing a startup, use the separate audit's customer, retention, paid-pilot and cost gates. Do not expand the portfolio MVP into business infrastructure before those signals exist.
