# Chess AI Coach: Plan and Opportunity Audit

Reviewed: 2026-11-10. Repository baseline: `ef74010`.

This audits the plan proposed in the conversation and the current repository. The replacement implementation plan is [PLAN.md](PLAN.md). Scores below are judgments about the proposal, not measured product results or probabilities of business success.

## Verdict and grades

Build it as a portfolio project with a possible open-source future. The engine-and-language-model architecture is practical. The difficult part is proving that the explanations are correct and teach more effectively than existing tools. A custom model is an experiment whose value must be demonstrated.

| Dimension | Grade | Reason |
| --- | --- | --- |
| Problem usefulness | 8/10 | Understanding why a move works and why an alternative fails is a useful learning task. Users still need to demonstrate that this implementation helps them. |
| Original plan quality | 6.5/10 | Good architecture and milestones, but gaps in evidence, evaluation, training permissions, and compute feasibility prevent calling it implementation-ready. |
| Local MVP feasibility | 8/10 | A single Python app, python-chess, a persistent engine, and a replaceable explanation model are achievable for a solo developer. |
| Entire original roadmap feasibility at $0 | 5/10 | Manual labeling, model training, public inference, and reliable hosting impose constraints the original plan understates. |
| Portfolio potential | 9/10 | A working demo, reproducible benchmark, technical trade-offs, and honest results can demonstrate substantial engineering and ML work. The present prototype has not achieved this yet. |
| Open-source potential | 8/10 | An accessible coach plus a useful benchmark, audited examples, and reproducible tooling could attract contributors. Adoption is not automatic. |
| Startup case today | 4/10 | There is no evidence yet of differentiated quality, repeat use, willingness to pay, distribution, or sustainable operating economics. |

The grades distinguish potential from current readiness. The repository is still a console prompt prototype, not an operating coach.

## What the original plan gets right

- Use chess software for legal moves and numerical analysis, and the language model for explanations.
- Keep analysis records separate from explanations so labels can improve without rerunning the engine.
- Review the learner's side and select instructive moments rather than narrating every move.
- Preserve opening knowledge as its own evidence source.
- Use a hosted baseline before spending substantial effort on a custom model.
- Review every trusted training label, split by game, and evaluate the fine-tuned model against its untuned base.
- Build a local application before taking on public hosting and accounts.

These are sound choices. The corrections below refine them rather than requiring a different project.

## Findings, in priority order

### 1. Training-label permissions are an unresolved dependency

The plan selects Gemini as both the initial explanation writer and the source of training drafts. Google's terms restrict developing competing models; generated-output ownership alone does not settle this proposed use. They also impose audience, region, and unpaid-service conditions relevant to a later beta. Do not assume that editing a generated draft removes those restrictions. Use independently authored labels or a source explicitly permitting this training use until permissions are clear. This is a concrete dependency warning, not a legal conclusion about the project. [Gemini API terms](https://aistudio.google.com/docs/terms)

### 2. Free Colab training is not a guaranteed path

Colab's FAQ explicitly lists "chess training" among workloads restricted on free managed runtimes. Whether that covers supervised chess-commentary fine-tuning needs clarification; the plan cannot assume exemption. Free GPU availability is also variable. Keep fine-tuning off the MVP's critical path and use an explicitly permitted compute environment. Do not use Colab for bulk engine analysis or public inference. [Colab FAQ](https://research.google.com/colaboratory/faq.html)

The earlier observation that `nvidia-smi` was unavailable did not prove the computer has no GPU. The user confirmed 16 GB RAM; GPU and performance assumptions remain unverified.

### 3. Opening statistics cannot establish move correctness

An opening name classifies a position. Game frequency records what people played. Neither is proof that a move is sound, nor a source for an explanation of its strategic purpose. Win rates are observational and affected by player strength and selection.

Keep an engine-free opening information mode for names, sample counts, common moves, and clearly identified general ideas. Require engine or independently curated evidence before calling a move best, inferior, or a tactical mistake. This preserves the user's preference to avoid Stockfish for routine opening information while avoiding unsupported judgments. Exact-position matching, sample-size disclosure, transpositions, and sparse coverage must be handled. [Lichess Masters API schema](https://raw.githubusercontent.com/lichess-org/api/master/doc/specs/tags/openingexplorer/masters.yaml)

### 4. Three candidates can omit the very mistake being explained

MultiPV three normally returns the strongest moves, which may exclude the played move. The coach then lacks the opponent's best response to the user's actual choice.

Always include the played move's evaluated continuation. If it is absent from the candidate set, run an additional search restricted to that root move using the same engine and analysis profile. Keep each search's provenance. One persistent process does not mean one search can answer every comparison. `python-chess` supports restricting root moves. [Engine analysis API](https://python-chess.readthedocs.io/en/latest/engine.html)

### 5. A legal variation does not validate its prose explanation

"This opens the bishop's diagonal" can be false even when every move in the paragraph is legal. An attacked piece is not necessarily hanging; a pawn break is not necessarily good; material can be sacrificed intentionally.

Validate observable board facts and reference each concrete claim to a board state or variation. Treat longer-term strategic interpretation as interpretation. Human review is still required. Avoid promising fully verified prose or interpreting a single PV as the engine's unique causal explanation.

Also, the program cannot know the player's intent from a move alone. Write "this move develops the bishop" rather than claiming to know what the user was thinking.

### 6. Several good moves should be presented honestly

A tiny engine gap does not justify a long explanation of why the user's reasonable move was wrong. Candidate rankings can change with search budget. Show close alternatives as playable, record depth/nodes/time, and suppress categorical judgments when analysis is unstable. Strong human teaching includes explaining equivalent plans and uncertainty.

### 7. Evaluation gates are too precise for the proposed evidence

The original thresholds of 98% formatting success, 5% critical contradictions, and a 0.5-point gap to Gemini lack a defined scoring rubric or sample-size justification. Fifty held-out examples are a pilot, not proof of broad reliability. Even zero failures in 50 independent examples leaves an approximate 95% upper failure-rate bound near 6%; clustered chess examples make inference weaker.

Define errors before testing, report counts and confidence intervals where appropriate, and separate automatic checks from human correctness judgments. Use validation data for model selection; evaluate the test set only after choices are locked. Similar opening positions can leak across games, so add position-level deduplication and a separate opening-family challenge set.

### 8. Five hundred labels are an experiment, not a capability guarantee

With four themes, only about 100 training examples per theme remain after splitting. This may teach output structure and recurring explanation patterns; it does not establish broad chess understanding.

Audit 30 examples first to stabilize the format. Benchmark the untuned model before labeling all 500. Budget roughly 10-20 minutes per accepted example as a planning assumption: 500 examples require about 83-167 hours of review, excluding rejected drafts, tooling, and second opinions. Check the actual rate during the pilot.

The user can own every approval, but difficult positional claims need another qualified chess reviewer on at least a sample. Self-approval cannot independently establish teaching quality.

### 9. Latency and local serving need measurement

A 1-3 minute review is plausible, not established. Fifteen detailed generations, engine searches, API quotas, and cold starts can exceed it. A quantized model fitting in 16 GB RAM does not imply acceptable CPU inference speed. Measure engine, external-data, generation, and queue time separately.

Make the top three cards immediately visible while retaining up to fifteen for users who want depth. Benchmark generation batches against individual requests before locking batching. Add a local smoke test for model conversion before committing to a base model.

### 10. Deployment promises need narrower boundaries

A single-process queue is suitable locally. It loses in-flight work on restart and does not coordinate multiple web workers. State that limitation and make jobs fail visibly rather than poll forever. An unlisted URL is not access control; quotas and request limits are still necessary if it is shared publicly.

Removing PGN names reduces direct identification but does not guarantee anonymity: exact move sequences can be matched to public games. Describe this as removing identifying headers, restrict retained data, and keep training contribution opt-in.

## Repository evidence

Verified by reading the current files; no application changes were made for this audit.

| Location | Evidence | Implication |
| --- | --- | --- |
| `AI_process.py:6` | The coordinator loads analysis and then calls a phase function that loads it again. | Duplicated processing should be removed. |
| `chess_part.py:29`, `:75`, `:83` | Three separate engine contexts, two timed searches and one depth-20 PV search. | Six process starts would be attempted through the coordinator; best move and PV come from different searches. |
| `chess_part.py:31`, `:78` | Before/after results retain different `PovScore` viewpoints. | Normalize before calculating losses or teaching comparisons; the score objects do encode their viewpoints. |
| `chess_part.py:39`, `:105` | First ten plies of the entire PGN feed opening lookup. | Can include moves after the reviewed position and misrepresent its opening context. |
| `opening_agent.py:38` | Theory classification checks SAN membership in a list. | A move name occurring elsewhere does not establish theory at this position. |
| Phase-specific prompt functions | Print prompts without calling a model or returning explanations. | No completed coaching output exists. |
| `data/data_collection.py:5` | Loads all games, prints one sample, and outputs UCI. | No streaming dataset exporter; analyzer expects SAN. |
| Local `data/finetuning.json` | Six records, one empty answer; near-duplicate prompts include an answer about unrelated moves. | Preserve as untrusted prototypes outside training exports. |
| Tracked files and current interpreter | No tests, web application, training code, or deployment files; `python-chess` unavailable in the active Python. | Setup and end-to-end implementation remain. |

Six project Python files passed syntax parsing. That does not verify runtime execution, engine behavior, or network integrations.

## Usefulness and competition

The premise that existing products do not explain moves is too broad. Chess.com documents explanations of threats and piece-specific ideas with board annotations. DecodeChess explicitly offers Stockfish explanations, threats, plans, candidate moves, and explanations of the played move. These substantially overlap this project's core proposition. Product pages establish advertised features; this audit did not independently compare their quality. [Chess.com Game Review](https://www.chess.com/news/view/chesscom-launches-game-review-v2), [DecodeChess features](https://decodechess.com/features/), [DecodeChess played-move explanations](https://decodechess.com/new-feature-explaining-your-moves/)

That competition validates the task's relevance but raises the bar for differentiation. "Longer explanations" and "uses a fine-tuned model" are weak reasons for a player to switch.

The most credible initial positioning is: **an inspectable chess coach that compares the move you played with plausible alternatives, supports its claims with replayable lines, and publishes its explanation-quality evaluation.** This is a hypothesis to test, not an achieved advantage.

Usefulness depends on whether learners can explain the lesson and apply it elsewhere. Test a small exercise after a report and a related position later. Keep detailed cards available, but watch whether users finish reading them.

## Portfolio, open-source, and business paths

### Portfolio: highest-confidence payoff

The strongest evidence for a recruiter would be a runnable demo, readable architecture, meaningful failure tests, a dataset with documented provenance, blind baseline comparisons, and measured latency. A fine-tuned model is valuable if its improvement or limitations are measured. An honest negative result can still demonstrate strong ML engineering.

Suggested project title: **Engine-grounded Chess Coach with Reproducible Explanation Evaluation**.

After implementation, a resume bullet can report actual results: "Built a PGN review tool combining Stockfish candidate analysis and board features with structured coaching; evaluated on [N] held-out positions with [measured result]." Do not claim ratings improvement, adoption, or a trained model before measuring them.

Planning estimate, not a commitment: roughly 60-120 focused hours for a narrow local demo and initial evaluation, plus the label-review time above. Training and publishing add work and external dependencies. Stop after the local milestone if it already meets the portfolio goal.

### Open source: credible with reusable artifacts

Start with reproducible setup, sample games, a documented analysis format, an evaluation harness, and a small set of reviewed examples that others can improve. A useful explanation benchmark or local coaching engine could matter even if the consumer website stays small.

Before release, document licenses for the app, dependencies, model, datasets, and board assets. `python-chess` is GPL-3.0-or-later; Stockfish also has distribution obligations. Do not assume MIT branding covers the complete package. [python-chess repository](https://github.com/niklasf/python-chess), [Stockfish license guidance](https://stockfishchess.org/about/)

### Startup: possible, with little supporting evidence today

A plausible later niche is helping chess coaches prepare annotated reviews for their students. It offers a clearer workflow and buyer than another general chess review site, but this is only a research hypothesis. Do not add those features to the portfolio MVP.

Before investing in a business, seek the following signals in sequence. These are proposed decision gates, not industry standards:

1. Interview ten target players and three coaches about recent review frustrations and existing spending.
2. Run a blind comparison on the same positions; seek at least 60% preference over the best practical baseline without a higher serious-error rate.
3. Invite 15-20 testers, observe repeated reviews over four weeks, and record why they return or stop. A useful early signal is at least five people returning voluntarily in week four.
4. Ask for a concrete paid pilot; three paying customers provide more evidence than many expressions of interest.
5. Measure cost per completed review, support time, and acquisition effort. Charge enough to cover more than just inference.

Failure at a gate is information: narrow the audience or keep the project as a portfolio/open-source contribution. No revenue, market-size, or venture-scale claim is justified by the present evidence. A custom model is not a durable moat by itself; a rights-cleared corpus, better outcomes, distribution, and a trusted workflow could become advantages.

## Audit limits

This was a static code and document audit plus primary-source research. It did not execute Stockfish, compare paid competitors hands-on, train models, benchmark the hardware, interview users, or obtain legal advice. It created documentation only. The revised plan reflects these limits rather than treating unknowns as established facts.
