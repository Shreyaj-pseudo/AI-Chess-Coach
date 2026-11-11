# Chess AI Coach

A local-first chess coaching project. T01 establishes a reproducible Miniconda environment and portable Stockfish configuration. The existing analysis scripts are prototypes: a working web coach and model inference come in later tasks.

Roadmap: [audited plan](docs/PLAN.md), [daily agent workflow](docs/DAILY_WORKFLOW.md), [detailed task backlog](docs/tasks.json).

## Try the PGN ingestion pipeline (T02)

After activating the environment below, inspect a sample or replace its path with your own PGN:

```powershell
python -m scripts.inspect_pgn tests/fixtures/pgn/castling.pgn --color white
python -m scripts.inspect_pgn tests/fixtures/pgn/black_setup.pgn --color black
```

This displays each chosen player's pre-move FEN, SAN/UCI played move, relative ply and the final board FEN. It runs locally without Stockfish or any API. The first coaching output is T12; the upload/report website is T14-T15, and the tested local-demo milestone is T17-T18.

The shared API is:

```python
from pathlib import Path
from pgn_input import learner_positions, parse_pgn

game = parse_pgn(Path("tests/fixtures/pgn/castling.pgn").read_bytes())
positions = learner_positions(game, "white")
first = positions[0]
print(first.fen, first.san, first.uci)
print(first.pgn_prefix)
```

`parse_pgn` accepts UTF-8 text/bytes, including an optional BOM, up to 256 KiB and 200 played plies. It rejects unknown mainline text, illegal/null moves, variants, malformed/duplicate headers, multiple games, moves after terminal positions and inconsistent setup/result data. A nonstandard starting board needs both `SetUp "1"` and a valid six-field `FEN`. Blank lines outside comments/variations separate games; do not insert them within a mainline. A legal unfinished line may omit the final result marker. A result-only game is accepted and produces no review decisions.

`learner_positions` requires White or Black explicitly. Each returned board is an independent copy with its prior move stack, preserving repetition history; `ply` counts half-moves from this PGN's starting position, including custom setups. `pgn_prefix` contains only moves before the reviewed move and necessary setup headers. It excludes identity/event headers, comments, annotations, variations, the reviewed move and later moves. Use that prefix for future model context, not the original uploaded PGN or untrusted game headers.

Comments and side variations are structurally checked and discarded; their move legality is not analyzed. Check/capture markers and explicit results are checked against the mainline. Numeric annotations are discarded. The original seven small fixtures under `tests/fixtures/pgn/` cover both colors, castling, en passant, promotion, custom starts, comments, checkmate and repetition. The legacy opening-move helper now uses this shared strict parser; its opening-classification logic is still deferred to T07.

## Analysis records and consistent scores (T03)

`analysis_records.py` defines four versioned, frozen [Pydantic](https://docs.pydantic.dev/latest/concepts/models/) records:

| Record | Purpose |
| --- | --- |
| `Candidate` | One legal move/PV, its normalized score, root FEN and engine-search provenance |
| `AnalysisRecord` | A history-backed position, up to three candidates, and separate evidence for the move actually played |
| `CoachingCard` | Detailed explanation fields linked to an analysis ID and exact candidate/ply references |
| `TrainingExample` | Analysis plus card, revision, draft/approved/rejected status, reviewer and label-source permissions metadata |

Each has `schema_version=1` and an SHA-256 content `id`. Identical validated content produces the same ID; edits produce a new ID. Imported IDs must match their content. JSON roundtrips use `Model.model_dump_json()` and `Model.model_validate_json(...)`; schemas are available through `Model.model_json_schema()`. Unknown fields, incorrect scalar types and unsupported versions fail validation. Tuples and frozen nested models prevent accidental edits. Create validated replacements without the old `id` when revising content; do not use `model_construct()` or `model_copy(update=...)` to bypass validation.

Convert a T02 decision into a serializable position without losing its initial board or repetition history:

```python
from analysis_records import PositionContext, source_game_id
from pgn_input import parse_pgn, learner_positions

game = parse_pgn("1. e4 e5 2. Nf3 *")
decision = learner_positions(game, "white")[1]
position = PositionContext.from_decision(decision, game_id=source_game_id(game))
print(position.model_dump_json(indent=2))
```

The game ID hashes the initial board and complete move sequence, excluding player names/comments. The position stores only prior moves plus the played move, not future movetext. Rebuilding `position.board()` creates a fresh board with its history; FEN/ply/side and the played move must agree with replay.

Scores are explicitly one of:

- `CentipawnScore(kind="cp", pov="white" or "black", cp=...)`.
- `MateScore(kind="mate", pov=..., moves=..., winning=True or False)`; mate-in-zero retains whether this side delivered mate or was mated.
- `UnavailableScore(kind="unavailable", pov=..., reason=...)`; unavailable is never treated as zero.

`normalize_score(engine_pov_score, learner_color)` uses python-chess's point-of-view conversion. `centipawn_loss(best, played)` rejects mixed viewpoints and returns a difference only when both are centipawn scores. For example, a Black-perspective best score of +70 cp and an after-move White-perspective score of -20 cp normalize to +70/+20 for Black, yielding a 50 cp loss. This is an illustrative calculation, not an engine result for a real position. Mate/unavailable comparisons return `None`, and negative gaps are preserved instead of clamped.

`SearchProvenance` requires an engine/version string, a SHA-256 of the engine options/search-budget configuration, the actual search root FEN, at least one positive limit, and achieved depth/nodes/elapsed time. `root_moves` records restrictions separately. `Candidate.from_engine(board, info, learner_color=..., search=...)` derives SAN and normalized score from the same supplied PV result. The caller supplies truthful search metadata; a schema cannot prove that a declared search really ran.

`AnalysisRecord` replays every PV against the full position history, checks legal/unique roots, played-move identity and learner-oriented scores. Contradictory rankings/scores or mixed engine configurations require an explicit uncertainty note or reanalysis. Its `loss_cp` property is derived from the first candidate and played evidence. Derived `pv_san`/`loss_cp` properties are not duplicated as mutable JSON fields.

Call `card.validate_for(analysis)` before using a standalone card; `TrainingExample` calls it automatically. It rejects foreign analysis IDs, unsupported recommendations, missing candidate references and out-of-range line plies. A line reference at ply zero refers to its root position. Training examples default to `draft`; `approved` requires reviewer metadata. Source-license/permission strings and reviewer names are declarations, not authenticated approval, legal clearance or proof of correct coaching. Human review/export gates remain later tasks.

T03 adds the shared data contract and conversion helpers. The legacy phase scripts still print their old raw-score prompts; migrating engine ownership/results and the complete explanation pipeline is T04-T12. Opening/tablebase evidence, board-feature extraction, full semantic claim checks and training/export workflows are not implemented by these schemas.

## Set up with Miniconda on Windows

Run from the repository root in a Miniconda PowerShell prompt:

```powershell
conda env create --prefix .\.venv --file environment.yml --yes
conda activate .\.venv
python --version
python -m pip check
python -m scripts.check_setup
```

`environment.yml` pins Python 3.12.16 and pip 26.2.2 from conda-forge, then installs the pinned runtime packages in `requirements.txt`. The `.venv` directory is a **Conda environment**, despite its conventional directory name. It is gitignored. Your Miniconda base environment is not the project runtime.

If `.venv` already exists, activate it and install the current pins instead of recreating or deleting it:

```powershell
conda activate .\.venv
python -m pip install -r requirements.txt
```

When activation is inconvenient, use the same environment explicitly:

```powershell
conda run --prefix .\.venv python -m scripts.check_setup
conda run --prefix .\.venv python -m unittest discover -s tests -v
```

In VS Code, select `.venv\python.exe` as the Python interpreter. If `conda` is not recognized, use Miniconda's own prompt; PowerShell integration can be enabled with `conda init powershell`, followed by reopening the shell. This project does not change your shell profile automatically.

Runtime dependencies use the maintained `chess` distribution (the same `import chess` API), Requests, python-dotenv and Pydantic. The unused `notopenai` requirement was removed after checking its callers. Training libraries and model downloads belong to later tasks, in a separate environment.

The environment specifies tested Python/pip/runtime versions, not a bit-for-bit cross-platform system image. Conda may select different native builds on another OS; record those and rerun setup/tests there.

## Install and configure Stockfish

Download an official binary compatible with your OS/CPU from [Stockfish downloads](https://stockfishchess.org/download/). Extract it locally; this repository ignores the `stockfish/` directory. For the existing Windows download, the executable is:

```text
stockfish/stockfish-windows-x86-64-avx2.exe
```

Stockfish discovery follows this order:

1. Explicit `ChessAnalyzer(stockfish_path=...)` argument.
2. `STOCKFISH_PATH` from your shell or the repository `.env` (shell values win).
3. A `stockfish` executable on PATH.
4. The existing local Windows AVX2 executable, or `stockfish/stockfish` on other systems.

Relative configured paths resolve from the repository root, not the shell's current directory. A configured path pointing to a missing file or directory fails with a setup message instead of silently selecting another engine. A file existing on disk does not prove that it is a compatible UCI engine; run the smoke check below.

No configuration is necessary if the local download works. Otherwise, add this non-secret setting to your existing `.env`, or set it in PowerShell:

```powershell
$env:STOCKFISH_PATH = 'C:\tools\stockfish\stockfish.exe'
```

Use `.env.example` as a template only when you do not already have a `.env`; do not overwrite existing keys. Leave `STOCKFISH_PATH` blank for automatic discovery. API variable names in the example are reserved for future provider integration. No API key is needed for setup or these tests.

Start the actual engine and check that it returns a legal move:

```powershell
python -m scripts.check_setup --engine
```

This performs a short UCI search and closes the engine. It makes no hosted-model or Lichess request. If the executable fails to start, check its path, CPU instruction requirements, operating-system compatibility and permissions. A passing smoke check is not validation of the full analysis pipeline.

Stockfish is GPLv3; preserve the corresponding source/license information if distributing its binary. [Official license guidance](https://stockfishchess.org/about/)

## Tests and task execution

```powershell
python -m unittest discover -s tests -p test_T01_environment.py -v
python -m unittest discover -s tests -p test_T02_pgn.py -v
python -m unittest discover -s tests -p test_T03_records.py -v
python -m unittest discover -s tests -v
python scripts/workflow.py status --track mvp
python scripts/workflow.py prompt --track mvp
```

Run tests from the activated project environment. T01 tests require Python 3.12 and the exact runtime pins. Offline tests use temporary files and mocked engine discovery; the real engine smoke is a separate command. In a restricted coding-agent sandbox, temporary-directory or executable access may need approval. Do not skip or report those blocked tests as passing.

Task evidence and logs are in gitignored `artifacts/tasks/` and `.workflow/`. Keep those folders if you want to retain local progress. The workflow guide explains verification, completion and the human review gates.

## Known prototype limitations

- Phase functions print prompts and return no generated coaching explanation; no model is called.
- The coordinator repeats analysis, and each load starts multiple engine processes; persistent ownership is T04.
- Best-move, played-move and principal-variation searches have different budgets. Scores are not normalized before comparison.
- Opening identification uses the first ten plies and SAN membership instead of the exact reviewed position; HTTP calls lack timeouts.
- Phase detection uses piece count, and piece lists accumulate on analyzer reuse.
- The collector loads all games before sampling and prints UCI, while the analyzer expects SAN.
- No website, reliable whole-game coach, approved training corpus, trained model or deployment exists yet. Existing six draft labels remain untrusted.

These are recorded baseline issues. T01 provides setup; T02 provides strict PGN parsing and replay; T03 provides the shared records and score normalization. Engine ownership, evidence collection, coaching and the website remain later tasks.
