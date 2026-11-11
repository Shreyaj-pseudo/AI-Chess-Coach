# Chess AI Coach

A local-first chess coaching project. T01 establishes a reproducible Miniconda environment and portable Stockfish configuration. The existing analysis scripts are prototypes: a working web coach and model inference come in later tasks.

Roadmap: [audited plan](docs/PLAN.md), [daily agent workflow](docs/DAILY_WORKFLOW.md), [detailed task backlog](docs/tasks.json).

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

Runtime dependencies use the maintained `chess` distribution (the same `import chess` API), Requests, and python-dotenv. The unused `notopenai` requirement was removed after checking its callers. Training libraries and model downloads belong to later tasks, in a separate environment.

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

These are recorded baseline issues, not claims that T01 fixes them. T02 starts strict PGN parsing and replay.
