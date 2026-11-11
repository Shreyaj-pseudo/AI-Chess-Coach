"""Find Stockfish without depending on one developer's filesystem."""

import os
from pathlib import Path
import shutil

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_BINARY = PROJECT_ROOT / "stockfish" / (
    "stockfish-windows-x86-64-avx2.exe" if os.name == "nt" else "stockfish"
)


def resolve_stockfish_path(stockfish_path=None):
    """Explicit path, project .env/environment, PATH, then the local download."""
    if stockfish_path is None:
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        stockfish_path = os.getenv("STOCKFISH_PATH")
    if stockfish_path:
        candidate = Path(stockfish_path).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
    else:
        candidate = Path(shutil.which("stockfish") or LOCAL_BINARY)
    if not candidate.is_file():
        raise FileNotFoundError(
            f"Stockfish executable not found: {candidate}. "
            "Install Stockfish on PATH or set STOCKFISH_PATH to its executable in the project .env."
        )
    if not os.access(candidate, os.X_OK):
        raise PermissionError(f"Stockfish is not executable: {candidate}. Check its execute permissions.")
    return str(candidate.resolve())
