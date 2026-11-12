"""Exact-position opening statistics; popularity is not a move-quality verdict."""

from contextlib import closing, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from typing import Annotated, Literal

import chess
from pydantic import Field, StringConstraints, model_validator
import requests
from dotenv import load_dotenv

from analysis_records import PositionContext, Text, UCI, ValidatedModel, VersionedRecord
import lichess_http
from lichess_http import REQUEST_LOCK as _REQUEST_LOCK, retry_seconds as _retry_seconds


MAX_FULL_MOVES = 20
SPARSE_SAMPLE = 20
CACHE_TTL = 24 * 60 * 60
CACHE_ROWS = 512
MAX_RESPONSE_BYTES = 256 * 1024
DEFAULT_CACHE = Path(__file__).resolve().parent / "artifacts" / "opening_cache.sqlite3"
BASE_URL = "https://explorer.lichess.org"
SOURCES = ("masters", "lichess")
QueryPair = Annotated[tuple[Text, Text], Field(strict=False)]


class OpeningName(ValidatedModel):
    eco: Annotated[str, StringConstraints(pattern=r"^[A-E][0-9]{2}$")]
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class GameCounts(ValidatedModel):
    white: int = Field(ge=0)
    draws: int = Field(ge=0)
    black: int = Field(ge=0)
    total_games: int = Field(ge=0)

    @model_validator(mode="after")
    def verify_total(self):
        if self.total_games != self.white + self.draws + self.black:
            raise ValueError("Opening sample total disagrees with its outcome counts.")
        return self


class OpeningMove(GameCounts):
    uci: UCI
    san: Text
    total_games: int = Field(ge=1)


class OpeningStats(GameCounts):
    opening: OpeningName | None = None
    moves: tuple[OpeningMove, ...] = Field(max_length=12, strict=False)

    def validate_for(self, board):
        if len({m.uci for m in self.moves}) != len(self.moves):
            raise ValueError("Opening response contains duplicate moves.")
        for move in self.moves:
            uci = chess.Move.from_uci(move.uci)
            if uci not in board.legal_moves or board.san(uci) != move.san:
                raise ValueError("Opening response contains an illegal move or mismatched SAN for this position.")
        for outcome in ("white", "draws", "black"):
            if sum(getattr(m, outcome) for m in self.moves) > getattr(self, outcome):
                raise ValueError("Listed move counts exceed the position's sample.")
        return self


class OpeningSource(ValidatedModel):
    source: Literal["masters", "lichess"]
    endpoint: Text
    query: tuple[QueryPair, ...] = Field(strict=False)
    status: Literal["ok", "sparse", "unavailable", "rate_limited", "outside_window"]
    data: OpeningStats | None = None
    played_games: int | None = Field(default=None, ge=1)
    fetched_at: str | None = None
    cache_hit: bool = False
    retry_after_seconds: int | None = Field(default=None, ge=0)
    error: Text | None = None


class HistoricalOpening(ValidatedModel):
    source: Literal["masters", "lichess"]
    opening: OpeningName
    position_key: Text
    position_ply: int = Field(ge=1)
    fetched_at: Text


class OpeningContext(VersionedRecord):
    position: PositionContext
    sources: tuple[OpeningSource, OpeningSource] = Field(strict=False)
    historical: HistoricalOpening | None = None
    max_full_moves: int = MAX_FULL_MOVES
    sparse_sample_games: int = SPARSE_SAMPLE
    cache_ttl_seconds: int = CACHE_TTL
    notes: tuple[Text, ...] = Field(strict=False)


def position_key(board):
    """Database position identity: pieces, turn, castling and legally relevant EP."""
    return " ".join(board.fen().split()[:4])


def _query(source, board):
    params = {"fen": position_key(board) + " 0 1", "moves": "12", "topGames": "0"}
    if source == "masters":
        params["since"] = "1952"
    else:
        params.update(variant="standard", speeds="rapid,classical", ratings="1200,1400,1600",
                      recentGames="0", history="false")
    return params


def _cache_key(source, board):
    return hashlib.sha256(json.dumps([f"{BASE_URL}/{source}", _query(source, board)], sort_keys=True).encode()).hexdigest()


def _counts(raw):
    counts = {key: raw[key] for key in ("white", "draws", "black")}
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ValueError("Opening outcome counts must be nonnegative integers.")
    return {**counts, "total_games": sum(counts.values())}


def _parse(raw, board):
    moves = tuple(OpeningMove(uci=m["uci"], san=m["san"], **_counts(m)) for m in raw["moves"])
    # Move-level opening names describe subsequent positions, not the reviewed root.
    return OpeningStats(**_counts(raw), opening=raw.get("opening"), moves=moves).validate_for(board)


def _utc(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


class OpeningExplorer:
    """Synchronous client for one local worker; HTTP is serialized process-wide."""

    def __init__(self, cache_path=DEFAULT_CACHE, *, token=None):
        if token is None:
            load_dotenv(Path(__file__).resolve().parent / ".env", override=False)
            token = os.getenv("LICHESS_API_TOKEN", "")
        if not isinstance(token, str) or not token.isascii() or any(c.isspace() for c in token.strip()):
            raise ValueError("LICHESS_API_TOKEN must be a single ASCII token; do not include a Bearer prefix.")
        self._token = token.strip()
        self.cache_path = Path(cache_path)
        self._session = requests.Session()
        # Keep ambient .netrc credentials from overriding explicit bearer authentication.
        self._session.auth = lambda request: request
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Opening explorer is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        with _REQUEST_LOCK:
            if not self._closed:
                self._session.close()
                self._closed = True

    @contextmanager
    def _db(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.cache_path, timeout=1.0)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS opening_cache (key TEXT PRIMARY KEY, fetched_at REAL NOT NULL, payload TEXT NOT NULL)")
            db.execute("CREATE TABLE IF NOT EXISTS opening_metadata (key TEXT PRIMARY KEY, value REAL NOT NULL)")
            yield db

    def _cached(self, db, source, board, notes):
        key = _cache_key(source, board)
        row = db.execute("SELECT fetched_at, payload FROM opening_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            fetched, payload = row
            if not 0 <= time.time() - fetched < CACHE_TTL:
                raise ValueError("expired timestamp")
            return OpeningStats.model_validate_json(payload).validate_for(board), fetched
        except (ValueError, TypeError, OverflowError):
            db.execute("DELETE FROM opening_cache WHERE key = ?", (key,))
            notes.append("An expired or invalid opening cache entry was ignored.")
            return None

    def _source(self, db, source, board, played, notes, in_window):
        base = dict(source=source, endpoint=f"{BASE_URL}/{source}", query=tuple(sorted(_query(source, board).items())))
        if not in_window:
            return OpeningSource(**base, status="outside_window", error="Opening HTTP queries stop after full move 20.")
        cached = self._cached(db, source, board, notes)
        if cached:
            data, fetched = cached
        else:
            remaining = lichess_http.COOLDOWN_UNTIL - time.monotonic()
            if remaining > 0:
                return OpeningSource(**base, status="rate_limited", retry_after_seconds=math.ceil(remaining),
                                     error="Lichess cooldown active; no request sent.")
            response = None
            try:
                headers = {"Accept": "application/json", "User-Agent": "ChessCoach/0.1 (local opening analysis)"}
                if self._token:
                    headers["Authorization"] = f"Bearer {self._token}"
                response = self._session.get(base["endpoint"], params=_query(source, board), timeout=(3.0, 5.0),
                                             stream=True, allow_redirects=False,
                                             headers=headers)
                if response.status_code == 429:
                    retry = _retry_seconds(response.headers.get("Retry-After"))
                    lichess_http.COOLDOWN_UNTIL = time.monotonic() + retry
                    db.execute("INSERT OR REPLACE INTO opening_metadata VALUES ('cooldown_until', ?)", (time.time() + retry,))
                    return OpeningSource(**base, status="rate_limited", retry_after_seconds=retry,
                                         error="HTTP 429; all explorer requests paused for at least one minute.")
                if response.status_code != 200:
                    message = f"Opening service returned HTTP {response.status_code}."
                    if response.status_code in (401, 403):
                        message += " Set/check LICHESS_API_TOKEN in the environment or existing project .env."
                    return OpeningSource(**base, status="unavailable", error=message)
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=16384):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise ValueError("Opening response exceeds 256 KiB.")
                    chunks.append(chunk)
                data = _parse(json.loads(b"".join(chunks)), board)
                fetched = time.time()
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                return OpeningSource(**base, status="unavailable", error=f"Opening lookup failed: {error}")
            finally:
                if response is not None:
                    response.close()
            db.execute("INSERT OR REPLACE INTO opening_cache VALUES (?, ?, ?)",
                       (_cache_key(source, board), fetched, data.model_dump_json()))
            db.execute("DELETE FROM opening_cache WHERE fetched_at < ?", (time.time() - CACHE_TTL,))
            db.execute("DELETE FROM opening_cache WHERE key IN (SELECT key FROM opening_cache ORDER BY fetched_at DESC LIMIT -1 OFFSET ?)", (CACHE_ROWS,))
        observed = next((move.total_games for move in data.moves if move.uci == played), None)
        return OpeningSource(**base, status="sparse" if data.total_games < SPARSE_SAMPLE else "ok", data=data,
                             played_games=observed, fetched_at=_utc(fetched), cache_hit=bool(cached))

    def _historical(self, db, board, notes):
        prior = board.copy(stack=True)
        while prior.move_stack:
            prior.pop()
            if prior.fullmove_number > MAX_FULL_MOVES:
                continue
            for source in SOURCES:
                cached = self._cached(db, source, prior, notes)
                if cached and cached[0].opening is not None:
                    return HistoricalOpening(source=source, opening=cached[0].opening, position_key=position_key(prior),
                                             position_ply=len(prior.move_stack) + 1, fetched_at=_utc(cached[1]))
        return None

    def lookup(self, position: PositionContext) -> OpeningContext:
        position = PositionContext.model_validate(position)
        board = position.board()
        notes = ["Opening names and game frequencies are descriptive, not proof of a best move or a mistake.",
                 "Lichess filters: rated rapid/classical, rating buckets 1200/1400/1600 (1200 <= rating < 1800).",
                 "Only the 12 most common moves are requested; an unlisted played move has unknown frequency."]
        with _REQUEST_LOCK:
            if self._closed:
                raise RuntimeError("Opening explorer is closed.")
            try:
                with self._db() as db:
                    saved = db.execute("SELECT value FROM opening_metadata WHERE key = 'cooldown_until'").fetchone()
                    if saved and isinstance(saved[0], (int, float)) and math.isfinite(saved[0]):
                        lichess_http.COOLDOWN_UNTIL = max(lichess_http.COOLDOWN_UNTIL, time.monotonic() + max(0, saved[0] - time.time()))
                    sources = tuple(self._source(db, s, board, position.played_uci, notes,
                                                 board.fullmove_number <= MAX_FULL_MOVES) for s in SOURCES)
                    historical = None if any(s.data and s.data.opening for s in sources) else self._historical(db, board, notes)
            except (OSError, sqlite3.Error) as error:
                sources = tuple(OpeningSource(source=s, endpoint=f"{BASE_URL}/{s}", query=tuple(sorted(_query(s, board).items())),
                                              status="unavailable", error=f"Opening cache unavailable: {error}") for s in SOURCES)
                historical = None
            return OpeningContext(position=position, sources=sources, historical=historical, notes=tuple(dict.fromkeys(notes)))
