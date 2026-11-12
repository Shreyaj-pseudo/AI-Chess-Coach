"""Seven-piece standard endgame evidence, qualified by the actual rule context."""

from datetime import datetime, timezone
import json
import math
import time
from typing import Literal

import chess
from pydantic import Field, model_validator
import requests

from analysis_records import Color, Text, UCI, ValidatedModel, VersionedRecord
import lichess_http


ENDPOINT = "https://tablebase.lichess.org/standard"
MAX_PIECES = 7
MAX_RESPONSE_BYTES = 256 * 1024
Category = Literal["win", "unknown", "syzygy-win", "maybe-win", "cursed-win", "draw",
                   "blessed-loss", "maybe-loss", "syzygy-loss", "loss"]
Result = Literal["win", "draw", "loss", "unknown"]
INVERT = dict(zip(
    ("win", "unknown", "syzygy-win", "maybe-win", "cursed-win", "draw", "blessed-loss", "maybe-loss", "syzygy-loss", "loss"),
    ("loss", "unknown", "syzygy-loss", "maybe-loss", "blessed-loss", "draw", "cursed-win", "maybe-win", "syzygy-win", "win"),
))


class RuleContext(ValidatedModel):
    side_to_move: Color
    halfmove_clock: int = Field(ge=0)
    repetition_count: int = Field(ge=1, le=5)
    repetition_in_reversible_history: bool
    can_claim_fifty_moves: bool
    can_claim_threefold_repetition: bool
    termination: Text | None = None


class TablebaseValue(ValidatedModel):
    source_category: Category
    source_pov: Color
    pov: Color
    category: Category
    outcome: Result
    dtz_plies: int | None = None
    precise_dtz_plies: int | None = None
    dtm_plies: int | None = None
    rule_dependent: bool


class TablebaseMove(ValidatedModel):
    uci: UCI
    san: Text
    fen_after: Text
    zeroing: bool
    rules: RuleContext
    value: TablebaseValue


def _validated_board(board):
    if not isinstance(board, chess.Board) or not board.is_valid() or board.chess960 or board.uci_variant != "chess":
        raise ValueError("Tablebase probing requires a valid standard chess board.")
    replay = board.root()
    for move in board.move_stack:
        if replay.is_game_over() or move not in replay.legal_moves:
            raise ValueError("Tablebase history contains an illegal move or continues after game over.")
        replay.push(move)
    if replay.fen() != board.fen():
        raise ValueError("Tablebase board disagrees with its replayed history.")
    return replay


def _rules(board):
    prior = board.copy(stack=True)
    seen = {" ".join(prior.fen().split()[:4])}
    repeated = False
    while prior.move_stack:
        move = prior.pop()
        if prior.is_irreversible(move):
            break
        key = " ".join(prior.fen().split()[:4])
        if key in seen:
            repeated = True
            break
        seen.add(key)
    outcome = board.outcome(claim_draw=False)
    return RuleContext(
        side_to_move="white" if board.turn else "black", halfmove_clock=board.halfmove_clock,
        repetition_count=next((n for n in range(5, 1, -1) if board.is_repetition(n)), 1),
        repetition_in_reversible_history=repeated, can_claim_fifty_moves=board.can_claim_fifty_moves(),
        can_claim_threefold_repetition=board.can_claim_threefold_repetition(),
        termination=outcome.termination.name.lower() if outcome else None,
    )


def _local_result(board, learner):
    outcome = board.outcome(claim_draw=False)
    if outcome is None:
        return "unknown"
    return "draw" if outcome.winner is None else "win" if outcome.winner == (learner == "white") else "loss"


class TablebaseContext(VersionedRecord):
    initial_fen: Text
    history_uci: tuple[UCI, ...] = Field(strict=False)
    fen: Text
    learner_color: Color
    rules: RuleContext
    status: Literal["ok", "rule_dependent", "terminal", "ineligible", "unavailable", "rate_limited"]
    outcome: Result
    engine_fallback: bool
    value: TablebaseValue | None = None
    moves: tuple[TablebaseMove, ...] = Field(default=(), max_length=256, strict=False)
    endpoint: Literal["https://tablebase.lichess.org/standard"] = ENDPOINT
    query_fen: str | None = None
    fetched_at: str | None = None
    retry_after_seconds: int | None = Field(default=None, ge=0)
    reason: Text
    notes: tuple[Text, ...] = Field(strict=False)

    @model_validator(mode="after")
    def verify_context(self):
        board = chess.Board(self.initial_fen)
        if not board.is_valid() or board.fen() != self.initial_fen:
            raise ValueError("Invalid tablebase initial board.")
        for uci in self.history_uci:
            move = chess.Move.from_uci(uci)
            if board.is_game_over() or move not in board.legal_moves:
                raise ValueError("Invalid tablebase history.")
            board.push(move)
        if board.fen() != self.fen or _rules(board) != self.rules:
            raise ValueError("Tablebase rule context does not match its board/history.")
        if self.query_fen is not None and self.query_fen != self.fen:
            raise ValueError("Tablebase query must preserve the actual FEN and halfmove clock.")
        if self.rules.termination:
            if (self.status != "terminal" or self.engine_fallback or self.value is not None or self.moves
                    or self.query_fen is not None or self.outcome != _local_result(board, self.learner_color)):
                raise ValueError("Terminal tablebase context must resolve locally without further searches.")
        elif self.value is None:
            if self.status not in ("ineligible", "unavailable", "rate_limited") or self.outcome != "unknown" or not self.engine_fallback or self.moves:
                raise ValueError("Unavailable tablebase evidence must request an explicit engine fallback.")
        else:
            expected_status = "rule_dependent" if self.value.rule_dependent else "ok"
            expected_fallback = self.outcome == "unknown" or any(m.value.outcome == "unknown" for m in self.moves)
            if (self.status != expected_status or self.query_fen != self.fen or self.fetched_at is None
                    or self.outcome != self.value.outcome or self.engine_fallback != expected_fallback):
                raise ValueError("Tablebase status/result has inconsistent evidence or fallback state.")
            _verify_value(self.value, board, self.learner_color, self.rules)
            if len({m.uci for m in self.moves}) != len(self.moves) or {m.uci for m in self.moves} != {m.uci() for m in board.legal_moves}:
                raise ValueError("Tablebase moves must cover each legal root move exactly once.")
            for item in self.moves:
                move = chess.Move.from_uci(item.uci)
                child = board.copy(stack=True)
                child.push(move)
                if (item.san != board.san(move) or item.fen_after != child.fen() or item.rules != _rules(child)
                        or item.zeroing != board.is_zeroing(move)):
                    raise ValueError("Tablebase move facts disagree with the replayed board.")
                _verify_value(item.value, child, self.learner_color, item.rules)
            if self.outcome != "unknown":
                ranking = {"loss": -1, "draw": 0, "win": 1}
                actor = 1 if board.turn == (self.learner_color == "white") else -1
                if any(not m.value.rule_dependent and m.value.outcome != "unknown"
                       and actor * ranking[m.value.outcome] > actor * ranking[self.outcome] for m in self.moves):
                    raise ValueError("Tablebase root verdict contradicts an available better move outcome.")
            if not self.value.rule_dependent and all(not m.value.rule_dependent and m.value.outcome != "unknown" for m in self.moves):
                ordering = {"loss": -1, "draw": 0, "win": 1}
                choose = max if board.turn == (self.learner_color == "white") else min
                expected = choose(ordering[m.value.outcome] for m in self.moves)
                if ordering[self.outcome] != expected:
                    raise ValueError("Tablebase root outcome disagrees with its legal move outcomes.")
        return self


def _value(raw, board, learner, rules):
    category = raw["category"]
    if category not in INVERT:
        raise ValueError("Unrecognized tablebase category.")
    for flag, actual in (("checkmate", board.is_checkmate()), ("stalemate", board.is_stalemate()),
                         ("insufficient_material", board.is_insufficient_material())):
        if flag in raw and (type(raw[flag]) is not bool or raw[flag] != actual):
            raise ValueError(f"Tablebase {flag} flag disagrees with the actual board.")
    for flag in ("variant_win", "variant_loss"):
        if flag in raw and raw[flag] is not False:
            raise ValueError("Variant outcome in a standard-chess tablebase response.")
    sign = 1 if board.turn == (learner == "white") else -1
    normalized = category if sign == 1 else INVERT[category]
    metrics = {key: raw.get(source) for key, source in (
        ("dtz_plies", "dtz"), ("precise_dtz_plies", "precise_dtz"), ("dtm_plies", "dtm"))}
    if any(value is not None and type(value) is not int for value in metrics.values()):
        raise ValueError("Tablebase distances must be integer plies or null.")
    if metrics["precise_dtz_plies"] is not None and metrics["precise_dtz_plies"] != metrics["dtz_plies"]:
        raise ValueError("Precise and rounded DTZ metadata disagree.")
    for distance in metrics.values():
        if distance is not None and (
            (category == "draw" and distance != 0)
            or (category in ("win", "cursed-win", "syzygy-win", "maybe-win") and distance < 0)
            or (category in ("loss", "blessed-loss", "syzygy-loss", "maybe-loss") and distance > 0)
        ):
            raise ValueError("Tablebase distance sign disagrees with its source category.")
    local = board.outcome(claim_draw=False)
    history_sensitive = (rules.repetition_in_reversible_history or rules.can_claim_fifty_moves or rules.can_claim_threefold_repetition)
    dependent = category not in ("win", "draw", "loss") or history_sensitive
    if local:
        dependent = False
        result = _local_result(board, learner)
        if local.winner is None and category != "draw":
            raise ValueError("Tablebase category disagrees with an automatic draw.")
        if local.termination == chess.Termination.CHECKMATE and category not in ("loss", "blessed-loss"):
            raise ValueError("Tablebase category disagrees with checkmate.")
    elif normalized == "draw":
        result = "draw"
    elif history_sensitive:
        result = "unknown"
    elif normalized in ("cursed-win", "blessed-loss"):
        result = "draw"
    elif normalized in ("win", "loss"):
        result = normalized
    else:
        result = "unknown"
    return TablebaseValue(source_category=category, source_pov="white" if board.turn else "black",
                          pov=learner, category=normalized, outcome=result, rule_dependent=dependent,
                          **{key: value * sign if value is not None else None for key, value in metrics.items()})


def _verify_value(value, board, learner, rules):
    sign = 1 if board.turn == (learner == "white") else -1
    raw = {"category": value.source_category, "dtz": value.dtz_plies * sign if value.dtz_plies is not None else None,
           "precise_dtz": value.precise_dtz_plies * sign if value.precise_dtz_plies is not None else None,
           "dtm": value.dtm_plies * sign if value.dtm_plies is not None else None}
    if value != _value(raw, board, learner, rules):
        raise ValueError("Tablebase value disagrees with its actual rule context or learner perspective.")


def _parse(raw, board, learner, rules):
    root = _value(raw, board, learner, rules)
    moves = []
    for item in raw["moves"]:
        move = chess.Move.from_uci(item["uci"])
        if move not in board.legal_moves or board.san(move) != item["san"]:
            raise ValueError("Tablebase move is illegal or has mismatched SAN.")
        zeroing = board.is_zeroing(move)
        if "zeroing" in item and (type(item["zeroing"]) is not bool or item["zeroing"] != zeroing):
            raise ValueError("Tablebase zeroing flag disagrees with the move.")
        child = board.copy(stack=True)
        child.push(move)
        child_rules = _rules(child)
        moves.append(TablebaseMove(uci=move.uci(), san=item["san"], fen_after=child.fen(), zeroing=zeroing,
                                   rules=child_rules, value=_value(item, child, learner, child_rules)))
    if len({m.uci for m in moves}) != len(moves) or {m.uci for m in moves} != {m.uci() for m in board.legal_moves}:
        raise ValueError("Tablebase response must contain every legal root move exactly once.")
    return root, tuple(moves)


class TablebaseClient:
    def __init__(self):
        self._session = requests.Session()
        self._session.auth = lambda request: request
        self._closed = False

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Tablebase client is closed.")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        with lichess_http.REQUEST_LOCK:
            if not self._closed:
                self._session.close()
                self._closed = True

    def probe(self, board: chess.Board, learner_color: Color) -> TablebaseContext:
        if learner_color not in ("white", "black"):
            raise ValueError("Choose white or black as the tablebase learner.")
        board = _validated_board(board)
        rules = _rules(board)
        base = dict(initial_fen=board.root().fen(), history_uci=tuple(m.uci() for m in board.move_stack),
                    fen=board.fen(), learner_color=learner_color, rules=rules,
                    notes=("DTZ is distance to zeroing in plies, not mate distance; DTZ may be rounded.",
                           "Categories and signed distances are normalized to the learner; source categories retain their own side-to-move POV.",
                           "Provider FEN includes the halfmove clock, but not repetition history; history-sensitive decisive results are withheld."))
        with lichess_http.REQUEST_LOCK:
            if self._closed:
                raise RuntimeError("Tablebase client is closed.")
            if rules.termination:
                return TablebaseContext(**base, status="terminal", outcome=_local_result(board, learner_color), engine_fallback=False,
                                        reason="Game already ended; resolved locally without HTTP or engine search.")
            if len(board.piece_map()) > MAX_PIECES or board.castling_rights:
                return TablebaseContext(**base, status="ineligible", outcome="unknown", engine_fallback=True,
                                        reason="Use engine analysis: tablebase requires at most seven pieces and no castling rights.")
            remaining = lichess_http.COOLDOWN_UNTIL - time.monotonic()
            if remaining > 0:
                return TablebaseContext(**base, status="rate_limited", outcome="unknown", engine_fallback=True,
                                        retry_after_seconds=math.ceil(remaining), reason="Lichess cooldown active; use engine analysis.")
            response = None
            try:
                response = self._session.get(ENDPOINT, params={"fen": board.fen(), "dtc": "never"},
                                             timeout=(3.0, 5.0), stream=True, allow_redirects=False,
                                             headers={"Accept": "application/json", "User-Agent": "ChessCoach/0.1 (endgame analysis)"})
                if response.status_code == 429:
                    # shortcut: tablebase-only cooldowns are process-local; persist with restartable review jobs in T13.
                    retry = lichess_http.retry_seconds(response.headers.get("Retry-After"))
                    lichess_http.COOLDOWN_UNTIL = time.monotonic() + retry
                    return TablebaseContext(**base, status="rate_limited", outcome="unknown", engine_fallback=True,
                                            query_fen=board.fen(), retry_after_seconds=retry, reason="HTTP 429; use engine analysis while requests pause.")
                if response.status_code != 200:
                    raise ValueError(f"Tablebase service returned HTTP {response.status_code}.")
                chunks, size = [], 0
                for chunk in response.iter_content(chunk_size=16384):
                    size += len(chunk)
                    if size > MAX_RESPONSE_BYTES:
                        raise ValueError("Tablebase response exceeds 256 KiB.")
                    chunks.append(chunk)
                value, moves = _parse(json.loads(b"".join(chunks)), board, learner_color, rules)
                status = "rule_dependent" if value.rule_dependent else "ok"
                fallback = value.outcome == "unknown" or any(m.value.outcome == "unknown" for m in moves)
                return TablebaseContext(**base, status=status, outcome=value.outcome, engine_fallback=fallback,
                                        value=value, moves=moves, query_fen=board.fen(), fetched_at=datetime.now(timezone.utc).isoformat(),
                                        reason="Use engine analysis for unknown outcomes; retain available tablebase evidence." if fallback
                                        else "Tablebase evidence available; respect rule qualifiers and metric units.")
            except (requests.RequestException, ValueError, KeyError, TypeError) as error:
                return TablebaseContext(**base, status="unavailable", outcome="unknown", engine_fallback=True,
                                        query_fen=board.fen(), reason=f"Use engine analysis: tablebase unavailable ({error}).")
            finally:
                if response is not None:
                    response.close()
